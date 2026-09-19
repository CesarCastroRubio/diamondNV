#!/usr/bin/env python3
import argparse
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree

MOLAR_MASS_H2O = 18.01528
AVOGADRO = 6.02214076e23
ANGSTROM3_TO_CM3 = 1e-24
C_SP3 = 4
LATTICE_CONSTANT = 3.567
NM_TO_ANGSTROM = 10.0

ATOMIC_MASS = {"H": 1.008, "B": 10.811, "C": 12.011, "N": 14.007, "O": 15.999,
               "F": 18.998, "Na": 22.990, "Si": 28.086, "P": 30.974, "S": 32.06,
               "Cl": 35.45, "K": 39.098, "Br": 79.904, "I": 126.904}
VDW_RADIUS = {"H": 1.20, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52,
              "F": 1.47, "Na": 2.27, "Si": 2.10, "P": 1.80, "S": 1.80,
              "Cl": 1.75, "K": 2.75, "Br": 1.85, "I": 1.98}
LABEL_TO_ELEMENT = {"C": "C", "CS": "C", "N": "N",
                    "O": "O", "OS": "O", "OW": "O",
                    "H": "H", "HW": "H"}
CLASH_KEYS = {"H": "clash_oh", "O": "clash_oo", "OS": "clash_oos"}

FCC_FRAC = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
DIAMOND_BASIS = [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]

def miller_family(n_nonzero):
    return np.array([p for p in itertools.product((1, -1, 0), repeat=3)
                     if sum(abs(c) for c in p) == n_nonzero], float)


MILLER_100, MILLER_110, MILLER_111 = (miller_family(n) for n in (1, 2, 3))
RATIO_OCTAHEDRON = np.sqrt(3.0)
RATIO_CUBE = 1.0 / np.sqrt(3.0)
LATTICE_LABELS = ("C", "CS", "N", "OS")
SOLVENT_LABELS = ("OW", "HW")


def die(msg):
    sys.stderr.write("\nERROR: " + msg.rstrip() + "\n\n")
    sys.exit(2)


def unit(v):
    return v / np.linalg.norm(v)


def normalize_element(symbol):
    el = symbol.capitalize() if len(symbol) > 1 else symbol.upper()
    return el if el in ATOMIC_MASS else None


def label_element(label):
    if label in LABEL_TO_ELEMENT:
        return LABEL_TO_ELEMENT[label]
    return normalize_element(label.split("_")[0])


def sphere_volume(r):
    return (4.0 / 3.0) * np.pi * r ** 3


def rotation_to(src, dst):
    a, b = unit(np.asarray(src, float)), unit(np.asarray(dst, float))
    c = float(np.dot(a, b))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        axis, theta = unit(np.cross(a, ref)), np.pi
    else:
        axis, theta = unit(np.cross(a, b)), np.arccos(c)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def wulff_planes(ratio, ratio_110):
    normals = np.vstack([MILLER_111 / np.sqrt(3.0), MILLER_100,
                         MILLER_110 / np.sqrt(2.0)])
    offsets = np.concatenate([np.ones(len(MILLER_111)),
                              np.full(len(MILLER_100), ratio),
                              np.full(len(MILLER_110), ratio_110)])
    return normals, offsets


def wulff_unit_volume(ratio, ratio_110):
    normals, offsets = wulff_planes(ratio, ratio_110)
    halfspaces = np.hstack([normals, -offsets[:, None]])
    hull = ConvexHull(HalfspaceIntersection(halfspaces, np.zeros(3)).intersections)
    return float(hull.volume)


def wulff_radii(r_angstrom, ratio, ratio_110):
    scale = (sphere_volume(r_angstrom) / wulff_unit_volume(ratio, ratio_110)) ** (1 / 3)
    return scale, scale * ratio, scale * ratio_110


def size_metrics(labeled, origin, lattice_only=False):
    keep = [a for a in labeled
            if (a[0] in LATTICE_LABELS if lattice_only else
                a[0] not in SOLVENT_LABELS and "_" not in a[0])]
    if len(keep) < 4:
        return None
    xyz = np.array([a[1:] for a in keep], float)
    m = np.array([ATOMIC_MASS[label_element(a[0])] for a in keep])
    com = m @ xyz / m.sum()
    d = xyz - com
    evals = np.linalg.eigvalsh((d * m[:, None]).T @ d / m.sum())[::-1]
    rg = float(np.sqrt(evals.sum()))
    r = np.linalg.norm(xyz - origin, axis=1)
    hull = ConvexHull(xyz)
    return {
        "n_atoms": len(keep),
        "rg_angstrom": rg,
        "d_equiv_rg_nm": 2.0 * np.sqrt(5.0 / 3.0) * rg / NM_TO_ANGSTROM,
        "r_max_angstrom": float(r.max()),
        "r_inscribed_angstrom": float(-(hull.equations[:, 3]
                                        + hull.equations[:, :3] @ origin).max()),
        "com_offset_angstrom": float(np.linalg.norm(com - origin)),
        "gyration_eigenvalues": [float(v) for v in evals],
    }


class DiamondCoreGenerator:
    def __init__(self, a_angstrom=LATTICE_CONSTANT, shape="sphere", wulff_ratio=RATIO_OCTAHEDRON,
                 wulff_ratio_110=RATIO_OCTAHEDRON, prune_cutoff=None):
        self.a = float(a_angstrom)
        self.shape = shape
        self.wulff_ratio = float(wulff_ratio)
        self.wulff_ratio_110 = float(wulff_ratio_110)
        self.metrics = {}
        self.prune_cutoff = prune_cutoff
        self.n_pruned = 0
        self.n_wrapped = 0
        self._pos_cache = {}
        self._surface_idx = set()
        self._replacements = {}
        self._extra_atoms = []
        self._nv_choice = None

    @property
    def _vacancy(self):
        return np.zeros(3)

    def facet_radii(self, r_angstrom):
        if self.shape == "sphere":
            return None
        return wulff_radii(float(r_angstrom), self.wulff_ratio, self.wulff_ratio_110)

    def _keep_mask(self, xyz, r_angstrom, cell):
        if self.shape == "sphere":
            r = float(r_angstrom)
            return np.einsum("ij,ij->i", xyz, xyz) < r * r + 1e-9
        r111 = self.facet_radii(r_angstrom)[0]
        normals, offsets = wulff_planes(self.wulff_ratio, self.wulff_ratio_110)
        normals = np.array([unit(n @ cell) for n in normals])
        return np.all(xyz @ normals.T <= (offsets * r111) + 1e-9, axis=1)

    def _bounding_radius(self, r_angstrom):
        if self.shape == "sphere":
            return float(r_angstrom)
        r111, r100, r110 = self.facet_radii(r_angstrom)
        return min(np.sqrt(3.0) * r111, np.sqrt(3.0) * r100, np.sqrt(2.0) * r110)

    def _prune_undercoordinated(self, xyz):
        if not self.prune_cutoff or not len(xyz):
            return xyz
        n_before = len(xyz)
        while len(xyz):
            pairs = cKDTree(xyz).query_pairs(self.prune_cutoff, output_type="ndarray")
            ncoord = np.bincount(pairs.ravel(), minlength=len(xyz)) if len(pairs) \
                else np.zeros(len(xyz), int)
            keep = ncoord > 1
            if keep.all():
                break
            xyz = xyz[keep]
        self.n_pruned = n_before - len(xyz)
        return xyz

    def positions(self, r_angstrom, rotate_111_to_z=False):
        key = (round(float(r_angstrom), 10), bool(rotate_111_to_z))
        cached = self._pos_cache.get(key)
        if cached is not None:
            return cached

        a = self.a
        n = int(np.ceil((self._bounding_radius(r_angstrom) + a * np.sqrt(3)) / a))
        cell = np.eye(3) * a
        if rotate_111_to_z:
            cell = cell @ rotation_to([1, 1, 1], [0, 0, 1]).T

        rng = np.arange(-n, n + 1)
        cells = np.array(list(itertools.product(rng, rng, rng)), float)
        motif = np.array([np.add(f, b) for f in FCC_FRAC for b in DIAMOND_BASIS])
        frac = (cells[:, None, :] + motif[None, :, :]).reshape(-1, 3)
        xyz = frac @ cell + self._vacancy
        xyz = xyz[self._keep_mask(xyz, r_angstrom, cell)]
        xyz = np.array(sorted({tuple(np.round(p, 10)) for p in xyz}))
        xyz = self._prune_undercoordinated(xyz)

        atoms = [tuple(p) for p in xyz]
        self._pos_cache[key] = atoms
        return atoms

    def to_xyz(self, r_angstrom, box_length, rotate_111_to_z=False, nv_vacancy=True,
               provenance="", wrap=False):
        atoms = self.positions(r_angstrom, rotate_111_to_z)
        syms = [self._replacements.get(i, "CS" if i in self._surface_idx else "C")
                for i in range(len(atoms))]
        xyz = np.array(atoms, float) if atoms else np.zeros((0, 3))

        if nv_vacancy and len(xyz):
            d = np.linalg.norm(xyz - self._vacancy, axis=1)
            keep = d > 1e-4 * np.sqrt(3)
            xyz, syms, d = xyz[keep], [s for s, k in zip(syms, keep) if k], d[keep]
            cand = np.flatnonzero(np.abs(d - self.a * np.sqrt(3) / 4) < 0.15 * self.a)
            if len(cand):
                if self._nv_choice is None:
                    self._nv_choice = random.randint(0, len(cand) - 1)
                syms[int(cand[self._nv_choice])] = "N"

        labeled = [(s, *p) for s, p in zip(syms, xyz)] + list(self._extra_atoms)

        self.metrics = {"particle": size_metrics(labeled, self._vacancy),
                        "core": size_metrics(labeled, self._vacancy, lattice_only=True)}
        mp = self.metrics["particle"]

        L = float(box_length)
        if wrap and labeled:
            pos = np.array([a[1:] for a in labeled], float)
            shifted = pos - L * np.floor(pos / L + 0.5)
            self.n_wrapped = int((np.abs(shifted - pos) > 1e-9).any(axis=1).sum())
            labeled = [(a[0], *p) for a, p in zip(labeled, shifted)]
        header = (
            f'Lattice="{L:.6f} 0.0 0.0  0.0 {L:.6f} 0.0  0.0 0.0 {L:.6f}" '
            f'Origin="{-L / 2:.6f} {-L / 2:.6f} {-L / 2:.6f}" '
            f'pbc="T T T" Properties=species:S:1:pos:R:3 '
            f'lattice_constant={self.a:.4f} diameter_nm={2.0 * r_angstrom / 10:.4f}'
        )
        if mp:
            header += (f' rg_angstrom={mp["rg_angstrom"]:.4f}'
                       f' d_equiv_rg_nm={mp["d_equiv_rg_nm"]:.4f}'
                       f' r_max_angstrom={mp["r_max_angstrom"]:.4f}'
                       f' r_inscribed_angstrom={mp["r_inscribed_angstrom"]:.4f}')
        if provenance:
            header += " " + provenance
        lines = [str(len(labeled)), header]
        lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, x, y, z in labeled]
        return "\n".join(lines) + "\n"


def surface_analysis(gen, r_angstrom, rotate_111_to_z, bond_cc, bond_tol):
    coords = np.array(gen.positions(r_angstrom, rotate_111_to_z=rotate_111_to_z))
    tree = cKDTree(coords)
    bonded = tree.query_ball_tree(tree, bond_cc + bond_tol)
    ncoord = np.array([len(neigh) - 1 for neigh in bonded])
    return coords, bonded, ncoord, np.where(ncoord < C_SP3)[0]


def resolve_targets(cfg, n_total, n_under, r_angstrom):
    if cfg.termination in ("bare", "h"):
        return 0, 0

    if cfg.n_oh is not None:
        target_OH = int(cfg.n_oh)
    elif cfg.termination == "oh":
        target_OH = n_under
    else:
        target_OH = max(3, int(cfg.oh_base_fraction * n_total *
                               (cfg.oh_scale_ref / r_angstrom)))
    target_OH = min(target_OH, n_under)

    if cfg.n_o is not None:
        target_O = int(cfg.n_o)
    elif cfg.termination == "oh" or cfg.oh_o_ratio <= 0:
        target_O = 0
    else:
        target_O = int(target_OH / cfg.oh_o_ratio)
    return target_OH, target_O


def _missing_directions(neigh_vecs, rvec, cfg):
    n = len(neigh_vecs)
    if n == 3:
        normal = np.cross(neigh_vecs[0] - neigh_vecs[1], neigh_vecs[0] - neigh_vecs[2])
        if np.linalg.norm(normal) < 1e-6:
            normal = -np.sum(neigh_vecs, axis=0)
        normal = unit(normal)
        return [normal if np.dot(normal, rvec) >= 0 else -normal]
    if n == 2:
        u, v = neigh_vecs
        cross, bis = unit(np.cross(u, v)), unit(u + v)
        theta = np.deg2rad(cfg.angle_hch / 2)
        dirs = [-np.cos(theta) * bis + np.sin(theta) * cross,
                -np.cos(theta) * bis - np.sin(theta) * cross]
        return dirs if np.dot(np.mean(dirs, axis=0), rvec) >= 0 else [-d for d in dirs]
    if n == 1:
        a = neigh_vecs[0]
        tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) <= 0.9 else np.array([0.0, 1.0, 0.0])
        x = unit(np.cross(a, tmp))
        y = np.cross(a, x)
        theta = np.deg2rad(cfg.angle_hch)
        dirs = [-np.cos(theta) * a + np.sin(theta) *
                (np.cos(p) * x + np.sin(p) * y)
                for p in np.deg2rad([0, 120, 240])]
        return dirs if np.dot(np.mean(dirs, axis=0), rvec) >= 0 else [-d for d in dirs]
    return []


def functionalize_surface(gen, r_angstrom, cfg):
    coords, bonded, ncoord, under_idx = surface_analysis(
        gen, r_angstrom, cfg.rotate_111, cfg.bond_cc_cut, cfg.bond_tol
    )
    gen._surface_idx = {int(i) for i in under_idx}
    n_total, n_under = len(coords), len(under_idx)

    if cfg.termination == "bare":
        if not cfg.quiet:
            print(f"Bare surface: {n_under} undercoordinated sites left untouched.")
        return gen, {"n_OH": 0, "n_O": 0, "n_H": 0, "target_OH": 0, "target_O": 0,
                     "n_undercoordinated": int(n_under), "n_core": int(n_total)}

    target_OH, target_O = resolve_targets(cfg, n_total, n_under, r_angstrom)

    ex_sym = ["C"] * n_total
    ex_xyz = [tuple(c) for c in coords]
    new_atoms = []

    def add(sym, pos):
        ex_sym.append(sym)
        ex_xyz.append(tuple(pos))

    chosen_O = []
    if target_O > 0:
        two_coord = [int(i) for i in under_idx if ncoord[i] == 2]
        random.shuffle(two_coord)
        blocked = set()
        for idx in two_coord:
            if len(chosen_O) >= target_O:
                break
            neigh = [n for n in bonded[idx] if n != idx]
            if blocked.intersection(neigh):
                continue
            chosen_O.append(idx)
            blocked.update(neigh)
        for idx in chosen_O:
            gen._replacements[idx] = "OS"
            ex_sym[idx] = "OS"

    count_H = 0
    for idx in map(int, under_idx):
        if idx in gen._replacements:
            continue
        neigh = [n for n in bonded[idx] if n != idx]
        if not 1 <= len(neigh) <= 3:
            continue
        pos = coords[idx]
        vecs = coords[neigh] - pos
        norms = np.linalg.norm(vecs, axis=1)
        vecs = vecs[norms > 1e-6] / norms[norms > 1e-6, None]
        if len(vecs) != len(neigh):
            continue
        for d in _missing_directions(vecs, unit(pos), cfg):
            H_pos = pos + cfg.bond_ch * unit(d)
            new_atoms.append(("H", *H_pos))
            add("H", H_pos)
            count_H += 1

    H_indices = [i for i, (s, *_) in enumerate(new_atoms) if s == "H"]
    random.shuffle(H_indices)
    center = coords.mean(axis=0)
    phis = np.deg2rad(np.arange(0, 360, 10))
    count_OH = 0
    visited_C = set()

    for hi in H_indices:
        if count_OH >= target_OH:
            break
        H_xyz = np.array(new_atoms[hi][1:])
        nearest_C = int(np.argmin(np.linalg.norm(coords - H_xyz, axis=1)))
        if nearest_C in visited_C:
            continue

        C_xyz = coords[nearest_C]
        CH_vec = H_xyz - C_xyz
        if np.linalg.norm(CH_vec) < 1e-6:
            continue
        CH_vec = unit(CH_vec)
        O_pos = C_xyz + cfg.bond_co * CH_vec

        pts = np.array(ex_xyz)
        d_all = np.linalg.norm(pts - O_pos, axis=1)
        clash = False
        for sym, key in CLASH_KEYS.items():
            sel = np.fromiter((s == sym for s in ex_sym), bool, len(ex_sym))
            sel &= np.linalg.norm(pts - H_xyz, axis=1) > 1e-3
            if np.any(d_all[sel] < getattr(cfg, key)):
                clash = True
                break
        if clash:
            continue
        visited_C.add(nearest_C)

        theta = np.deg2rad(cfg.angle_coh)
        tmp = np.random.randn(3)
        x = np.cross(CH_vec, tmp)
        if np.linalg.norm(x) < 1e-6:
            x = np.cross(CH_vec, np.array([1.0, 0, 0]))
        x = unit(x)
        y = np.cross(CH_vec, x)

        dirs = (np.cos(theta) * (-CH_vec) +
                np.sin(theta) * (np.cos(phis)[:, None] * x + np.sin(phis)[:, None] * y))
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        ok = dirs @ (O_pos - center) >= -0.5
        if not np.any(ok):
            continue
        dirs = dirs[ok]
        cands = O_pos + cfg.bond_oh * dirs
        best_H = cands[cKDTree(pts).query(cands)[0].argmax()]

        new_atoms[hi] = ("O", *O_pos)
        new_atoms.append(("H", *best_H))
        add("O", O_pos)
        add("H", best_H)
        count_OH += 1

    gen._extra_atoms = list(gen._extra_atoms) + new_atoms
    counts = {
        "n_OH": int(count_OH),
        "n_O": int(len(chosen_O)),
        "n_H": int(count_H - count_OH),
        "n_undercoordinated": int(n_under),
        "n_core": int(n_total),
        "target_OH": int(target_OH),
        "target_O": int(target_O),
    }
    if not cfg.quiet:
        print(f"Inserted {len(chosen_O)} bridging O groups, {count_OH} OH groups, and "
              f"{count_H - count_OH} hydrogens out of {n_under} possible "
              f"undercoordinated sites.")
    return gen, counts


def print_termination(cfg, counts):
    n_O, n_OH, n_H = counts["n_O"], counts["n_OH"], counts["n_H"]
    n_C = counts["n_core"] - n_O - (2 if cfg.nv_vacancy else 0)
    formula = f"C{n_C}"
    if n_H:
        formula += f" H{n_H}"
    if n_OH:
        formula += f" (OH){n_OH}"
    if n_O:
        formula += f" O{n_O}"
    if cfg.nv_vacancy:
        formula += " N"

    if cfg.termination == "bare":
        detail = f"no adatoms; {counts['n_undercoordinated']} sites left open"
    elif cfg.termination == "h":
        detail = f"{n_H} H on {counts['n_undercoordinated']} undercoordinated sites"
    else:
        ratio = f"{n_OH / n_O:.2f}" if n_O else "inf"
        detail = (f"OH:O requested {cfg.oh_o_ratio:g}, realized {ratio}; "
                  f"{n_OH} OH, {n_O} bridging O, {n_H} H on "
                  f"{counts['n_undercoordinated']} undercoordinated sites")
    print(f"Termination [{cfg.termination}]: {detail}")
    print(f"Core composition: {formula}")


def n_water_to_volume(n_water, density):
    return (n_water * MOLAR_MASS_H2O / (AVOGADRO * density)) / ANGSTROM3_TO_CM3


def volume_to_n_water(volume_A3, density):
    return volume_A3 * ANGSTROM3_TO_CM3 * AVOGADRO * density / MOLAR_MASS_H2O


def box_from_water_count(n_water, r_angstrom, gap, density, v_add=0.0):
    v_water = n_water_to_volume(n_water, density)
    v_excl = sphere_volume(r_angstrom + gap)
    return float((v_water + v_excl + v_add) ** (1.0 / 3.0)), v_water, v_excl


def min_water_count(r_angstrom, gap, density, v_add=0.0):
    r_excl = r_angstrom + gap
    return int(np.ceil(volume_to_n_water(
        (2.0 * r_excl) ** 3 - sphere_volume(r_excl) - v_add, density)))


def water_count_from_box(box_length, r_angstrom, gap, density, v_add=0.0):
    v_excl = sphere_volume(r_angstrom + gap)
    v_free = box_length ** 3 - v_excl - v_add
    return int(round(volume_to_n_water(v_free, density))), v_free, v_excl


def effective_density(n_water, box_length, r_angstrom, gap, v_add=0.0):
    v_free = box_length ** 3 - sphere_volume(r_angstrom + gap) - v_add
    if v_free <= 0:
        return float("nan")
    return n_water * MOLAR_MASS_H2O / (AVOGADRO * v_free * ANGSTROM3_TO_CM3)


def system_composition(xyz_text):
    counts = {}
    for line in xyz_text.splitlines()[2:]:
        parts = line.split()
        if not parts:
            continue
        el = label_element(parts[0])
        if el is None:
            die(f"unknown atom label '{parts[0]}' while totalling the system mass.")
        counts[el] = counts.get(el, 0) + 1
    return counts, sum(ATOMIC_MASS[el] * n for el, n in counts.items())


def hill_formula(counts):
    counts = {el: n for el, n in counts.items() if n}
    order = [el for el in ("C", "H") if el in counts]
    order += sorted(el for el in counts if el not in ("C", "H"))
    return "".join(f"{el}{counts[el]}" if counts[el] > 1 else el for el in order)


def system_density(mass_amu, box_length):
    return mass_amu / (AVOGADRO * box_length ** 3 * ANGSTROM3_TO_CM3)


def write_text(path, text):
    with open(path, "w") as f:
        f.write(text)


def ensure_water_template(path, quiet=False):
    if not os.path.exists(path):
        write_text(path, "3\nWater molecule\n"
                         "O  0.000  0.000  0.000\n"
                         "H  0.757  0.586  0.000\n"
                         "H -0.757  0.586  0.000\n")
        if not quiet:
            print(f"Wrote water template {path}")
    return path


def read_xyz(path):
    try:
        lines = open(path).read().splitlines()
    except OSError as exc:
        die(f"cannot read structure template '{path}': {exc}")
    try:
        n = int(lines[0].split()[0])
    except (IndexError, ValueError):
        die(f"'{path}' does not start with an atom count; it is not an XYZ file.")
    atoms = []
    for line in lines[2:2 + n]:
        p = line.split()
        if len(p) >= 4:
            atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    if len(atoms) != n:
        die(f"'{path}' declares {n} atoms but only {len(atoms)} coordinate lines parse.")
    return atoms


def flatten_list_arg(tokens):
    if tokens is None:
        return None
    items = [t.strip("[](),") for t in re.split(r"[,\s]+", " ".join(tokens))]
    return [t for t in items if t]


def parse_int_list(tokens, flag):
    values = []
    for t in flatten_list_arg(tokens) or []:
        try:
            values.append(int(t))
        except ValueError:
            die(f"{flag} got '{t}', which is not an integer.")
    return values


def additive_tag(path):
    return os.path.splitext(os.path.basename(path))[0]


def tagged_label(symbol, tag):
    el = normalize_element(symbol)
    if el is None:
        die(f"unknown element '{symbol}' in additive '{tag}'; add it to ATOMIC_MASS.")
    return f"{el}_{tag}"


def molecule_volume(atoms):
    return sum(sphere_volume(VDW_RADIUS[normalize_element(s)]) for s, *_ in atoms)


def resolve_additives(cfg):
    paths = flatten_list_arg(cfg.additives) or []
    counts = parse_int_list(cfg.n_additives, "--n-additives")
    if not paths:
        if counts:
            die("--n-additives was given without --additives.")
        return []
    if not counts:
        die("--additives was given without --n-additives; state a count per file.")
    if len(paths) != len(counts):
        die(f"--additives lists {len(paths)} files but --n-additives lists "
            f"{len(counts)} counts; they must match one-to-one.")

    specs = []
    for path, n in zip(paths, counts):
        if n < 0:
            die(f"--n-additives entry for '{path}' is negative.")
        tag = additive_tag(path)
        if any(s["tag"] == tag for s in specs):
            die(f"two --additives entries share the basename '{tag}'; atom labels "
                f"would collide. Rename one of the files.")
        atoms = read_xyz(path)
        xyz = np.array([a[1:] for a in atoms])
        specs.append({
            "tag": tag,
            "path": os.path.abspath(path),
            "n": int(n),
            "n_atoms": len(atoms),
            "labels": [tagged_label(s, tag) for s, *_ in atoms],
            "formula": hill_formula({normalize_element(s): sum(
                normalize_element(t) == normalize_element(s) for t, *_ in atoms)
                for s, *_ in atoms}),
            "volume_angstrom3": molecule_volume(atoms),
            "mass_amu": sum(ATOMIC_MASS[normalize_element(s)] for s, *_ in atoms),
            "radius_angstrom": float(np.linalg.norm(xyz - xyz.mean(axis=0), axis=1).max()),
        })
    return [s for s in specs if s["n"] > 0]


def additive_volume(specs):
    return sum(s["n"] * s["volume_angstrom3"] for s in specs)


def run_packmol(cfg, inp_path, text, out_path, what):
    exe = shutil.which(cfg.packmol_exe)
    if exe is None:
        die(f"packmol executable '{cfg.packmol_exe}' not found on PATH.\n"
            f"Install packmol or point at it with --packmol-exe /path/to/packmol,\n"
            f"or run with --n-water 0 --box <edge_angstrom> for a dry structure.")
    write_text(inp_path, text)
    if os.path.exists(out_path):
        os.remove(out_path)
    with open(inp_path) as inp:
        proc = subprocess.run([exe], stdin=inp, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-25:])
        die(f"packmol failed (exit {proc.returncode}) while packing {what} "
            f"({inp_path}).\n--- packmol output (tail) ---\n{tail}\n"
            f"{(proc.stderr or '').strip()}")
    if cfg.verbose:
        print(proc.stdout)
    if not os.path.exists(out_path):
        die(f"packmol reported success but produced no {out_path}.")
    return read_xyz(out_path)


def packmol_header(cfg, box_length, out_path, pbc=True):
    half = box_length / 2.0
    head = (f"tolerance {cfg.packmol_tolerance}\n"
            f"seed {cfg.seed}\n"
            f"filetype xyz\n"
            f"output {out_path}\n")
    if pbc:
        head += (f"pbc {-half:.6f} {-half:.6f} {-half:.6f} "
                 f"{half:.6f} {half:.6f} {half:.6f}\n")
    return head + "\n"


def pack_additives(cfg, r_excl, box_length, specs):
    half = box_length / 2.0 - cfg.additive_margin
    if half <= r_excl:
        die(f"--box leaves no room for additives: the {2 * r_excl:.2f} angstrom "
            f"exclusion sphere fills the cell once the {cfg.additive_margin:g} "
            f"angstrom additive margin is applied.")
    box = (f"  inside box {-half:.6f} {-half:.6f} {-half:.6f} "
           f"{half:.6f} {half:.6f} {half:.6f}\n")
    body = "".join(
        f"structure {s['path']}\n"
        f"  number {s['n']}\n"
        f"{box}"
        f"  outside sphere 0.0 0.0 0.0 {r_excl:.6f}\n"
        f"end structure\n\n"
        for s in specs
    )
    atoms = run_packmol(cfg, cfg.additive_inp,
                        packmol_header(cfg, box_length, cfg.additive_out, pbc=False) + body,
                        cfg.additive_out, "additives")

    expected = sum(s["n"] * s["n_atoms"] for s in specs)
    if len(atoms) != expected:
        die(f"packmol wrote {len(atoms)} additive atoms but {expected} were requested.")

    overflow = np.abs(np.array([a[1:] for a in atoms])).max() - box_length / 2.0
    if overflow > 0:
        die(f"packmol pushed an additive atom {overflow:.3f} angstrom outside the "
            f"cell; the water pass would reject it as a fixed structure.\n"
            f"Raise --additive-margin above {cfg.additive_margin:g} angstrom, or "
            f"enlarge --box.")

    labeled, i = [], 0
    for s in specs:
        for _ in range(s["n"]):
            for label in s["labels"]:
                labeled.append((label, *atoms[i][1:]))
                i += 1
    if not cfg.quiet:
        summary = ", ".join(f"{s['n']} x {s['tag']}" for s in specs)
        print(f"Packed additives ({summary}) into {len(labeled)} atoms")
    return labeled


def pack_water(cfg, r_excl, box_length, additive_atoms):
    water_xyz = ensure_water_template(cfg.water_xyz, cfg.quiet)
    body = (f"structure {water_xyz}\n"
            f"  number {cfg.n_water}\n"
            f"  outside sphere 0.0 0.0 0.0 {r_excl:.6f}\n"
            f"  radius {cfg.water_radius}\n"
            f"end structure\n\n")
    if additive_atoms:
        body += (f"structure {cfg.additive_out}\n"
                 f"  number 1\n"
                 f"  fixed 0. 0. 0. 0. 0. 0.\n"
                 f"end structure\n\n")

    atoms = run_packmol(cfg, cfg.packmol_inp,
                        packmol_header(cfg, box_length, cfg.packmol_out) + body,
                        cfg.packmol_out, "water")

    n_expected = 3 * cfg.n_water
    if len(atoms) != n_expected + len(additive_atoms):
        die(f"packmol wrote {len(atoms)} atoms but {n_expected} water atoms plus "
            f"{len(additive_atoms)} fixed additive atoms were expected.")

    relabel = {"O": "OW", "H": "HW"}
    water_atoms = [(relabel.get(s, s), x, y, z) for s, x, y, z in atoms[:n_expected]]
    if any(s not in ("OW", "HW") for s, *_ in water_atoms):
        die("the packmol water block did not come back as pure water; refusing to "
            "label a structure whose manifest would be wrong.")
    return water_atoms


def add_solvent_shell(gen, r_angstrom, cfg, box_length, specs):
    r_excl = r_angstrom + cfg.water_gap
    additive_atoms = pack_additives(cfg, r_excl, box_length, specs) if specs else []
    water_atoms = pack_water(cfg, r_excl, box_length, additive_atoms) if cfg.n_water else []

    if not cfg.quiet and cfg.n_water:
        free_v = box_length ** 3 - sphere_volume(r_excl) - additive_volume(specs)
        print(f"Packed {cfg.n_water} water molecules ({len(water_atoms)} atoms) into "
              f"{free_v:.2f} cubic angstrom of free volume")

    gen._extra_atoms = list(gen._extra_atoms) + additive_atoms + water_atoms
    return gen, cfg.n_water, additive_atoms


def build_parser():
    p = argparse.ArgumentParser(
        prog="diamond.py",
        description="Stage nanodiamond XYZ structures with fully explicit, recorded inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("particle")
    g.add_argument("--diameter", type=float, default=1.1,
                   help="nanodiamond diameter in nm; for --shape wulff this is the "
                        "equal-volume sphere diameter")
    g.add_argument("--shape", choices=["sphere", "wulff"], default="sphere",
                   help="sphere = isotropic cut; wulff = {111} octahedron truncated "
                        "by {100} planes")
    g.add_argument("--wulff-ratio", type=float, default=RATIO_OCTAHEDRON,
                   help=f"r100/r111 = gamma100/gamma111; {RATIO_CUBE:.4f} or below is a "
                        f"cube, {RATIO_OCTAHEDRON:.4f} or above a pure octahedron, "
                        f"{2 / np.sqrt(3):.4f} a cuboctahedron")
    g.add_argument("--wulff-ratio-110", type=float, default=RATIO_OCTAHEDRON,
                   help="r110/r111; the default is high enough that {110} never cuts. "
                        "Detonation-nanodiamond-like faceting is r100:r110:r111 = "
                        "0.99:1.03:1, i.e. --wulff-ratio 0.99 --wulff-ratio-110 1.03")
    g.add_argument("--keep-ch3", dest="prune_ch3", action="store_false", default=True,
                   help="keep one-coordinate surface carbons that passivate into CH3")
    g.add_argument("--rotate-111", dest="rotate_111", action="store_true", default=True,
                   help="orient the [111] NV axis along z")
    g.add_argument("--no-rotate-111", dest="rotate_111", action="store_false",
                   help="keep the crystal in the cubic frame")
    g.add_argument("--nv", dest="nv_vacancy", action="store_true", default=True,
                   help="create the NV centre (vacancy + substitutional N)")
    g.add_argument("--no-nv", dest="nv_vacancy", action="store_false",
                   help="leave a pristine core with no NV centre")

    g = p.add_argument_group("termination chemistry")
    g.add_argument("--termination", choices=["mixed", "h", "oh", "bare"], default="mixed",
                   help="mixed = OH + bridging O + H; h = all hydrogen; "
                        "oh = maximal hydroxyl; bare = no adatoms at all")
    g.add_argument("--oh-o-ratio", type=float, default=4.0,
                   help="OH groups per bridging ether O (0 disables bridging O)")
    g.add_argument("--oh-base-fraction", type=float, default=25.0 / 67.0,
                   help="OH coverage prefactor for the automatic target")
    g.add_argument("--oh-scale-ref", type=float, default=4.0,
                   help="reference radius in angstrom for the 1/r coverage scaling")
    g.add_argument("--n-oh", type=int, default=None,
                   help="explicit OH count, overrides the automatic target")
    g.add_argument("--n-o", type=int, default=None,
                   help="explicit bridging O count, overrides --oh-o-ratio")

    g = p.add_argument_group("geometry tolerances")
    g.add_argument("--bond-cc-cut", type=float, default=1.7, help="C-C neighbour cutoff")
    g.add_argument("--bond-tol", type=float, default=0.2, help="slack on the C-C cutoff")
    g.add_argument("--bond-ch", type=float, default=1.09, help="C-H bond length")
    g.add_argument("--bond-co", type=float, default=1.43, help="C-O bond length")
    g.add_argument("--bond-oh", type=float, default=0.96, help="O-H bond length")
    g.add_argument("--angle-hch", type=float, default=109.5, help="tetrahedral angle in degrees")
    g.add_argument("--angle-coh", type=float, default=104.5, help="C-O-H angle in degrees")
    g.add_argument("--clash-oh", type=float, default=1.00, help="min O...H separation")
    g.add_argument("--clash-oo", type=float, default=1.65, help="min O...O separation")
    g.add_argument("--clash-oos", type=float, default=1.60, help="min O...bridging-O separation")

    g = p.add_argument_group("solvent and box")
    g.add_argument("--n-water", type=int, default=None,
                   help="number of water molecules; omit to fill --box to "
                        "--water-density, 0 for a dry structure (then --box is required)")
    g.add_argument("--box", type=float, default=None,
                   help="cubic cell edge in nm, same units as --diameter; omit to derive "
                        "it from --n-water. At least one of --box / --n-water is required")
    g.add_argument("--water-density", type=float, default=1.0,
                   help="solvent density in g/cm^3 used to derive the box")
    g.add_argument("--water-gap", type=float, default=3.0,
                   help="solvent exclusion shell added to the particle radius, in "
                        "angstrom (packmol units)")
    g.add_argument("--water-radius", type=float, default=1.25,
                   help="packmol per-atom radius for water, in angstrom")
    g.add_argument("--packmol-tolerance", type=float, default=2.0,
                   help="packmol tolerance in angstrom")
    g.add_argument("--additives", nargs="+", default=None, metavar="XYZ",
                   help="co-solvent/solute XYZ templates packed before the water, "
                        "space- or comma-separated and optionally bracketed: "
                        "A.xyz B.xyz or [A.xyz,B.xyz]. Each file's basename becomes "
                        "the atom-label tag, e.g. DMSO.xyz -> S_DMSO, C_DMSO")
    g.add_argument("--n-additives", nargs="+", default=None, metavar="N",
                   help="molecule count per --additives entry, in the same order and "
                        "the same list syntax, e.g. 1 1 or [1,1]")
    g.add_argument("--additive-margin", type=float, default=1.0,
                   help="keep additives this far inside the cell wall, in angstrom, so "
                        "the water pass can hold them fixed")
    g.add_argument("--water-xyz", default="water.xyz",
                   help="water template; written if missing")
    g.add_argument("--packmol-exe", default="packmol", help="packmol executable")

    g = p.add_argument_group("realization and output")
    g.add_argument("--seed", type=int, default=None,
                   help="placement realization seed; drawn and recorded if omitted")
    g.add_argument("--prefix", default=None,
                   help="basename for all outputs; overrides the individual defaults")
    g.add_argument("--outdir", default=".", help="directory for all generated files")
    g.add_argument("--out", default=None,
                   help="final structure (default diamond_all.extxyz)")
    g.add_argument("--core-out", default=None,
                   help="core-only structure (default diamond_np.extxyz)")
    g.add_argument("--packmol-inp", default=None, help="packmol input (default packmol.inp)")
    g.add_argument("--manifest", default=None, help="JSON manifest (default <out>.manifest.json)")
    g.add_argument("--no-wrap", dest="wrap", action="store_false", default=True,
                   help="leave atoms where packmol put them instead of wrapping them "
                        "into the periodic cell")
    g.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    g.add_argument("-v", "--verbose", action="store_true", help="stream packmol output")
    return p


def resolve_paths(cfg):
    if cfg.prefix:
        defaults = (f"{cfg.prefix}.extxyz", f"{cfg.prefix}_core.extxyz",
                    f"{cfg.prefix}_packmol.inp")
    else:
        defaults = ("diamond_all.extxyz", "diamond_np.extxyz", "packmol.inp")

    cfg.out, cfg.core_out, cfg.packmol_inp = (
        getattr(cfg, attr) or default
        for attr, default in zip(("out", "core_out", "packmol_inp"), defaults)
    )

    if cfg.outdir and cfg.outdir != ".":
        os.makedirs(cfg.outdir, exist_ok=True)
        for attr in ("out", "core_out", "packmol_inp", "water_xyz"):
            v = getattr(cfg, attr)
            if not os.path.isabs(v):
                setattr(cfg, attr, os.path.join(cfg.outdir, v))

    stem = os.path.splitext(cfg.out)[0]
    cfg.manifest = cfg.manifest or (stem + ".manifest.json")
    cfg.packmol_out = stem + "_solvent.xyz"
    cfg.additive_out = stem + "_additives.xyz"
    cfg.additive_inp = os.path.splitext(cfg.packmol_inp)[0] + "_additives.inp"
    return cfg


PHYSICAL_KEYS = [
    "diameter", "lattice_constant", "rotate_111", "nv_vacancy",
    "shape", "wulff_ratio", "wulff_ratio_110", "prune_ch3",
    "termination", "oh_o_ratio", "oh_base_fraction", "oh_scale_ref", "n_oh", "n_o",
    "bond_cc_cut", "bond_tol", "bond_ch", "bond_co", "bond_oh",
    "angle_hch", "angle_coh", "clash_oh", "clash_oo", "clash_oos",
    "n_water", "box", "water_density", "water_gap", "water_radius",
    "packmol_tolerance", "additive_margin", "seed",
]


def physical_config(cfg, specs=()):
    inputs = {k: getattr(cfg, k) for k in PHYSICAL_KEYS}
    inputs["additives"] = [{"tag": s["tag"], "n": s["n"], "n_atoms": s["n_atoms"],
                            "formula": s["formula"]} for s in specs]
    return inputs


def config_hash(inputs):
    blob = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def provenance_string(inputs, chash, manifest_path, box_source, n_water_source):
    return " ".join([
        f"config_hash={chash}",
        f"seed={inputs['seed']}",
        f"shape={inputs['shape']}",
        f"wulff_ratio={inputs['wulff_ratio']:g}",
        f"termination={inputs['termination']}",
        f"oh_o_ratio={inputs['oh_o_ratio']:g}",
        f"n_water={inputs['n_water']}",
        f"n_water_source={n_water_source}",
        f"box_nm={inputs['box']:.6f}",
        f"box_source={box_source}",
        f"water_density={inputs['water_density']:g}",
        f"nv={int(inputs['nv_vacancy'])}",
        f"rotate_111={int(inputs['rotate_111'])}",
        f"manifest={os.path.basename(manifest_path)}",
    ])


def resolve_cell(cfg, r, r_excl, v_add=0.0):
    legacy_box_nm = 2.25 * cfg.diameter
    box_source = n_water_source = "explicit"

    if cfg.n_water is None and cfg.box is None:
        die("neither --n-water nor --box was given, so the cell is undetermined.\n"
            "Give --n-water N to derive the box from the solvent, --box L to fill "
            "that cell to --water-density, both to pin them independently, or "
            f"--n-water 0 --box L for a dry structure.\n"
            f"(--box {legacy_box_nm:.4f} is the legacy 2.25*D cell for a "
            f"{cfg.diameter:g} nm particle.)")

    if cfg.n_water == 0:
        if cfg.box is None:
            die("--n-water 0 was requested, so the box cannot be derived from the "
                "solvent and must be stated explicitly.\n"
                f"Re-run with --box <edge_in_nm> (e.g. --box {legacy_box_nm:.4f} "
                f"reproduces the legacy 2.25*D box for a {cfg.diameter:g} nm particle).")
        if cfg.box <= cfg.diameter:
            die(f"--box {cfg.box:g} nm is not larger than the particle diameter "
                f"({cfg.diameter:g} nm).")
        box_length = float(cfg.box) * NM_TO_ANGSTROM
        v_excl = sphere_volume(r_excl)
        return (box_length, box_length ** 3 - v_excl - v_add, v_excl,
                box_source, n_water_source)

    if cfg.box is None:
        n_min = min_water_count(r, cfg.water_gap, cfg.water_density, v_add)
        if cfg.n_water < n_min:
            die(f"--n-water {cfg.n_water} derives a box smaller than the "
                f"{2.0 * r_excl / NM_TO_ANGSTROM:.4f} nm exclusion sphere"
                + (f" plus {v_add:.1f} cubic angstrom of additives" if v_add else "")
                + f".\nUse at least --n-water {n_min}, or give --box explicitly.")
        box_length, v_free, v_excl = box_from_water_count(
            cfg.n_water, r, cfg.water_gap, cfg.water_density, v_add)
        return box_length, v_free, v_excl, "derived_from_water_count", n_water_source

    box_length = float(cfg.box) * NM_TO_ANGSTROM
    if box_length <= 2.0 * r_excl:
        die(f"--box {cfg.box:g} nm does not enclose the "
            f"{2.0 * r_excl / NM_TO_ANGSTROM:.4f} nm exclusion sphere "
            f"(particle {cfg.diameter:g} nm + 2 x --water-gap {cfg.water_gap:g} A).\n"
            "Enlarge --box, or shrink --water-gap.")
    if cfg.n_water is None:
        cfg.n_water, v_free, v_excl = water_count_from_box(
            box_length, r, cfg.water_gap, cfg.water_density, v_add)
        if cfg.n_water < 1:
            die(f"--box {cfg.box:g} nm leaves too little free volume to hold a "
                f"single water at --water-density {cfg.water_density:g} g/cm^3"
                + (f" once {v_add:.1f} cubic angstrom of additives are placed" if v_add
                   else "") + ".")
        return box_length, v_free, v_excl, box_source, "derived_from_box"

    v_excl = sphere_volume(r_excl)
    return (box_length, box_length ** 3 - v_excl - v_add, v_excl,
            box_source, n_water_source)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = build_parser().parse_args(argv)
    cfg.lattice_constant = LATTICE_CONSTANT

    if cfg.n_water is not None and cfg.n_water < 0:
        die("--n-water must be >= 0.")
    if cfg.shape == "wulff" and min(cfg.wulff_ratio, cfg.wulff_ratio_110) <= 0:
        die("--wulff-ratio and --wulff-ratio-110 must be > 0.")

    specs = resolve_additives(cfg)
    v_add = additive_volume(specs)

    r = cfg.diameter * NM_TO_ANGSTROM / 2.0
    r_excl = r + cfg.water_gap
    box_length, v_free, v_excl, box_source, n_water_source = resolve_cell(
        cfg, r, r_excl, v_add)

    box_nm = box_length / NM_TO_ANGSTROM
    rho_solvent = (effective_density(cfg.n_water, box_length, r, cfg.water_gap, v_add)
                   if cfg.n_water > 0 else 0.0)
    if cfg.n_water > 0 and not 0.1 <= rho_solvent <= 1.5:
        sys.stderr.write(
            f"WARNING: {cfg.n_water} waters in a {box_nm:.4f} nm box give an "
            f"effective solvent density of {rho_solvent:.3f} g/cm^3; packmol may fail "
            f"or the structure may be unphysical.\n")

    cfg.box = box_nm
    cfg = resolve_paths(cfg)

    if cfg.seed is None:
        cfg.seed = int.from_bytes(os.urandom(4), "big")
        if not cfg.quiet:
            print(f"No --seed given; drew seed {cfg.seed} for this realization.")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed % (2 ** 32))

    inputs = physical_config(cfg, specs)
    chash = config_hash(inputs)
    prov = provenance_string(inputs, chash, cfg.manifest, box_source, n_water_source)

    if not cfg.quiet:
        box_note = "given explicitly" if box_source == "explicit" else "derived from water count"
        n_note = "given explicitly" if n_water_source == "explicit" else "derived from box"
        print(f"Box edge {box_nm:.4f} nm ({box_note}); "
              f"{cfg.n_water} waters ({n_note})"
              + (f" at {rho_solvent:.4f} g/cm^3 solvent density" if cfg.n_water > 0 else "")
              + f"; config {chash}")
        if specs:
            print("Additives: " + ", ".join(
                f"{s['n']} x {s['tag']} ({s['formula']}, {s['n_atoms']} atoms)"
                for s in specs)
                + f"; {v_add:.1f} cubic angstrom of van der Waals volume reserved")

    gen = DiamondCoreGenerator(
        shape=cfg.shape, wulff_ratio=cfg.wulff_ratio,
        wulff_ratio_110=cfg.wulff_ratio_110,
        prune_cutoff=(cfg.bond_cc_cut + cfg.bond_tol) if cfg.prune_ch3 else None,
    )
    facets = gen.facet_radii(r)
    if facets and not cfg.quiet:
        r111, r100, r110 = facets
        print(f"Wulff core: r111 {r111:.4f} A, r100 {r100:.4f} A, r110 {r110:.4f} A "
              f"(r100:r110:r111 {cfg.wulff_ratio:g}:{cfg.wulff_ratio_110:g}:1)")

    gen, counts = functionalize_surface(gen, r, cfg)
    if cfg.prune_ch3 and gen.n_pruned and not cfg.quiet:
        print(f"Pruned {gen.n_pruned} under-coordinated carbons before passivation.")
    print_termination(cfg, counts)

    def print_size(gen):
        mp, mc = gen.metrics.get("particle"), gen.metrics.get("core")
        if not mp or cfg.quiet:
            return
        print(f"Size (core only): Rg {mc['rg_angstrom']:.3f} A, "
              f"equiv sphere D {mc['d_equiv_rg_nm']:.4f} nm, "
              f"r_max {mc['r_max_angstrom']:.3f} A, "
              f"r_inscribed {mc['r_inscribed_angstrom']:.3f} A")
        print(f"Size (terminated): Rg {mp['rg_angstrom']:.3f} A, "
              f"equiv sphere D {mp['d_equiv_rg_nm']:.4f} nm, "
              f"r_max {mp['r_max_angstrom']:.3f} A, "
              f"centroid {mp['com_offset_angstrom']:.3f} A off the vacancy")

    xyz_kwargs = dict(rotate_111_to_z=cfg.rotate_111, nv_vacancy=cfg.nv_vacancy,
                      provenance=prov)
    core_str = gen.to_xyz(r, box_length, **xyz_kwargs)
    n_core_atoms = len(core_str.splitlines()) - 2
    print_size(gen)

    write_text(cfg.core_out, core_str)
    n_packed, additive_atoms = 0, []
    if cfg.n_water > 0 or specs:
        gen, n_packed, additive_atoms = add_solvent_shell(
            gen, r, cfg, box_length, specs)

    final_str = gen.to_xyz(r, box_length, wrap=cfg.wrap, **xyz_kwargs)
    n_total_atoms = len(final_str.splitlines()) - 2
    write_text(cfg.out, final_str)
    if cfg.wrap and gen.n_wrapped and not cfg.quiet:
        print(f"Wrapped {gen.n_wrapped} atoms into the periodic cell "
              f"[{-box_nm / 2:.4f}, {box_nm / 2:.4f}] nm on each axis.")

    elements, mass_amu = system_composition(final_str)
    rho_system = system_density(mass_amu, box_length)

    manifest = {
        "config_hash": chash,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": argv,
        "command": "python " + os.path.basename(__file__) + " " + " ".join(argv),
        "inputs": inputs,
        "derived": {
            "radius_nm": cfg.diameter / 2.0,
            "shape": cfg.shape,
            "r111_angstrom": facets[0] if facets else None,
            "r100_angstrom": facets[1] if facets else None,
            "r110_angstrom": facets[2] if facets else None,
            "size_metrics": gen.metrics,
            "n_carbons_pruned": int(gen.n_pruned),
            "n_atoms_wrapped": int(gen.n_wrapped),
            "box_length_nm": box_nm,
            "box_length_angstrom": box_length,
            "box_source": box_source,
            "n_water_source": n_water_source,
            "free_volume_angstrom3": v_free,
            "excluded_volume_angstrom3": v_excl,
            "additive_volume_angstrom3": v_add,
            "exclusion_radius_angstrom": r_excl,
            "solvent_density_g_cm3": rho_solvent,
            "system_density_g_cm3": rho_system,
            "system_mass_amu": mass_amu,
            "elements": elements,
        },
        "counts": dict(counts, n_core_atoms=n_core_atoms,
                       n_water_packed=n_packed,
                       n_additive_molecules=sum(s["n"] for s in specs),
                       n_additive_atoms=len(additive_atoms),
                       n_total_atoms=n_total_atoms),
        "additives": [dict(s, labels=sorted(set(s["labels"]))) for s in specs],
        "outputs": {
            "structure": cfg.out,
            "core": cfg.core_out,
            "packmol_input": cfg.packmol_inp if cfg.n_water > 0 else None,
            "solvent": cfg.packmol_out if cfg.n_water > 0 else None,
            "additive_packmol_input": cfg.additive_inp if specs else None,
            "additive_structure": cfg.additive_out if specs else None,
        },
    }
    write_text(cfg.manifest, json.dumps(manifest, indent=2, sort_keys=False) + "\n")

    formula = hill_formula(elements)
    print(f"Total system: {n_total_atoms} atoms ({formula}) in a {box_nm:.4f} nm box")
    print(f"System density: {rho_system:.4f} g/cm^3")

    if not cfg.quiet:
        print(f"{n_core_atoms} atoms at D={cfg.diameter:.2f} nm (core)")
        print(f"{n_total_atoms} atoms -> {cfg.out}")
        print(f"provenance -> {cfg.manifest} (config {chash}, seed {cfg.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
