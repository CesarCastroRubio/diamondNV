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

# example: python diamond.py --diameter 1.1 --box 2.5 --seed 1 --shape wulff --wulff-ratio 0.99 --wulff-ratio-110 1.03 --additives [TEMPOL.xyz] --n-additives [1]

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
WATER_CHARGE = {"OW": -0.834, "HW": 0.417}

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


def parse_float_list(tokens, flag):
    values = []
    for t in flatten_list_arg(tokens) or []:
        try:
            values.append(float(t))
        except ValueError:
            die(f"{flag} got '{t}', which is not a number.")
    return values


def apportion(weights, total):
    w = np.array(weights, float)
    if (w < 0).any():
        die("--additive-ratios entries must be >= 0.")
    if w.sum() <= 0:
        die("--additive-ratios must contain at least one positive entry.")
    exact = w / w.sum() * total
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - counts), kind="stable")[:total - counts.sum()]:
        counts[i] += 1
    return counts.tolist()


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
    ratios = parse_float_list(cfg.additive_ratios, "--additive-ratios")
    if not paths:
        if counts or ratios:
            die("--n-additives/--additive-ratios were given without --additives.")
        return []

    if ratios:
        if len(ratios) != len(paths):
            die(f"--additives lists {len(paths)} files but --additive-ratios lists "
                f"{len(ratios)} weights; they must match one-to-one.")
        if len(counts) != 1:
            die("--additive-ratios sets the mix, so --n-additives must be a single "
                "total, e.g. --additive-ratios [1,1,1,1] --n-additives 40.")
        if counts[0] < 0:
            die("--n-additives total must be >= 0.")
        counts = apportion(ratios, counts[0])
    else:
        if not counts:
            die("--additives was given without --n-additives; state a count per file, "
                "or give --additive-ratios plus a single --n-additives total.")
        if len(paths) != len(counts):
            die(f"--additives lists {len(paths)} files but --n-additives lists "
                f"{len(counts)} counts; they must match one-to-one.")
        ratios = [None] * len(paths)

    specs = []
    for path, n, ratio in zip(paths, counts, ratios):
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
            "ratio": ratio,
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


def random_rotation(rng=None):
    w, x, y, z = unit(np.random.randn(4) if rng is None else rng.standard_normal(4))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def excluded_volume_mc(np_xyz, gap, box_length, samples=400000):
    pts = np.random.uniform(-box_length / 2.0, box_length / 2.0, (samples, 3))
    hit = cKDTree(np_xyz).query(pts, distance_upper_bound=gap)[0] < gap
    return box_length ** 3 * hit.mean()


def np_clearance_ok(cfg, m, r_excl, np_tree):
    if cfg.exclusion == "atoms":
        return np_tree.query(m)[0].min() >= cfg.surface_gap
    return np.linalg.norm(m, axis=1).min() >= r_excl


def pack_additives(cfg, r_excl, box_length, specs, np_atoms):
    half = box_length / 2.0 - cfg.additive_margin
    if half <= r_excl:
        die(f"--box leaves no room for additives: the {2 * r_excl:.2f} angstrom "
            f"exclusion sphere fills the cell once the {cfg.additive_margin:g} "
            f"angstrom additive margin is applied.")

    np_tree = cKDTree(np.array([a[1:] for a in np_atoms], float))
    order_rng = np.random.default_rng([cfg.additive_seed, 0xA11])
    place_rng = np.random.default_rng([cfg.additive_seed, 0xB22])

    order = np.repeat(np.arange(len(specs)), [s["n"] for s in specs])
    order_rng.shuffle(order)
    bases = [np.array([a[1:] for a in read_xyz(s["path"])], float) for s in specs]
    bases = [b - b.mean(axis=0) for b in bases]

    placed = np.zeros((0, 3))
    labeled = []
    for slot, si in enumerate(order):
        s, base = specs[si], bases[si]
        for _ in range(cfg.additive_attempts):
            m = base @ random_rotation(place_rng).T + place_rng.uniform(-half, half, 3)
            if np.abs(m).max() > half:
                continue
            if not np_clearance_ok(cfg, m, r_excl, np_tree):
                continue
            if len(placed) and cKDTree(placed).query(m)[0].min() < cfg.packmol_tolerance:
                continue
            placed = np.vstack([placed, m])
            labeled += [(lab, *p) for lab, p in zip(s["labels"], m)]
            break
        else:
            die(f"could not place {s['tag']} (slot {slot + 1} of {len(order)}) in "
                f"{cfg.additive_attempts} random tries without a clash.\n"
                f"Enlarge --box, lower --n-additives, or raise "
                f"--additive-attempts above {cfg.additive_attempts}.")

    write_text(cfg.additive_out, f"{len(labeled)}\nadditives\n" + "".join(
        f"{label_element(lab)} {x:.6f} {y:.6f} {z:.6f}\n" for lab, x, y, z in labeled))

    if not cfg.quiet:
        summary = ", ".join(f"{s['n']} x {s['tag']}" for s in specs)
        r = np.linalg.norm(placed, axis=1)
        print(f"Placed additives ({summary}) at random: {len(labeled)} atoms, "
              f"{r.min():.2f}-{r.max():.2f} angstrom from the vacancy")
    return labeled


def write_fixed_xyz(path, labeled):
    write_text(path, f"{len(labeled)}\nfixed\n" + "".join(
        f"{label_element(s)} {x:.6f} {y:.6f} {z:.6f}\n" for s, x, y, z in labeled))


def pack_water(cfg, r_inner, box_length, additive_atoms, np_atoms):
    water_xyz = ensure_water_template(cfg.water_xyz, cfg.quiet)
    body = (f"structure {water_xyz}\n"
            f"  number {cfg.n_water}\n"
            f"  outside sphere 0.0 0.0 0.0 {r_inner:.6f}\n"
            f"  radius {cfg.water_radius}\n"
            f"end structure\n\n")
    if additive_atoms:
        body += (f"structure {cfg.additive_out}\n"
                 f"  number 1\n"
                 f"  fixed 0. 0. 0. 0. 0. 0.\n"
                 f"end structure\n\n")
    n_fixed = len(additive_atoms)
    if cfg.exclusion == "atoms":
        write_fixed_xyz(cfg.np_out, np_atoms)
        body += (f"structure {cfg.np_out}\n"
                 f"  number 1\n"
                 f"  fixed 0. 0. 0. 0. 0. 0.\n"
                 f"  radius {cfg.surface_gap - cfg.water_radius:.6f}\n"
                 f"end structure\n\n")
        n_fixed += len(np_atoms)

    atoms = run_packmol(cfg, cfg.packmol_inp,
                        packmol_header(cfg, box_length, cfg.packmol_out) + body,
                        cfg.packmol_out, "water")

    n_expected = 3 * cfg.n_water
    if len(atoms) != n_expected + n_fixed:
        die(f"packmol wrote {len(atoms)} atoms but {n_expected} water atoms plus "
            f"{n_fixed} fixed atoms were expected.")

    relabel = {"O": "OW", "H": "HW"}
    water_atoms = [(relabel.get(s, s), x, y, z) for s, x, y, z in atoms[:n_expected]]
    if any(s not in ("OW", "HW") for s, *_ in water_atoms):
        die("the packmol water block did not come back as pure water; refusing to "
            "label a structure whose manifest would be wrong.")
    return water_atoms


def water_template_frame(cfg):
    atoms = read_xyz(ensure_water_template(cfg.water_xyz, cfg.quiet))
    xyz = np.array([a[1:] for a in atoms], float)
    order = np.argsort([0 if normalize_element(a[0]) == "O" else 1 for a in atoms])
    xyz = xyz[order]
    return xyz[1:] - xyz[0]


def kabsch(src, dst):
    u, _, vt = np.linalg.svd(dst.T @ src)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def water_hbonds(water_atoms):
    W = np.array([a[1:] for a in water_atoms], float).reshape(-1, 3, 3)
    return hbond_count(W[:, 0], W[:, 1:])


def hbond_count(O, H):
    tree = cKDTree(O)
    n = 0
    for i, hs in enumerate(H):
        for h in hs:
            for j in tree.query_ball_point(h, 2.5):
                if j == i:
                    continue
                v1, v2 = O[i] - h, O[j] - h
                if np.degrees(np.arccos(np.clip(
                        v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))) > 130:
                    n += 2
    return n / max(len(O), 1)


def relax_water_orientations(water_atoms, blocker_xyz, cfg):
    if cfg.water_relax_sweeps <= 0 or not water_atoms:
        return water_atoms
    W = np.array([a[1:] for a in water_atoms], float).reshape(-1, 3, 3)
    O, H = W[:, 0], W[:, 1:]
    h_local = water_template_frame(cfg)
    q = np.array([WATER_CHARGE["HW"], WATER_CHARGE["HW"]])
    q_env = np.array([WATER_CHARGE["OW"], WATER_CHARGE["HW"], WATER_CHARGE["HW"]])
    tree = cKDTree(O)
    blockers = cKDTree(blocker_xyz) if len(blocker_xyz) else None

    for _ in range(cfg.water_relax_sweeps):
        for i in np.random.permutation(len(O)):
            neigh = [j for j in tree.query_ball_point(O[i], cfg.water_relax_cutoff)
                     if j != i]
            if not neigh:
                continue
            env = np.concatenate([np.concatenate([O[neigh][:, None, :], H[neigh]], axis=1)])
            env_xyz = env.reshape(-1, 3)
            env_q = np.tile(q_env, len(neigh))
            trials = np.stack([np.eye(3)] + [random_rotation()
                                             for _ in range(cfg.water_relax_trials)])
            cand = O[i] + np.einsum("tab,hb->tha", trials, h_local)
            d = np.linalg.norm(cand[:, :, None, :] - env_xyz[None, None, :, :], axis=3)
            e = (q[None, :, None] * env_q[None, None, :] / np.maximum(d, 0.5)).sum((1, 2))
            if blockers is not None:
                floor = (cfg.surface_gap if cfg.exclusion == "atoms"
                         else cfg.water_relax_clash)
                e[blockers.query(cand.reshape(-1, 3))[0].reshape(len(trials), 2).min(1)
                  < floor] = np.inf
            best = int(np.argmin(e))
            if np.isfinite(e[best]):
                H[i] = cand[best]

    out = []
    for o, hs in zip(O, H):
        out.append(("OW", *o))
        out += [("HW", *h) for h in hs]
    return out


def ice_lattice_constant(density):
    return (8.0 * MOLAR_MASS_H2O /
            (AVOGADRO * density * ANGSTROM3_TO_CM3)) ** (1.0 / 3.0)


def ice_cells(box_length, density):
    return max(1, int(round(box_length / ice_lattice_constant(density))))


def ice_water(cfg, r_excl, box_length, additive_atoms, np_atoms):
    n_cells = ice_cells(box_length, cfg.water_density)
    a = box_length / n_cells
    motif = np.array([np.add(f, b) for f in FCC_FRAC for b in DIAMOND_BASIS])
    rng = np.arange(n_cells)
    cells = np.array(list(itertools.product(rng, rng, rng)), float)
    O = ((cells[:, None, :] + motif[None, :, :]).reshape(-1, 3) * a) - box_length / 2.0

    if cfg.exclusion == "atoms":
        keep = cKDTree(np.array([a[1:] for a in np_atoms], float)).query(
            O)[0] >= cfg.surface_gap
    else:
        keep = np.linalg.norm(O, axis=1) >= r_excl
    if additive_atoms:
        keep &= cKDTree(np.array([a[1:] for a in additive_atoms], float)).query(
            O)[0] >= cfg.packmol_tolerance
    O = O[keep]

    if not len(O):
        die(f"--water-model ice leaves no lattice site outside the "
            f"{r_excl:.2f} angstrom exclusion sphere; enlarge --box.")

    d_oo = a * np.sqrt(3.0) / 4.0
    L = float(box_length)
    pairs = cKDTree((O + L / 2.0) % L, boxsize=L).query_pairs(
        d_oo * 1.15, output_type="ndarray")

    def bond_dir(src, dst):
        delta = O[dst] - O[src]
        return unit(delta - L * np.round(delta / L))

    degree = np.bincount(pairs.ravel(), minlength=len(O))
    target = np.minimum(degree, 2)
    donor = np.random.rand(len(pairs)) < 0.5

    def donations(donor):
        d = np.zeros(len(O), int)
        np.add.at(d, pairs[donor, 0], 1)
        np.add.at(d, pairs[~donor, 1], 1)
        return d

    d = donations(donor)
    for _ in range(cfg.ice_repair_sweeps):
        over = np.flatnonzero(d > target)
        if not len(over):
            break
        fwd = [[] for _ in range(len(O))]
        for e, (i, j) in enumerate(pairs):
            src, dst = (i, j) if donor[e] else (j, i)
            fwd[src].append((e, dst))
        augmented = False
        for s in over:
            prev, queue, found = {int(s): None}, [int(s)], None
            while queue and found is None:
                u = queue.pop(0)
                for e, v in fwd[u]:
                    if v in prev:
                        continue
                    prev[v] = (u, e)
                    if d[v] < target[v]:
                        found = v
                        break
                    queue.append(v)
            if found is None:
                continue
            d[found] += 1
            d[s] -= 1
            v = found
            while prev[v] is not None:
                u, e = prev[v]
                donor[e] = not donor[e]
                v = u
            augmented = True
            break
        if not augmented:
            break
    satisfied = float((d == target).mean())

    out_dirs = [[] for _ in range(len(O))]
    for e, (i, j) in enumerate(pairs):
        src, dst = (i, j) if donor[e] else (j, i)
        if len(out_dirs[src]) < 2:
            out_dirs[src].append(bond_dir(src, dst))

    h_local = water_template_frame(cfg)
    h_dirs = np.array([unit(v) for v in h_local])
    blocked = np.array([a[1:] for a in np_atoms] + [a[1:] for a in additive_atoms], float)
    blockers = cKDTree(blocked) if len(blocked) else None
    gap = cfg.surface_gap if cfg.exclusion == "atoms" else 0.0
    water_atoms, reoriented = [], 0

    for i, o in enumerate(O):
        dirs = out_dirs[i]
        while len(dirs) < 2:
            v = np.random.randn(3)
            for u in dirs:
                v -= (v @ u) * u
            dirs = dirs + [unit(v)]
        target = np.array(dirs[:2])
        R = kabsch(h_dirs, target)
        H = o + h_local @ R.T
        if blockers is not None and gap and blockers.query(H)[0].min() < gap:
            cand = np.stack([random_rotation() for _ in range(128)])
            pos = o + np.einsum("tab,hb->tha", cand, h_local)
            ok = blockers.query(pos.reshape(-1, 3))[0].reshape(len(cand), 2).min(1) >= gap
            if ok.any():
                score = np.einsum("tab,hb,ha->t", cand[ok], h_local / np.linalg.norm(
                    h_local, axis=1)[:, None], target)
                R = cand[ok][int(np.argmax(score))]
                H = o + h_local @ R.T
                reoriented += 1
        water_atoms.append(("OW", *o))
        water_atoms += [("HW", *h) for h in H]

    if not cfg.quiet:
        rho = (len(O) * MOLAR_MASS_H2O
               / (AVOGADRO * (box_length ** 3 - sphere_volume(r_excl))
                  * ANGSTROM3_TO_CM3))
        print(f"Ice Ic lattice: a {a:.4f} A ({n_cells} cells per edge), "
              f"O-O {d_oo:.3f} A, {len(O)} waters at {rho:.4f} g/cm^3, "
              f"ice rules satisfied on {100 * satisfied:.1f}% of sites, "
              f"H-bonds per water {water_hbonds(water_atoms):.2f}")
    return water_atoms


def add_solvent_shell(gen, r_angstrom, cfg, box_length, specs, np_atoms):
    r_excl = r_angstrom + cfg.water_gap
    r_inner = (gen.metrics.get("core", {}) or {}).get("r_inscribed_angstrom", r_angstrom)
    r_inner = r_inner if cfg.exclusion == "atoms" else r_excl
    additive_atoms = (pack_additives(cfg, r_excl, box_length, specs, np_atoms)
                      if specs else [])

    water_atoms = []
    if cfg.n_water:
        if cfg.water_model == "ice":
            water_atoms = ice_water(cfg, r_excl, box_length, additive_atoms, np_atoms)
        else:
            water_atoms = pack_water(cfg, r_inner, box_length, additive_atoms, np_atoms)
            if not cfg.quiet:
                free_v = box_length ** 3 - additive_volume(specs) - (
                    excluded_volume_mc(np.array([a[1:] for a in np_atoms], float),
                                       cfg.surface_gap, box_length)
                    if cfg.exclusion == "atoms" else sphere_volume(r_excl))
                print(f"Packed {cfg.n_water} water molecules ({len(water_atoms)} atoms) "
                      f"into {free_v:.2f} cubic angstrom of free volume")
            blockers = np.array([a[1:] for a in np_atoms] +
                                [a[1:] for a in additive_atoms], float) \
                if (np_atoms or additive_atoms) else np.zeros((0, 3))
            before = water_hbonds(water_atoms)
            water_atoms = relax_water_orientations(water_atoms, blockers, cfg)
            if not cfg.quiet and cfg.water_relax_sweeps > 0:
                print(f"Relaxed water orientations over {cfg.water_relax_sweeps} sweeps: "
                      f"H-bonds per water {before:.2f} -> "
                      f"{water_hbonds(water_atoms):.2f}")

    gen._extra_atoms = list(gen._extra_atoms) + additive_atoms + water_atoms
    return gen, len(water_atoms) // 3, additive_atoms


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
    g.add_argument("--exclusion", choices=["atoms", "sphere"], default="atoms",
                   help="atoms = keep solvent --surface-gap away from every "
                        "nanoparticle atom, which follows facets and terminations; "
                        "sphere = legacy single exclusion sphere of radius "
                        "particle-radius + --water-gap")
    g.add_argument("--surface-gap", type=float, default=2.0,
                   help="min distance from any nanoparticle atom to any solvent or "
                        "additive atom, in angstrom, when --exclusion atoms")
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
                        "the same list syntax, e.g. 1 1 or [1,1]. With "
                        "--additive-ratios this is instead a single grand total")
    g.add_argument("--additive-ratios", nargs="+", default=None, metavar="W",
                   help="relative weights per --additives entry, any positive floats, "
                        "renormalised to the --n-additives total by largest remainder, "
                        "e.g. --additive-ratios [1,1,1,1] --n-additives 40")
    g.add_argument("--additive-seed", type=int, default=None,
                   help="seed for additive identity and placement only; defaults to "
                        "--seed, and is independent of the particle realization")
    g.add_argument("--additive-margin", type=float, default=1.0,
                   help="keep additives this far inside the cell wall, in angstrom, so "
                        "the water pass can hold them fixed")
    g.add_argument("--additive-attempts", type=int, default=20000,
                   help="random placement tries per additive molecule before giving up")
    g.add_argument("--water-model", choices=["liquid", "ice"], default="liquid",
                   help="liquid = packmol positions with the orientations relaxed into "
                        "a hydrogen-bond network; ice = cubic ice Ic lattice obeying "
                        "the Bernal-Fowler rules, which must be melted")
    g.add_argument("--water-relax-sweeps", type=int, default=4,
                   help="orientation relaxation passes over the water; 0 disables")
    g.add_argument("--water-relax-trials", type=int, default=96,
                   help="candidate orientations tried per water per sweep")
    g.add_argument("--water-relax-cutoff", type=float, default=6.0,
                   help="neighbour cutoff for the orientation energy, in angstrom")
    g.add_argument("--water-relax-clash", type=float, default=1.4,
                   help="min distance from a relaxed water H to any non-water atom")
    g.add_argument("--ice-repair-sweeps", type=int, default=20000,
                   help="augmenting passes used to enforce the ice rules on the protons")
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
    cfg.np_out = stem + "_np_fixed.xyz"
    return cfg


PHYSICAL_KEYS = [
    "diameter", "lattice_constant", "rotate_111", "nv_vacancy",
    "shape", "wulff_ratio", "wulff_ratio_110", "prune_ch3",
    "termination", "oh_o_ratio", "oh_base_fraction", "oh_scale_ref", "n_oh", "n_o",
    "bond_cc_cut", "bond_tol", "bond_ch", "bond_co", "bond_oh",
    "angle_hch", "angle_coh", "clash_oh", "clash_oo", "clash_oos",
    "n_water", "box", "water_density", "water_gap", "water_radius",
    "packmol_tolerance", "additive_margin", "additive_attempts",
    "exclusion", "surface_gap", "additive_seed",
    "water_model", "water_relax_sweeps", "water_relax_trials", "water_relax_cutoff",
    "water_relax_clash", "ice_repair_sweeps", "seed",
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
    if cfg.exclusion == "atoms" and cfg.surface_gap <= cfg.water_radius:
        die(f"--surface-gap {cfg.surface_gap:g} must exceed --water-radius "
            f"{cfg.water_radius:g}; packmol needs a positive radius for the fixed "
            f"nanoparticle.")

    specs = resolve_additives(cfg)
    v_add = additive_volume(specs)

    r = cfg.diameter * NM_TO_ANGSTROM / 2.0
    r_excl = r + cfg.water_gap
    box_length, v_free, v_excl, box_source, n_water_source = resolve_cell(
        cfg, r, r_excl, v_add)

    if cfg.water_model == "ice" and cfg.n_water:
        n_cells = ice_cells(box_length, cfg.water_density)
        snapped = n_cells * ice_lattice_constant(cfg.water_density)
        if abs(snapped - box_length) > 1e-9:
            sys.stderr.write(
                f"NOTE: --water-model ice snapped the box from "
                f"{box_length / NM_TO_ANGSTROM:.4f} to {snapped / NM_TO_ANGSTROM:.4f} nm "
                f"so that {n_cells} ice cells tile it exactly at --water-density "
                f"{cfg.water_density:g} g/cm^3.\n")
            box_length = snapped
            v_excl = sphere_volume(r_excl)
            v_free = box_length ** 3 - v_excl - v_add
            box_source += "+ice_snapped"

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
    if cfg.additive_seed is None:
        cfg.additive_seed = cfg.seed

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

    if cfg.exclusion == "atoms" and cfg.n_water:
        core_xyz = np.array(gen.positions(r, rotate_111_to_z=cfg.rotate_111)
                            + [a[1:] for a in gen._extra_atoms], float)
        v_excl = excluded_volume_mc(core_xyz, cfg.surface_gap, box_length)
        v_free = box_length ** 3 - v_excl - v_add
        if n_water_source == "derived_from_box":
            cfg.n_water = max(1, int(round(volume_to_n_water(v_free, cfg.water_density))))
            n_water_source = "derived_from_box_atomwise"
        rho_solvent = (cfg.n_water * MOLAR_MASS_H2O
                       / (AVOGADRO * v_free * ANGSTROM3_TO_CM3))

    inputs = physical_config(cfg, specs)
    chash = config_hash(inputs)
    prov = provenance_string(inputs, chash, cfg.manifest, box_source, n_water_source)

    if not cfg.quiet:
        box_note = "given explicitly" if box_source.startswith("explicit") else "derived from water count"
        n_note = "given explicitly" if n_water_source == "explicit" else "derived from box"
        print(f"Box edge {box_nm:.4f} nm ({box_note}); "
              f"{cfg.n_water} waters ({n_note})"
              + (f" at {rho_solvent:.4f} g/cm^3 solvent density" if cfg.n_water > 0 else "")
              + f"; config {chash}")
        if specs:
            total = sum(s["n"] for s in specs)
            print("Additives: " + ", ".join(
                f"{s['n']} x {s['tag']}"
                + (f" ({s['n'] / total:.3f} of {total}, asked {s['ratio']:g})"
                   if s["ratio"] is not None else f" ({s['formula']})")
                for s in specs)
                + f"; {v_add:.1f} cubic angstrom of van der Waals volume reserved")

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
        np_atoms = [(l.split()[0], *[float(v) for v in l.split()[1:4]])
                    for l in core_str.splitlines()[2:] if l.split()]
        gen, n_packed, additive_atoms = add_solvent_shell(
            gen, r, cfg, box_length, specs, np_atoms)

    final_str = gen.to_xyz(r, box_length, wrap=cfg.wrap, **xyz_kwargs)
    n_total_atoms = len(final_str.splitlines()) - 2
    write_text(cfg.out, final_str)
    if cfg.wrap and gen.n_wrapped and not cfg.quiet:
        print(f"Wrapped {gen.n_wrapped} atoms into the periodic cell "
              f"[{-box_nm / 2:.4f}, {box_nm / 2:.4f}] nm on each axis.")

    if n_packed != cfg.n_water:
        if not cfg.quiet:
            print(f"--water-model {cfg.water_model} set the water count from the "
                  f"lattice: {n_packed} waters, not the {cfg.n_water} the box and "
                  f"--water-density implied.")
        cfg.n_water = n_packed
        rho_solvent = effective_density(n_packed, box_length, r, cfg.water_gap, v_add)

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
