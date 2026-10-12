#!/usr/bin/env python3
import argparse
import collections
import hashlib
import itertools
import json
import os
import re
import sys
from datetime import datetime, timezone

# example: python diamond.py --diameter 1.1 --box 2.5 --seed 1 --shape wulff --gamma-100 0.99 --gamma-110 1.03 --substitution ["*O*"] --substitution-ratio [0.1] --functionalization ["*O","*[H]"] --functionalization-ratio [0.3,rest] --additives [TEMPOL.xyz] --n-additives [1]

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree
print(rdkit.__version__)

RDLogger.DisableLog("rdApp.*")

MOLAR_MASS_H2O = 18.01528
AVOGADRO = 6.02214076e23
ANGSTROM3_TO_CM3 = 1e-24
NM_TO_ANGSTROM = 10.0
LATTICE_CONSTANT = 3.567
BOND_LENGTH = LATTICE_CONSTANT * np.sqrt(3.0) / 4.0
C_VALENCE = 4
CH_BOND = 1.09

ATOMIC_MASS = {"H": 1.008, "B": 10.811, "C": 12.011, "N": 14.007, "O": 15.999,
               "F": 18.998, "Na": 22.990, "Si": 28.086, "P": 30.974, "S": 32.06,
               "Cl": 35.45, "K": 39.098, "Br": 79.904, "I": 126.904}
VDW_RADIUS = {"H": 1.20, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52,
              "F": 1.47, "Na": 2.27, "Si": 2.10, "P": 1.80, "S": 1.80,
              "Cl": 1.75, "K": 2.75, "Br": 1.85, "I": 1.98}
LABEL_TO_ELEMENT = {"CS": "C", "OW": "O", "HW": "H"}
WATER_TEMPLATE = np.array([[0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
SHORTHAND = {"*H": "*[H]", "*OH": "*O", "*NH2": "*N", "*CH3": "*C", "*SH": "*S",
             "*COOH": "*C(=O)O", "*CHO": "*C=O"}
STREAMS = {"nv": 1, "substitution": 2, "functionalization": 3, "volume": 4, "water": 5,
           "additive_mix": 6, "additive_order": 7, "additive_place": 8}
EMBED_SEED = 7
NEIGHBOURS = 16
REACH = 5.0
AZIMUTHS = 24
FREE, USED, GONE = 0, 1, 2

FCC_FRAC = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
DIAMOND_BASIS = [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]


def miller_family(n_nonzero):
    return np.array([p for p in itertools.product((1, -1, 0), repeat=3)
                     if sum(abs(c) for c in p) == n_nonzero], float)


MILLER_100, MILLER_110, MILLER_111 = (miller_family(n) for n in (1, 2, 3))
RATIO_OCTAHEDRON = np.sqrt(3.0)
RATIO_CUBE = 1.0 / np.sqrt(3.0)


def die(msg):
    sys.stderr.write("\nERROR: " + msg.rstrip() + "\n\n")
    sys.exit(2)


def unit(v):
    return v / np.linalg.norm(v)


def stream(seed, name):
    return np.random.default_rng([seed, STREAMS[name]])


def normalize_element(symbol):
    el = symbol.capitalize() if len(symbol) > 1 else symbol.upper()
    return el if el in ATOMIC_MASS else None


def label_element(label):
    head = label.split("_")[0]
    return LABEL_TO_ELEMENT.get(head) or normalize_element(head)


def sphere_volume(r):
    return (4.0 / 3.0) * np.pi * r ** 3


def rotation_about(axis, angle):
    x, y, z = unit(np.asarray(axis, float))
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rotation_to(src, dst):
    a, b = unit(np.asarray(src, float)), unit(np.asarray(dst, float))
    c = float(np.dot(a, b))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        return rotation_about(np.cross(a, ref), np.pi)
    return rotation_about(np.cross(a, b), np.arccos(c))


def spin_about_z(target):
    return rotation_about([0.0, 0.0, 1.0], -np.arctan2(target[1], target[0]))


def random_rotation(rng):
    w, x, y, z = unit(rng.standard_normal(4))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def kabsch(src, dst):
    u, _, vt = np.linalg.svd(dst.T @ src)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def wulff_planes(ratio_100, ratio_110):
    normals = np.vstack([MILLER_111 / np.sqrt(3.0), MILLER_100,
                         MILLER_110 / np.sqrt(2.0)])
    offsets = np.concatenate([np.ones(len(MILLER_111)),
                              np.full(len(MILLER_100), ratio_100),
                              np.full(len(MILLER_110), ratio_110)])
    return normals, offsets


def wulff_unit_volume(ratio_100, ratio_110):
    normals, offsets = wulff_planes(ratio_100, ratio_110)
    halfspaces = np.hstack([normals, -offsets[:, None]])
    hull = ConvexHull(HalfspaceIntersection(halfspaces, np.zeros(3)).intersections)
    return float(hull.volume)


def wulff_radii(r_angstrom, ratio_100, ratio_110):
    scale = (sphere_volume(r_angstrom)
             / wulff_unit_volume(ratio_100, ratio_110)) ** (1.0 / 3.0)
    return scale, scale * ratio_100, scale * ratio_110


def cut_lattice(r_angstrom, shape, ratio_100, ratio_110, cutoff):
    a = LATTICE_CONSTANT
    facets = None if shape == "sphere" else wulff_radii(r_angstrom, ratio_100, ratio_110)
    bound = r_angstrom if facets is None else min(
        np.sqrt(3.0) * facets[0], np.sqrt(3.0) * facets[1], np.sqrt(2.0) * facets[2])
    n = int(np.ceil((bound + a * np.sqrt(3.0)) / a))
    span = np.arange(-n, n + 1)
    cells = np.array(list(itertools.product(span, span, span)), float)
    motif = np.array([np.add(f, b) for f in FCC_FRAC for b in DIAMOND_BASIS])
    xyz = (cells[:, None, :] + motif[None, :, :]).reshape(-1, 3) * a
    if facets is None:
        keep = np.einsum("ij,ij->i", xyz, xyz) < r_angstrom ** 2 + 1e-9
    else:
        normals, offsets = wulff_planes(ratio_100, ratio_110)
        keep = np.all(xyz @ normals.T <= offsets * facets[0] + 1e-9, axis=1)
    xyz = np.array(sorted(map(tuple, np.round(xyz[keep], 10)))).reshape(-1, 3)

    n_cut = len(xyz)
    while len(xyz):
        pairs = cKDTree(xyz).query_pairs(cutoff, output_type="ndarray")
        coord = np.bincount(pairs.ravel(), minlength=len(xyz))
        if (coord > 1).all():
            break
        xyz = xyz[coord > 1]
    return xyz, n_cut - len(xyz), facets


class Particle:
    def __init__(self, xyz, cutoff):
        self.xyz = xyz
        self.nbr = [[] for _ in range(len(xyz))]
        for i, j in cKDTree(xyz).query_pairs(cutoff, output_type="ndarray"):
            self.nbr[i].append(int(j))
            self.nbr[j].append(int(i))
        self.coord = np.array([len(n) for n in self.nbr])
        self.depth = np.full(len(xyz), -1)
        frontier = [int(i) for i in np.flatnonzero(self.coord < C_VALENCE)]
        self.depth[frontier] = 0
        while frontier:
            reached = []
            for i in frontier:
                for j in self.nbr[i]:
                    if self.depth[j] < 0:
                        self.depth[j] = self.depth[i] + 1
                        reached.append(j)
            frontier = reached
        self.labels = ["CS" if c < C_VALENCE else "C" for c in self.coord]
        self.vacancy = None
        self.nitrogen = None
        self.nv = {}
        self.groups = []

    def ball(self, start, radius):
        seen, frontier = {int(start)}, [int(start)]
        for _ in range(radius):
            reached = []
            for i in frontier:
                for j in self.nbr[i]:
                    if j not in seen:
                        seen.add(j)
                        reached.append(j)
            frontier = reached
        return seen

    def origin(self):
        return np.zeros(3) if self.vacancy is None else self.xyz[self.vacancy]

    def lattice_atoms(self):
        return [(s, *p) for i, (s, p) in enumerate(zip(self.labels, self.xyz))
                if i != self.vacancy]

    def atoms(self):
        return self.lattice_atoms() + list(self.groups)


def nitrogen_site(p, vac, floor, rng):
    hosts = p.nbr[vac]
    if p.depth[vac] <= floor:
        shallowest = min(p.depth[j] for j in hosts)
        hosts = [j for j in hosts if p.depth[j] == shallowest]
    return int(hosts[rng.integers(len(hosts))])


def nv_frame(p, vac, nit, rng):
    frame = rotation_to(p.xyz[nit] - p.xyz[vac], [0.0, 0.0, 1.0])
    rest = [j for j in p.nbr[vac] if j != nit]
    carbon = rest[rng.integers(len(rest))]
    return spin_about_z(frame @ (p.xyz[carbon] - p.xyz[vac])) @ frame


def place_nv(p, cfg, rng):
    vac = int(np.argmin(np.linalg.norm(p.xyz, axis=1)))
    if p.depth[vac] < cfg.nv_min_depth:
        die(f"the centre of a {cfg.diameter:g} nm particle is only {p.depth[vac]} bond "
            f"shells below the surface, too shallow to host an NV centre "
            f"(--nv-min-depth {cfg.nv_min_depth}). Enlarge --diameter or pass --no-nv.")
    nit = nitrogen_site(p, vac, cfg.nv_min_depth, rng)
    frame = nv_frame(p, vac, nit, rng)

    theta = phi = reach = 0.0
    if cfg.nv_r > 0:
        theta = cfg.nv_theta if cfg.nv_theta is not None else float(rng.uniform(0.0, 360.0))
        phi = cfg.nv_phi if cfg.nv_phi is not None else \
            float(np.degrees(np.arccos(rng.uniform(-1.0, 1.0))))
        t, f = np.deg2rad(theta), np.deg2rad(phi)
        u = frame.T @ np.array([np.sin(f) * np.cos(t), np.sin(f) * np.sin(t), np.cos(f)])
        cand = np.flatnonzero(p.depth >= cfg.nv_min_depth)
        reach = float((p.xyz[cand] @ u).max())
        vac = int(cand[np.argmin(np.linalg.norm(
            p.xyz[cand] - cfg.nv_r * reach * u, axis=1))])
        nit = nitrogen_site(p, vac, cfg.nv_min_depth, rng)
        frame = nv_frame(p, vac, nit, rng)

    offset = float(np.linalg.norm(p.xyz[vac]))
    if cfg.nv_align:
        p.xyz = p.xyz @ frame.T
    p.vacancy, p.nitrogen = vac, nit
    p.labels[nit] = "N"
    p.nv = {
        "requested_r": float(cfg.nv_r),
        "theta_deg": float(theta),
        "phi_deg": float(phi),
        "reach_angstrom": reach,
        "realized_r": offset / reach if reach else 0.0,
        "offset_angstrom": offset,
        "depth_shells": int(p.depth[vac]),
        "nitrogen_depth_shells": int(p.depth[nit]),
        "aligned": bool(cfg.nv_align),
    }
    return p.nv


class Surface:
    def __init__(self, p):
        self.carbons = [int(i) for i in np.flatnonzero(p.coord < C_VALENCE)]
        site, dirs = [], []
        for i in self.carbons:
            V = p.xyz[p.nbr[i]] - p.xyz[i]
            if p.coord[i] == 3:
                missing = [-V.sum(axis=0)]
            elif p.coord[i] == 2:
                half = -(V[0] + V[1]) / 2.0
                side = unit(np.cross(V[0], V[1])) * np.linalg.norm(V[0] - V[1]) / 2.0
                missing = [half + side, half - side]
            else:
                die(f"lattice site {i} has {p.coord[i]} bonds after pruning; the bond "
                    f"cutoff does not match the lattice.")
            for m in missing:
                site.append(i)
                dirs.append(unit(m))
        self.site = np.array(site, int)
        self.dirs = np.array(dirs, float).reshape(-1, 3)
        self.state = np.full(len(site), FREE)
        self.of = collections.defaultdict(list)
        for b, i in enumerate(site):
            self.of[i].append(b)
        ghosts = p.xyz[self.site] + BOND_LENGTH * self.dirs
        self.census = {
            "undercoordinated_carbons": len(self.carbons),
            "three_coordinate": int((p.coord == 3).sum()),
            "two_coordinate": int((p.coord == 2).sum()),
            "dangling_bonds": len(site),
            "convergent_pairs": len(cKDTree(ghosts).query_pairs(0.05)) if len(site) else 0,
        }


class Environment:
    def __init__(self, p, surf):
        holders = p.xyz[surf.site] + CH_BOND * surf.dirs
        self.n_lattice = len(p.xyz)
        self.xyz = np.vstack([p.xyz, holders])
        self.radii = np.array([VDW_RADIUS[label_element(s)] for s in p.labels]
                              + [VDW_RADIUS["H"]] * len(holders))
        self.hydrogen = np.arange(len(self.xyz)) >= self.n_lattice
        self.alive = np.ones(len(self.xyz), bool)
        if p.vacancy is not None:
            self.alive[p.vacancy] = False
        self.tree = cKDTree(self.xyz)
        self.added_xyz = np.zeros((0, 3))
        self.added_radii = np.zeros(0)
        self.added_hydrogen = np.zeros(0, bool)
        self.added_tree = None

    def add(self, xyz, radii, hydrogen):
        self.added_xyz = np.vstack([self.added_xyz, xyz])
        self.added_radii = np.concatenate([self.added_radii, radii])
        self.added_hydrogen = np.concatenate([self.added_hydrogen, hydrogen])
        self.added_tree = None

    def clearance(self, points, radii, hydrogen, exclude=()):
        if self.added_tree is None and len(self.added_xyz):
            self.added_tree = cKDTree(self.added_xyz)
        block = np.full(len(points), np.inf)
        score = np.full(len(points), np.inf)
        for tree, env_radii, env_hydrogen, alive in (
                (self.tree, self.radii, self.hydrogen, self.alive),
                (self.added_tree, self.added_radii, self.added_hydrogen, None)):
            if tree is None:
                continue
            d, idx = tree.query(points, k=NEIGHBOURS, distance_upper_bound=REACH)
            valid = np.isfinite(d)
            idx = np.where(valid, idx, 0)
            if alive is not None:
                valid &= alive[idx]
                for skip in exclude:
                    valid &= idx != skip
            ratio = np.where(valid, d / (radii[:, None] + env_radii[idx]), np.inf)
            score = np.minimum(score, ratio.min(axis=1))
            pair_of_h = hydrogen[:, None] & env_hydrogen[idx]
            block = np.minimum(block, np.where(pair_of_h, np.inf, ratio).min(axis=1))
        return block, score


def enclosed(text):
    depth = 0
    for i, ch in enumerate(text):
        depth += (ch == "[") - (ch == "]")
        if depth == 0:
            return ch == "]" and i == len(text) - 1
    return False


def array(tokens, flag):
    if tokens is None:
        return None
    text = " ".join(tokens).strip()
    if not enclosed(text):
        items = [t if re.fullmatch(r"[\w.+/-]+", t) else f'"{t}"'
                 for t in re.split(r"[,\s]+", text[text.startswith("["):]) if t]
        die(f"{flag} takes an array such as {flag} [{','.join(items)}] ([] for none), "
            f"not '{text}'.")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [t if isinstance(t, str) else json.dumps(t) for t in parsed]
    except ValueError:
        pass
    inner = text[1:-1].strip()
    items, depth, start = [], 0, 0
    for i, ch in enumerate(inner + ","):
        depth += (ch == "[") - (ch == "]")
        if ch == "," and depth == 0 and inner:
            items.append(inner[start:i].strip().strip("\"'"))
            start = i + 1
    if "" in items:
        die(f"{flag} array '{text}' has an empty entry.")
    if any(re.search(r"\s", t) and not enclosed(t) for t in items):
        die(f"{flag} array '{text}' must separate its entries with commas.")
    return items


def condensed_formula(elements):
    counts = collections.Counter(elements)
    order = []
    for el in elements:
        if el != "H" and el not in order:
            order.append(el)
    if counts["H"]:
        order.append("H")
    return "".join(el + (str(counts[el]) if counts[el] > 1 else "") for el in order)


def embed(mol, n_conf):
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, randomSeed=EMBED_SEED))
    closed = Chem.RWMol(mol)
    for n in closed.GetAtoms():
        for o in n.GetNeighbors():
            if n.GetSymbol() == "N" and o.GetSymbol() == "O" and o.GetNumRadicalElectrons():
                closed.GetBondBetweenAtoms(n.GetIdx(), o.GetIdx()).SetBondType(
                    Chem.BondType.DOUBLE)
                n.SetFormalCharge(1)
                o.SetNumRadicalElectrons(0)
    closed = closed.GetMol()
    try:
        Chem.SanitizeMol(closed)
        AllChem.UFFOptimizeMoleculeConfs(closed, maxIters=2000)
    except Exception:
        return [mol.GetConformer(i).GetPositions() for i in ids]
    return [closed.GetConformer(i).GetPositions() for i in ids]


def stand_in(mol, stars, element, flag, text):
    standin = Chem.RWMol(mol)
    for s in stars:
        atom = standin.GetAtomWithIdx(s)
        atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(element))
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        atom.SetIsotope(0)
        atom.SetFormalCharge(0)
    standin = standin.GetMol()
    try:
        Chem.SanitizeMol(standin)
    except Exception as exc:
        die(f"{flag} entry '{text}' is not a chemically valid group: {exc}")
    full = Chem.AddHs(standin)
    padding = {n.GetIdx() for s in stars for n in full.GetAtomWithIdx(s).GetNeighbors()
               if n.GetAtomicNum() == 1 and n.GetIdx() >= mol.GetNumAtoms()}
    group = [i for i in range(full.GetNumAtoms()) if i not in padding and i not in stars]
    return standin, full, group


def template(e, element):
    if element not in e["confs"]:
        standin, full, group = stand_in(e["mol"], [e["star"]], element,
                                        "--functionalization", e["input"])
        rotors = rdMolDescriptors.CalcNumRotatableBonds(standin)
        shapes = embed(full, 1 if not rotors else min(8 * rotors, 16))
        if not shapes:
            die(f"--functionalization entry '{e['input']}': RDKit could not build a 3D "
                f"geometry for it on a {element} site.")
        e["confs"][element] = [(X[group] - X[e["star"]], X[e["arms"]] - X[e["star"]])
                               for X in shapes]
    return e["confs"][element]


def build_pattern(text, kind, alias, fraction, sites):
    flag = f"--{kind}"
    mol = Chem.MolFromSmiles(SHORTHAND.get(text, text))
    if mol is None:
        die(f"{flag} entry '{text}' is not valid SMILES. Hydrogens are implicit or "
            f"bracketed: '*O' is a hydroxyl, '*[H]' a hydrogen, '*C(=O)O' a carboxylic "
            f"acid, '*O*' an ether oxygen.")
    smiles = Chem.MolToSmiles(mol)
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if not stars:
        die(f"{flag} entry '{text}' has no '*' marking where it meets the diamond.")
    if kind == "functionalization":
        if len(stars) != 1:
            die(f"{flag} entry '{text}' has {len(stars)} '*'. A functional group has "
                f"one, the surface atom it binds to; a group that takes the place of "
                f"a carbon belongs in --substitution.")
        arms = sorted(n.GetIdx() for n in mol.GetAtomWithIdx(stars[0]).GetNeighbors())
        if not arms:
            die(f"{flag} entry '{text}': '*' is bonded to nothing. Write the group "
                f"as a separate atom, e.g. '[*][H]' for a hydrogen, not '[*H]'.")
        if len(arms) > 3:
            die(f"{flag} entry '{text}': '*' has {len(arms)} groups bonded to it, so "
                f"no connection point is left to hold it on the surface; three is "
                f"the most.")
        first = "C" if not sites or "C" in sites else sites[0]
    else:
        anchors = {n.GetIdx() for s in stars for n in mol.GetAtomWithIdx(s).GetNeighbors()}
        if (len(anchors) != 1 or any(mol.GetAtomWithIdx(s).GetDegree() != 1 for s in stars)
                or mol.GetAtomWithIdx(min(anchors)).GetAtomicNum() == 0):
            die(f"{flag} entry '{text}': every '*' must be bonded to one and the same "
                f"atom of the group. To leave sites bare, lower the ratio instead.")
        if len(stars) not in (2, 3):
            die(f"{flag} entry '{text}' asks for a carbon with {len(stars)} carbon "
                f"neighbours. Surface carbons have 2 or 3; use '*X*' or '*X(*)*'.")
        anchor = min(anchors)

    standin, full, group = stand_in(
        mol, stars, first if kind == "functionalization" else "C", flag, text)
    elements = [full.GetAtomWithIdx(i).GetSymbol() for i in group]
    unknown = sorted({el for el in elements if el not in VDW_RADIUS})
    if unknown:
        die(f"{flag} entry '{text}' contains {', '.join(unknown)}, which this script "
            f"has no mass or radius for.")

    tag = alias or condensed_formula(elements)
    entry = {"input": text, "smiles": smiles, "kind": kind,
             "stars": len(stars), "fraction": fraction, "tag": tag,
             "aliased": alias is not None, "formula": condensed_formula(elements)}

    members = [i for i in group if i != anchor] if kind == "substitution" else group
    member_elements = [full.GetAtomWithIdx(i).GetSymbol() for i in members]
    entry["elements"] = member_elements
    entry["radii"] = np.array([VDW_RADIUS[el] for el in member_elements])
    entry["heavy"] = np.array([el != "H" for el in member_elements], bool)
    entry["lone_h"] = kind == "functionalization" and member_elements == ["H"]
    entry["labels"] = ["H" if entry["lone_h"] and tag == "H" else f"{el}_{tag}"
                       for el in member_elements]

    if kind == "functionalization":
        entry.update(mol=mol, star=stars[0], arms=arms, bonds=len(arms), sites=sites,
                     confs={})
        template(entry, first)
        return entry

    rotors = rdMolDescriptors.CalcNumRotatableBonds(standin)
    shapes = embed(full, 1 if not rotors else min(8 * rotors, 16)) if members else []
    if members and not shapes:
        die(f"{flag} entry '{text}': RDKit could not build a 3D geometry for it.")
    atom = full.GetAtomWithIdx(anchor)
    entry["anchor_element"] = atom.GetSymbol()
    entry["anchor_label"] = f"{entry['anchor_element']}_{tag}"
    entry["free"] = atom.GetNumRadicalElectrons()
    entry["shell"] = [members.index(n.GetIdx()) for n in atom.GetNeighbors()
                      if n.GetIdx() in members]
    entry["confs"] = [(np.array([unit(X[s] - X[anchor]) for s in stars]),
                       X[members] - X[anchor]) for X in shapes]
    return entry


def allot(fractions, total):
    capped = [f for f in fractions if f is not None]
    exact = np.array(capped, float) * total
    if len(capped) == len(fractions) and abs(sum(capped) - 1.0) <= 1e-9:
        counts = np.floor(exact + 1e-9).astype(int)
        for i in np.argsort(-(exact - counts), kind="stable")[:total - counts.sum()]:
            counts[i] += 1
    else:
        counts = np.floor(exact + 0.5).astype(int)
    counts = iter(int(x) for x in counts)
    return [total if f is None else next(counts) for f in fractions]


def resolve_fractions(tokens, n, flag):
    items = array(tokens, flag)
    values = []
    for t in ["1"] * n if items is None else items:
        try:
            values.append(None if t.lower() == "rest" else float(t))
        except ValueError:
            die(f"{flag} got '{t}', which is neither a number nor 'rest'.")
    if len(values) != n:
        die(f"{flag} lists {len(values)} values for {n} entries; they must match "
            f"one-to-one.")
    total = sum(v for v in values if v is not None)
    if any(v is not None and v < 0 for v in values):
        die(f"{flag} entries must be >= 0.")
    return [v if v is None else v / max(total, 1.0) for v in values]


def resolve_aliases(tokens, n, flag):
    aliases = array(tokens, flag)
    if aliases is None:
        return [None] * n
    if len(aliases) != n:
        die(f"{flag} lists {len(aliases)} aliases for {n} entries; they must match "
            f"one-to-one.")
    for alias in aliases:
        if not re.fullmatch(r"[A-Za-z0-9_]+", alias):
            die(f"{flag} '{alias}' must be letters, digits and underscores only.")
    return aliases


def resolve_attachments(tokens, n, flag):
    lists = array(tokens, flag)
    if lists is None:
        return [None] * n
    if len(lists) != n:
        die(f"{flag} lists {len(lists)} arrays for {n} entries; they must match "
            f"one-to-one.")
    sites = []
    for text in lists:
        if not enclosed(text):
            die(f"{flag} takes one array of elements per entry, e.g. {flag} "
                f"[[C,N],[],[C]], where [] allows any element; got '{text}'.")
        symbols = array([text], flag)
        unknown = [s for s in symbols if normalize_element(s) not in VDW_RADIUS]
        if unknown:
            die(f"{flag} names {', '.join(unknown)}, which this script has no mass or "
                f"radius for. An empty array [] allows any element.")
        sites.append(sorted({normalize_element(s) for s in symbols}) or None)
    return sites


def parse_patterns(cfg, kind, default=()):
    texts = array(getattr(cfg, kind), f"--{kind}")
    if texts is None:
        texts = list(default)
    fractions = resolve_fractions(getattr(cfg, f"{kind}_ratio"), len(texts),
                                  f"--{kind}-ratio")
    aliases = resolve_aliases(getattr(cfg, f"{kind}_alias"), len(texts),
                              f"--{kind}-alias")
    sites = resolve_attachments(cfg.functionalization_attachments, len(texts),
                                "--functionalization-attachments") \
        if kind == "functionalization" else [None] * len(texts)
    order = parse_int_list(getattr(cfg, f"{kind}_order"), f"--{kind}-order") \
        or list(range(1, len(texts) + 1))
    if sorted(order) != list(range(1, len(texts) + 1)):
        die(f"--{kind}-order must give each of the {len(texts)} entries its turn, "
            f"using each of 1 to {len(texts)} once, e.g. "
            f"[{','.join(map(str, range(len(texts), 0, -1)))}].")
    entries = [build_pattern(t, kind, a, f, s)
               for t, a, f, s in zip(texts, aliases, fractions, sites)]
    return [e for _, e in sorted(zip(order, entries), key=lambda pair: pair[0])]


def best_clear(env, e, pts, exclude, scale):
    block, score = env.clearance(pts.reshape(-1, 3), np.tile(e["radii"], len(pts)),
                                 np.tile(~e["heavy"], len(pts)), exclude)
    ok = (block.reshape(len(pts), -1) >= scale).all(axis=1)
    if not ok.any():
        return None
    return pts[int(np.argmax(np.where(ok, score.reshape(len(pts), -1).min(axis=1), -np.inf)))]


def orient_substituents(p, surf, env, e, site, scale):
    if not len(e["elements"]):
        return np.zeros((0, 3))
    origin = p.xyz[site]
    target = np.array([unit(p.xyz[j] - origin) for j in p.nbr[site]])
    cands = []
    for arms, members in e["confs"]:
        fits = []
        for perm in itertools.permutations(range(len(arms))):
            R = kabsch(arms[list(perm)], target)
            fits.append((float(((arms[list(perm)] @ R.T - target) ** 2).sum()), R))
        best = min(f[0] for f in fits)
        cands += [origin + members @ R.T for miss, R in fits if miss <= best + 1e-6]
    return best_clear(env, e, np.array(cands),
                      [site] + [env.n_lattice + b for b in surf.of[site]], scale)


def substitute(p, surf, env, entries, cfg, rng):
    n_sites = len(surf.carbons)
    protected = p.ball(p.vacancy, 2) if p.vacancy is not None else set()
    pools = {}
    for e, cap in zip(entries, allot([e["fraction"] for e in entries], n_sites)):
        e["requested"] = cap
        e["placed"] = e["spaced_out"] = e["rejected"] = 0
        pools.setdefault(e["stars"], [i for i in surf.carbons
                                      if p.coord[i] == e["stars"] and i not in protected])
        e["eligible"] = len(pools[e["stars"]])
    zone, taken_sites = set(), set()
    for e in entries:
        pool = pools[e["stars"]]
        for site in (rng.permutation(pool) if pool else []):
            if e["placed"] == e["requested"]:
                break
            site = int(site)
            if site in taken_sites:
                continue
            if site in zone:
                e["spaced_out"] += 1
                continue
            placed = orient_substituents(p, surf, env, e, site, cfg.contact_scale)
            if placed is None:
                e["rejected"] += 1
                continue
            p.labels[site] = e["anchor_label"]
            env.radii[site] = VDW_RADIUS[e["anchor_element"]]
            taken = {max(surf.of[site], key=lambda b: surf.dirs[b] @ (x - p.xyz[site]))
                     for x in placed[e["shell"]]}
            spare = [b for b in surf.of[site] if b not in taken][:e["free"]]
            for bond in surf.of[site]:
                if bond not in spare:
                    surf.state[bond] = GONE
                    env.alive[env.n_lattice + bond] = False
            if len(placed):
                env.add(placed, e["radii"], ~e["heavy"])
                p.groups += [(lab, *xyz) for lab, xyz in zip(e["labels"], placed)]
            zone |= p.ball(site, cfg.substitution_spacing - 1)
            taken_sites.add(site)
            e["placed"] += 1
    return n_sites


def orient_group(env, e, confs, origin, options, exclude, scale, turns):
    if e["lone_h"]:
        return (origin + np.linalg.norm(confs[0][1][0]) * options[0][0])[None, :]
    cands = []
    for members, arms in confs:
        for dirs in options:
            if len(dirs) > 1:
                fit = kabsch(arms / np.linalg.norm(arms, axis=1)[:, None], dirs)
                cands.append(origin + members @ fit.T)
                continue
            base = members @ rotation_to(arms[0], dirs[0]).T
            cands += [origin + base] if len(members) == 1 else \
                [origin + base @ rotation_about(dirs[0], t).T for t in turns]
    return best_clear(env, e, np.array(cands), exclude, scale)


def functionalize(p, surf, env, entries, cfg, rng):
    free = [int(b) for b in np.flatnonzero(surf.state == FREE)]
    element = {int(i): label_element(p.labels[i]) for i in set(surf.site[free])}

    def spare(site):
        return [b for b in surf.of[site] if surf.state[b] == FREE]

    for e, cap in zip(entries, allot([e["fraction"] for e in entries], len(free))):
        e["requested"] = cap
        e["placed"] = e["rejected"] = 0
        fits = [i for i, el in element.items()
                if (not e["sites"] or el in e["sites"]) and len(spare(i)) >= e["bonds"]]
        e["eligible"] = len(fits) if e["bonds"] > 1 else sum(len(spare(i)) for i in fits)
    turns = np.linspace(0.0, 2.0 * np.pi, AZIMUTHS, endpoint=False)
    crowded = set()

    for e in entries:
        queue = iter(rng.permutation(free))
        bulky = bool(e["heavy"].any())
        for _ in range(e["requested"]):
            for b in queue:
                b = int(b)
                site = int(surf.site[b])
                bonds = [b] if e["bonds"] == 1 else spare(site)[:e["bonds"]]
                if (surf.state[b] != FREE or len(bonds) < e["bonds"] or b != bonds[0]
                        or (e["sites"] and element[site] not in e["sites"])
                        or (bulky and site in crowded)):
                    continue
                placed = orient_group(
                    env, e, template(e, element[site]), p.xyz[site],
                    [surf.dirs[list(o)] for o in itertools.permutations(bonds)],
                    [site] + [env.n_lattice + x for x in bonds], cfg.contact_scale, turns)
                if placed is None:
                    e["rejected"] += 1
                    continue
                for x in bonds:
                    surf.state[x] = USED
                    env.alive[env.n_lattice + x] = False
                env.add(placed, e["radii"], ~e["heavy"])
                p.groups += [(lab, *xyz) for lab, xyz in zip(e["labels"], placed)]
                if bulky:
                    crowded.add(site)
                e["placed"] += 1
                break
    return len(free), int((surf.state == FREE).sum())


def asked(e):
    if e["fraction"] is None:
        return f"{'the rest':<15s}"
    return f"{100 * e['fraction']:5.1f}% = {e['requested']:<5d}"


def report_surface(surf, substitutions, functions, n_sites, n_left, n_bare):
    c = surf.census
    print(f"Surface: {c['undercoordinated_carbons']} undercoordinated carbons "
          f"({c['three_coordinate']} with 3 carbon neighbours, {c['two_coordinate']} "
          f"with 2), {c['dangling_bonds']} dangling bonds, {c['convergent_pairs']} "
          f"convergent pairs")
    if substitutions:
        print(f"Substitution, as a fraction of the {n_sites} undercoordinated carbons:")
        for e in substitutions:
            why = []
            if e["placed"] < e["requested"] and e["fraction"] is not None:
                why = [w for w, hit in (("spacing rule", e["spaced_out"]),
                                        ("no clash-free orientation", e["rejected"])) if hit]
                why = why or [f"only {e['eligible']} such carbons"]
            print(f"  {e['tag']:<8s} {e['input']:<22s} needs {e['stars']} carbon "
                  f"neighbours ({e['eligible']} eligible)   asked {asked(e)} "
                  f"placed {e['placed']}" + (f"   <- {', '.join(why)}" if why else ""))
    print(f"Functionalization, as a fraction of the {n_left} dangling bonds left:")
    for e in functions:
        short = e["placed"] < e["requested"] and e["fraction"] is not None
        plain = not e["sites"] and e["bonds"] == 1
        print(f"  {e['tag']:<8s} {e['input']:<22s} "
              + ("" if plain else
                 f"needs {' or '.join(e['sites'] or ['any atom'])} with {e['bonds']} "
                 f"dangling bond{'s' * (e['bonds'] > 1)} ({e['eligible']} eligible)   ")
              + f"asked {asked(e)} placed {e['placed']}"
              + ("   <- no clash-free site left" if short and e["rejected"] else
                 "   <- no dangling bond left" if short and plain else
                 "   <- no such site left" if short else ""))
    print(f"  {'bare':<8s} {'':<22s} {n_bare} dangling bonds left uncapped")


def size_metrics(labels, xyz, origin):
    if len(xyz) < 4:
        return None
    m = np.array([ATOMIC_MASS[label_element(s)] for s in labels])
    com = m @ xyz / m.sum()
    d = xyz - com
    evals = np.linalg.eigvalsh((d * m[:, None]).T @ d / m.sum())[::-1]
    rg = float(np.sqrt(evals.sum()))
    hull = ConvexHull(xyz)
    return {
        "n_atoms": len(xyz),
        "rg_angstrom": rg,
        "d_equiv_rg_nm": 2.0 * np.sqrt(5.0 / 3.0) * rg / NM_TO_ANGSTROM,
        "r_max_angstrom": float(np.linalg.norm(xyz - origin, axis=1).max()),
        "r_inscribed_angstrom": float(-(hull.equations[:, 3]
                                        + hull.equations[:, :3] @ origin).max()),
        "com_offset_angstrom": float(np.linalg.norm(com - origin)),
        "gyration_eigenvalues": [float(v) for v in evals],
    }


def to_extxyz(atoms, box_length, fields, wrap=False):
    xyz = np.array([a[1:] for a in atoms], float).reshape(-1, 3)
    L = float(box_length)
    n_wrapped = 0
    if wrap and len(xyz):
        shifted = xyz - L * np.floor(xyz / L + 0.5)
        n_wrapped = int((np.abs(shifted - xyz) > 1e-9).any(axis=1).sum())
        xyz = shifted
    header = (f'Lattice="{L:.6f} 0.0 0.0  0.0 {L:.6f} 0.0  0.0 0.0 {L:.6f}" '
              f'Origin="{-L / 2:.6f} {-L / 2:.6f} {-L / 2:.6f}" '
              f'pbc="T T T" Properties=species:S:1:pos:R:3 {fields}')
    lines = [str(len(atoms)), header]
    lines += [f"{a[0]} {x:.6f} {y:.6f} {z:.6f}" for a, (x, y, z) in zip(atoms, xyz)]
    return "\n".join(lines) + "\n", n_wrapped


def n_water_to_volume(n_water, density):
    return n_water * MOLAR_MASS_H2O / (AVOGADRO * density * ANGSTROM3_TO_CM3)


def excluded_volume(xyz, tree, gap, rng, samples=400000):
    lo, hi = xyz.min(axis=0) - gap, xyz.max(axis=0) + gap
    pts = rng.uniform(lo, hi, (samples, 3))
    hit = np.isfinite(tree.query(pts, distance_upper_bound=gap)[0])
    return float(np.prod(hi - lo) * hit.mean())


def composition(atoms):
    counts = collections.Counter(label_element(a[0]) for a in atoms)
    if None in counts:
        bad = sorted({a[0] for a in atoms if label_element(a[0]) is None})
        die(f"unknown atom label(s) {bad} while totalling the system mass.")
    return dict(counts), sum(ATOMIC_MASS[el] * n for el, n in counts.items())


def hill_formula(counts):
    counts = {el: n for el, n in counts.items() if n}
    order = [el for el in ("C", "H") if el in counts]
    order += sorted(el for el in counts if el not in ("C", "H"))
    return "".join(f"{el}{counts[el]}" if counts[el] > 1 else el for el in order)


def write_text(path, text):
    with open(path, "w") as f:
        f.write(text)


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


def parse_int_list(tokens, flag):
    values = []
    for t in array(tokens, flag) or []:
        try:
            values.append(int(t))
        except ValueError:
            die(f"{flag} got '{t}', which is not an integer.")
    return values


def parse_float_list(tokens, flag):
    values = []
    for t in array(tokens, flag) or []:
        try:
            values.append(float(t))
        except ValueError:
            die(f"{flag} got '{t}', which is not a number.")
    return values


def apportion(weights, total, rng):
    w = np.array(weights, float)
    if (w < 0).any():
        die("--additive-ratios entries must be >= 0.")
    if w.sum() <= 0:
        die("--additive-ratios must contain at least one positive entry.")
    exact = w / w.sum() * total
    counts = np.floor(exact).astype(int)
    leftover = total - counts.sum()
    if leftover:
        order = np.lexsort((rng.random(len(w)), -(exact - counts)))
        counts[order[:leftover]] += 1
    return counts.tolist()


def resolve_additives(cfg):
    paths = array(cfg.additives, "--additives") or []
    counts = parse_int_list(cfg.n_additives, "--n-additives")
    ratios = parse_float_list(cfg.additive_ratios, "--additive-ratios")
    if not paths:
        if counts or ratios or array(cfg.additive_alias, "--additive-alias"):
            die("--n-additives/--additive-ratios/--additive-alias were given without "
                "--additives.")
        return []
    aliases = resolve_aliases(cfg.additive_alias, len(paths), "--additive-alias")

    if ratios:
        if len(ratios) != len(paths):
            die(f"--additives lists {len(paths)} files but --additive-ratios lists "
                f"{len(ratios)} weights; they must match one-to-one.")
        if len(counts) != 1:
            die("--additive-ratios sets the mix, so --n-additives must be a single "
                "total, e.g. --additive-ratios [1,1,1,1] --n-additives [40].")
        if counts[0] < 0:
            die("--n-additives total must be >= 0.")
        counts = apportion(ratios, counts[0], stream(cfg.additive_seed, "additive_mix"))
    else:
        if not counts:
            die("--additives was given without --n-additives; state a count per file, "
                "or give --additive-ratios plus a single --n-additives total.")
        if len(paths) != len(counts):
            die(f"--additives lists {len(paths)} files but --n-additives lists "
                f"{len(counts)} counts; they must match one-to-one.")
        ratios = [None] * len(paths)

    specs = []
    for path, n, ratio, alias in zip(paths, counts, ratios, aliases):
        if n < 0:
            die(f"--n-additives entry for '{path}' is negative.")
        tag = alias or os.path.splitext(os.path.basename(path))[0]
        atoms = read_xyz(path)
        elements = [normalize_element(s) for s, *_ in atoms]
        if None in elements:
            die(f"additive '{tag}' contains an element this script has no mass for.")
        xyz = np.array([a[1:] for a in atoms], float)
        specs.append({
            "tag": tag,
            "aliased": alias is not None,
            "path": os.path.abspath(path),
            "n": int(n),
            "ratio": ratio,
            "n_atoms": len(atoms),
            "labels": [f"{el}_{tag}" for el in elements],
            "formula": hill_formula(collections.Counter(elements)),
            "volume_angstrom3": sum(sphere_volume(VDW_RADIUS[el]) for el in elements),
            "mass_amu": sum(ATOMIC_MASS[el] for el in elements),
            "radius_angstrom": float(np.linalg.norm(xyz - xyz.mean(axis=0), axis=1).max()),
            "xyz": xyz - xyz.mean(axis=0),
        })
    return specs


def periodic_tree(xyz, L):
    wrapped = (xyz + L / 2.0) % L
    return cKDTree(np.where(wrapped < L, wrapped, 0.0), boxsize=L)


def image_overlap(xyz, L, gap):
    pairs = periodic_tree(xyz, L).query_pairs(gap, output_type="ndarray")
    delta = xyz[pairs[:, 0]] - xyz[pairs[:, 1]]
    across = np.linalg.norm(delta, axis=1) >= gap
    if not across.any():
        return None
    pairs, delta = pairs[across], delta[across]
    d = np.linalg.norm(delta - L * np.round(delta / L), axis=1)
    k = int(np.argmin(d))
    return float(d[k]), int(pairs[k, 0]), int(pairs[k, 1])


def place_additives(cfg, box_length, specs, np_tree):
    half = box_length / 2.0 - cfg.additive_margin
    order = np.repeat(np.arange(len(specs)), [s["n"] for s in specs])
    stream(cfg.additive_seed, "additive_order").shuffle(order)
    rng = stream(cfg.additive_seed, "additive_place")

    placed = np.zeros((0, 3))
    labeled = []
    for slot, si in enumerate(order):
        s = specs[si]
        for _ in range(cfg.additive_attempts):
            m = s["xyz"] @ random_rotation(rng).T + rng.uniform(-half, half, 3)
            if np.abs(m).max() > half:
                continue
            image = m + box_length / 2.0
            if np_tree.query(image)[0].min() < cfg.surface_gap:
                continue
            if len(placed) and periodic_tree(placed, box_length).query(
                    image)[0].min() < cfg.additive_gap:
                continue
            placed = np.vstack([placed, m])
            labeled += [(lab, *xyz) for lab, xyz in zip(s["labels"], m)]
            break
        else:
            die(f"could not place {s['tag']} (slot {slot + 1} of {len(order)}) in "
                f"{cfg.additive_attempts} random tries without a clash.\n"
                f"Enlarge --box, lower --n-additives, or raise "
                f"--additive-attempts above {cfg.additive_attempts}.")

    return labeled


def water_frame(cfg):
    if cfg.water_xyz is None:
        return WATER_TEMPLATE.copy()
    atoms = read_xyz(cfg.water_xyz)
    xyz = np.array([a[1:] for a in atoms], float)
    order = np.argsort([normalize_element(a[0]) != "O" for a in atoms], kind="stable")
    if len(atoms) != 3 or [normalize_element(atoms[i][0]) for i in order] != ["O", "H", "H"]:
        die(f"--water-xyz '{cfg.water_xyz}' is not a single three-atom water molecule.")
    return xyz[order][1:] - xyz[order][0]


def water_atoms_of(O, H):
    out = []
    for o, hs in zip(O, H):
        out.append(("OW", *o))
        out += [("HW", *h) for h in hs]
    return out


def hbonds_per_water(O, H):
    if not len(O):
        return 0.0
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


def resolve_water_box(path):
    if os.path.exists(path) or os.path.isabs(path):
        return path
    beside = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    return beside if os.path.exists(beside) else path


def read_water_box(path):
    path = resolve_water_box(path)
    try:
        lines = open(path).read().splitlines()
    except OSError as exc:
        die(f"cannot read --water-box '{path}': {exc}\n"
            f"A pre-equilibrated box is required for --water-model equilibrated_liquid.\n"
            f"The standard one is spc216.gro from the GROMACS distribution:\n"
            f"  curl -O https://gitlab.com/gromacs/gromacs/-/raw/main/share/top/spc216.gro")

    if path.lower().endswith(".gro"):
        n = int(lines[1].split()[0])
        box = np.array([float(v) for v in lines[2 + n].split()[:3]]) * NM_TO_ANGSTROM
        sym = [lines[2 + i][10:15].strip() for i in range(n)]
        xyz = np.array([[float(lines[2 + i][c:c + 8]) for c in (20, 28, 36)]
                        for i in range(n)]) * NM_TO_ANGSTROM
    else:
        atoms = read_xyz(path)
        m = re.search(r'Lattice="([^"]+)"', lines[1])
        if not m:
            die(f"--water-box '{path}' has no Lattice=\"...\" in its comment line, so "
                f"its periodic box is unknown. Use a .gro box instead.")
        v = [float(x) for x in m.group(1).split()]
        box = np.array([v[0], v[4], v[8]])
        sym = [a[0] for a in atoms]
        xyz = np.array([a[1:] for a in atoms], float)

    el = [label_element(s) or (s[:1].upper() if s[:1].upper() in ATOMIC_MASS else None)
          for s in sym]
    if len(el) % 3 or el[::3] != ["O"] * (len(el) // 3) \
            or set(el[1::3]) | set(el[2::3]) != {"H"}:
        die(f"--water-box '{path}' is not a plain O,H,H-ordered water box "
            f"(first atoms parsed as {el[:6]}).")
    W = xyz.reshape(-1, 3, 3)
    return W[:, 0] % box, W[:, 1:] - W[:, :1], box


def tile_water(cfg, L, h_local, np_tree, additive_tree, O_src, H_src, src_box):
    lo = np.floor(-L / 2.0 / src_box).astype(int)
    hi = np.floor(L / 2.0 / src_box).astype(int)
    grid = [np.arange(lo[a], hi[a] + 1) for a in range(3)]
    shifts = np.array(list(itertools.product(*grid)), float) * src_box
    O = (O_src[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    Hrel = np.tile(H_src, (len(shifts), 1, 1))

    keep = np.abs(O).max(axis=1) < L / 2.0
    O, Hrel = O[keep], Hrel[keep]
    n_tiled = len(O)

    h_dirs = h_local / np.linalg.norm(h_local, axis=1)[:, None]
    H = np.stack([o + h_local @ kabsch(h_dirs,
                                      hr / np.linalg.norm(hr, axis=1)[:, None]).T
                  for o, hr in zip(O, Hrel)]) if n_tiled else np.zeros((0, 2, 3))

    mol = np.concatenate([O[:, None, :], H], axis=1).reshape(-1, 3) + L / 2.0
    keep = np.ones(n_tiled, bool)
    if n_tiled:
        keep &= np_tree.query(mol)[0].reshape(-1, 3).min(axis=1) >= cfg.surface_gap
        if additive_tree is not None:
            keep &= additive_tree.query(mol)[0].reshape(-1, 3).min(axis=1) >= cfg.additive_gap
    O, H = O[keep], H[keep]
    n_carved = len(O)

    drop = set()
    if n_carved:
        for i, j in periodic_tree(O, L).query_pairs(
                cfg.water_box_clash, output_type="ndarray"):
            if i not in drop and j not in drop:
                drop.add(int(j))
    if drop:
        mask = np.ones(len(O), bool)
        mask[list(drop)] = False
        O, H = O[mask], H[mask]
    return O, H, (n_tiled, n_carved, len(drop), [len(g) for g in grid])


def tiled_water(cfg, L, h_local, solutes, grow, rng):
    # spc216.gro: 216-molecule SPC water box equilibrated at 300 K, from GROMACS
    # share/top; SPC model of Berendsen, Postma, van Gunsteren & Hermans,
    # "Interaction models for water in relation to protein hydration",
    # in Intermolecular Forces (Reidel, 1981) 331-342.
    O_src, H_src, box_src = read_water_box(cfg.water_box)
    rho_src = (len(O_src) * MOLAR_MASS_H2O
               / (AVOGADRO * np.prod(box_src) * ANGSTROM3_TO_CM3))
    scale = (rho_src / cfg.water_density) ** (1.0 / 3.0)
    for attempt in range(12):
        if attempt:
            L = (L ** 3 + n_water_to_volume((cfg.n_water - len(O)) * 2 ** (attempt - 1),
                                            cfg.water_density)) ** (1.0 / 3.0)
        np_tree, additive_atoms, additive_tree = solutes(L)
        O, H, (n_tiled, n_carved, n_seam, reps) = tile_water(
            cfg, L, h_local, np_tree, additive_tree, O_src * scale, H_src, box_src * scale)
        if not grow or len(O) >= cfg.n_water:
            break

    surplus = 0 if cfg.n_water is None else max(len(O) - cfg.n_water, 0)
    if surplus:
        tight = periodic_tree(O, L).query(O + L / 2.0, k=2)[0][:, 1]
        clashing = np.flatnonzero(tight < cfg.water_box_clash + 0.3)
        clashing = clashing[np.argsort(tight[clashing], kind="stable")]
        rest = rng.permutation(np.setdiff1d(np.arange(len(O)), clashing))
        keep = np.setdiff1d(np.arange(len(O)), np.concatenate([clashing, rest])[:surplus])
        O, H = O[keep], H[keep]

    summary = None if cfg.quiet else (
        f"Tiled {os.path.basename(cfg.water_box)}: {len(O_src)} waters at "
        f"{rho_src:.4f} g/cm^3, affine x{scale:.5f} -> {cfg.water_density:.4f} g/cm^3, "
        f"{'x'.join(map(str, reps))} cells, {n_tiled} placed, "
        f"{n_tiled - n_carved} carved out, {n_seam} dropped across the periodic "
        f"boundary, " + (f"{surplus} trimmed to --n-water, " if surplus else "")
        + f"{len(O)} kept, H-bonds per water {hbonds_per_water(O, H):.2f}")
    tile = box_src * scale
    over = float(max(L % tile))
    if 0.5 < over < cfg.water_box_clash + 0.4 and not cfg.quiet:
        fit = L - over
        sys.stderr.write(
            f"NOTE: the box is {over:.2f} A longer than a whole number of "
            f"{tile.max():.3f} A water tiles, so the periodic seam holds "
            f"{'voids' if over < cfg.water_box_clash else 'near-duplicate molecules'}. "
            f"--box {fit / NM_TO_ANGSTROM:.4f} tiles seamlessly; "
            f"{(fit + cfg.water_box_clash + 0.6) / NM_TO_ANGSTROM:.4f} or more is clean "
            f"too.\n")
    return L, additive_atoms, water_atoms_of(O, H), summary


def ice_lattice_constant(density):
    return (8.0 * MOLAR_MASS_H2O /
            (AVOGADRO * density * ANGSTROM3_TO_CM3)) ** (1.0 / 3.0)


def ice_cells(box_length, density):
    return max(1, int(round(box_length / ice_lattice_constant(density))))


def ice_water(cfg, L, h_local, np_tree, additive_tree, rng):
    n_cells = ice_cells(L, cfg.water_density)
    a = L / n_cells
    motif = np.array([np.add(f, b) for f in FCC_FRAC for b in DIAMOND_BASIS])
    span = np.arange(n_cells)
    cells = np.array(list(itertools.product(span, span, span)), float)
    O = ((cells[:, None, :] + motif[None, :, :]).reshape(-1, 3) * a) - L / 2.0

    keep = np_tree.query(O + L / 2.0)[0] >= cfg.surface_gap
    if additive_tree is not None:
        keep &= additive_tree.query(O + L / 2.0)[0] >= cfg.additive_gap
    O = O[keep]
    if not len(O):
        die("--water-model ice leaves no lattice site clear of the particle; "
            "enlarge --box.")

    d_oo = a * np.sqrt(3.0) / 4.0
    pairs = periodic_tree(O, L).query_pairs(d_oo * 1.15, output_type="ndarray")

    def bond_dir(src, dst):
        delta = O[dst] - O[src]
        return unit(delta - L * np.round(delta / L))

    degree = np.bincount(pairs.ravel(), minlength=len(O))
    target = np.minimum(degree, 2)
    donor = rng.random(len(pairs)) < 0.5
    d = np.zeros(len(O), int)
    np.add.at(d, pairs[donor, 0], 1)
    np.add.at(d, pairs[~donor, 1], 1)

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

    def clear(points):
        image = points + L / 2.0
        return (np_tree.query(image)[0].min() >= cfg.surface_gap
                and (additive_tree is None
                     or additive_tree.query(image)[0].min() >= cfg.additive_gap))

    h_dirs = h_local / np.linalg.norm(h_local, axis=1)[:, None]
    H = np.zeros((len(O), 2, 3))
    for i, o in enumerate(O):
        dirs = out_dirs[i]
        while len(dirs) < 2:
            v = rng.standard_normal(3)
            for u in dirs:
                v -= (v @ u) * u
            dirs = dirs + [unit(v)]
        aim = np.array(dirs[:2])
        H[i] = o + h_local @ kabsch(h_dirs, aim).T
        if not clear(H[i]):
            trials = np.stack([random_rotation(rng) for _ in range(128)])
            pos = o + np.einsum("tab,hb->tha", trials, h_local)
            ok = np.array([clear(q) for q in pos])
            if ok.any():
                score = np.einsum("tab,hb,ha->t", trials[ok], h_dirs, aim)
                H[i] = pos[ok][int(np.argmax(score))]

    summary = None if cfg.quiet else (
        f"Ice Ic lattice: a {a:.4f} A ({n_cells} cells per edge), "
        f"O-O {d_oo:.3f} A, {len(O)} waters, ice rules satisfied on "
        f"{100 * satisfied:.1f}% of sites, "
        f"H-bonds per water {hbonds_per_water(O, H):.2f}")
    return water_atoms_of(O, H), summary


class ArrayHelp(argparse.ArgumentDefaultsHelpFormatter):
    def _format_args(self, action, default_metavar):
        if action.nargs == "+":
            return self._metavar_formatter(action, default_metavar)(1)[0]
        return super()._format_args(action, default_metavar)


def build_parser():
    p = argparse.ArgumentParser(
        prog="diamond.py",
        description="Stage nanodiamond structures with fully explicit, recorded inputs. "
                    "Every list is an array: [a,b,c].",
        formatter_class=ArrayHelp,
    )

    g = p.add_argument_group("particle")
    g.add_argument("--diameter", type=float, default=1.1,
                   help="nanodiamond diameter in nm; for --shape wulff this is the "
                        "equal-volume sphere diameter")
    g.add_argument("--shape", choices=["sphere", "wulff"], default="sphere",
                   help="sphere = isotropic cut; wulff = {111} octahedron truncated "
                        "by {100} and {110} planes")
    g.add_argument("--gamma-111", type=float, default=1.0,
                   help="surface energy of the {111} facets; only ratios matter, so "
                        "this just sets the unit the other two are quoted in")
    g.add_argument("--gamma-100", type=float, default=None,
                   help="surface energy of the {100} facets. By the Wulff condition "
                        "r_hkl is proportional to gamma_hkl, so gamma100/gamma111 sets "
                        f"the truncation: {RATIO_CUBE:.4f} or below gives a cube, "
                        f"{2 / np.sqrt(3):.4f} a cuboctahedron, {RATIO_OCTAHEDRON:.4f} "
                        "or above leaves a pure octahedron. Omit to disable {100}")
    g.add_argument("--gamma-110", type=float, default=None,
                   help="surface energy of the {110} facets; omit to disable them. "
                        "Detonation-nanodiamond faceting is gamma100:gamma110:gamma111 "
                        "= 0.99:1.03:1")
    g.add_argument("--bond-cutoff", type=float, default=1.9,
                   help="C-C distance below which two lattice sites count as bonded, "
                        "in angstrom")

    g = p.add_argument_group("NV centre")
    g.add_argument("--nv", dest="nv_vacancy", action="store_true", default=True,
                   help="create the NV centre (vacancy + substitutional N)")
    g.add_argument("--no-nv", dest="nv_vacancy", action="store_false",
                   help="leave a pristine core with no NV centre")
    g.add_argument("--nv-align", action="store_true",
                   help="rotate the particle about the cell origin so the V->N bond "
                        "lies on +z and +x points at one of the three remaining "
                        "vacancy-neighbour carbons (the C3v frame)")
    g.add_argument("--nv-r", type=float, default=0.0,
                   help="vacancy offset from the centre as a fraction of the travel "
                        "available along --nv-theta/--nv-phi; 0 is the centre, 1 is "
                        "the shallowest site still --nv-min-depth shells deep")
    g.add_argument("--nv-theta", type=float, default=None,
                   help="azimuth of the vacancy offset in degrees from +x, in the C3v "
                        "frame of a centred NV; drawn from --seed if omitted")
    g.add_argument("--nv-phi", type=float, default=None,
                   help="polar angle of the vacancy offset in degrees from +z, 0 to "
                        "180, in the same frame; drawn from --seed if omitted")
    g.add_argument("--nv-min-depth", type=int, default=2,
                   help="minimum bond-graph distance from the vacancy to the surface "
                        "shell; at the minimum the nitrogen takes the shallowest "
                        "neighbour so the dangling carbons stay buried")

    g = p.add_argument_group("surface chemistry")
    g.add_argument("--substitution", nargs="+", default=None, metavar="ARRAY",
                   help="array of SMILES groups that take the place of a surface "
                        "carbon, e.g. [\"*O*\",\"*N(*)*\"]. The number of '*' is the "
                        "number of carbon neighbours that carbon must have and the "
                        "atom they share takes its lattice site: '*O*' an ether "
                        "oxygen, '*N(*)*' a nitrogen, '*C(=O)*' a ketone. An atom "
                        "written short of its valence, '*[N]*', stays undercoordinated "
                        "and is left for --functionalization to bind")
    g.add_argument("--substitution-ratio", nargs="+", default=None, metavar="ARRAY",
                   help="array with the fraction of all undercoordinated carbons each "
                        "substitution may take, e.g. [0.1,rest]; an upper cap, never "
                        "exceeded. An array summing to more than 1 is rescaled to "
                        "sum to 1, and 'rest' in place of a number takes every site "
                        "the capped entries leave")
    g.add_argument("--substitution-order", nargs="+", default=None, metavar="ARRAY",
                   help="array giving each --substitution entry its turn: [2,1] "
                        "places the second entry first. Earlier turns get first pick of the sites. "
                        "Default is [1,2,...,n], the order written")
    g.add_argument("--substitution-alias", nargs="+", default=None, metavar="ARRAY",
                   help="array with one name per --substitution entry; its atoms are "
                        "labelled <element>_<name>, e.g. [ether] -> O_ether. Default "
                        "is each group's condensed formula. Entries given the same "
                        "alias share labels")
    g.add_argument("--substitution-spacing", type=int, default=3,
                   help="minimum number of bonds between two substituted sites; 3 "
                        "forbids neighbours and sites sharing a neighbour")
    g.add_argument("--functionalization", nargs="+", default=None, metavar="ARRAY",
                   help="array of SMILES groups attached to the dangling bonds left "
                        "after substitution, e.g. [\"*O\",\"*[H]\"]; '*' is the "
                        "undercoordinated atom: '*[H]' a hydrogen, '*O' a hydroxyl, "
                        "'*C(=O)O' a carboxyl. Every atom bonded to '*' is a bond to "
                        "that one surface atom, so '[*](C)CC' or a '*' closing a ring "
                        "needs an atom with two dangling bonds. Default is "
                        "[\"*[H]\"]; [] leaves the particle bare")
    g.add_argument("--functionalization-ratio", nargs="+", default=None, metavar="ARRAY",
                   help="array with the fraction of the remaining dangling bonds each "
                        "group may take, with the same cap, rescaling and 'rest' "
                        "rules, e.g. [0.3,rest]; bonds nothing claims stay bare")
    g.add_argument("--functionalization-attachments", nargs="+", default=None,
                   metavar="ARRAY",
                   help="array with one array of elements per --functionalization "
                        "entry: the undercoordinated atoms that group may bind, e.g. "
                        "[[C,N],[],[C]]. An empty array allows any, and leaving the "
                        "flag out is the same as every array empty")
    g.add_argument("--functionalization-order", nargs="+", default=None, metavar="ARRAY",
                   help="array giving each --functionalization entry its turn: [3,1,2] "
                        "places the second entry first and the first last. Earlier turns get first pick of the sites. "
                        "Default is [1,2,...,n], the order written")
    g.add_argument("--functionalization-alias", nargs="+", default=None, metavar="ARRAY",
                   help="array with one name per --functionalization entry, e.g. "
                        "[OH,H,Grotz] -> O_OH, H_OH, C_Grotz, H_Grotz. A lone hydrogen "
                        "cap stays a plain H unless it is given a name other than H")
    g.add_argument("--contact-scale", type=float, default=0.6,
                   help="no atom of a placed group may come closer to "
                        "another atom than this fraction of the sum of their van der "
                        "Waals radii; only hydrogen-hydrogen contacts are exempt")
    g.add_argument("--surface-seed", type=int, default=None,
                   help="seed for which sites are substituted and functionalized; "
                        "defaults to --seed and leaves the core and NV untouched")

    g = p.add_argument_group("solvent and box")
    g.add_argument("--n-water", type=int, default=None,
                   help="number of water molecules. Omit it to keep every water "
                        "--water-density leaves in --box; given alone, the box is "
                        "sized to hold that many at --water-density; given with "
                        "--box it is a cap, trimmed down to but never compressed up "
                        "to; 0 for a dry structure (then --box is required)")
    g.add_argument("--box", type=float, default=None,
                   help="cubic cell edge in nm; omit to derive it from --n-water. At "
                        "least one of --box / --n-water is required. It may be "
                        "narrower than the particle: a long group then wraps around "
                        "the cell, as long as it stays --surface-gap clear of the "
                        "particle's own periodic image")
    g.add_argument("--water-density", type=float, default=1.0,
                   help="density of the water itself in g/cm^3: the pre-equilibrated "
                        "box is scaled to it before it is replicated and carved, so "
                        "it holds wherever water fills space, not as a cell average")
    g.add_argument("--surface-gap", type=float, default=2.0,
                   help="min distance from any particle atom to any solvent or "
                        "additive atom, and to the particle's own periodic image, in "
                        "angstrom")
    g.add_argument("--water-model", choices=["equilibrated_liquid", "ice"],
                   default="equilibrated_liquid",
                   help="equilibrated_liquid = replicate a pre-equilibrated "
                        "--water-box and carve the cavity out of it; ice = cubic ice "
                        "Ic obeying the Bernal-Fowler rules, which must be melted")
    g.add_argument("--water-box", default="spc216.gro",
                   help="pre-equilibrated periodic water box; .gro, or extxyz "
                        "carrying a Lattice= entry")
    g.add_argument("--water-box-clash", type=float, default=2.2,
                   help="O-O distance below which a tiled water is dropped at the "
                        "periodic seam, in angstrom")
    g.add_argument("--water-xyz", default=None,
                   help="rigid water geometry as a three-atom XYZ file; default is "
                        "the built-in TIP3P geometry")
    g.add_argument("--ice-repair-sweeps", type=int, default=20000,
                   help="augmenting passes used to enforce the ice rules on the protons")
    g.add_argument("--additives", nargs="+", default=None, metavar="ARRAY",
                   help="array of co-solvent/solute XYZ templates placed before the "
                        "water, e.g. [A.xyz,B.xyz]. Each file's basename becomes the "
                        "atom-label tag, e.g. DMSO.xyz -> S_DMSO, C_DMSO")
    g.add_argument("--additive-alias", nargs="+", default=None, metavar="ARRAY",
                   help="array with one name per --additives entry to use as the "
                        "label tag instead of the basename, e.g. [tempol] -> "
                        "O_tempol, N_tempol. Files given the same alias share labels")
    g.add_argument("--n-additives", nargs="+", default=None, metavar="ARRAY",
                   help="array with the molecule count per --additives entry, in the "
                        "same order, e.g. [1,1]. With --additive-ratios it is "
                        "instead the grand total, e.g. [40]")
    g.add_argument("--additive-ratios", nargs="+", default=None, metavar="ARRAY",
                   help="array of relative weights per --additives entry, any "
                        "positive floats, renormalised to the --n-additives total by "
                        "largest remainder, e.g. --additive-ratios [1,1,1,1] "
                        "--n-additives [40]")
    g.add_argument("--additive-seed", type=int, default=None,
                   help="seed for additive identity and placement only; defaults to "
                        "--seed, and is independent of the particle realization")
    g.add_argument("--additive-gap", type=float, default=2.0,
                   help="min distance from any additive atom to another additive or "
                        "to water, in angstrom")
    g.add_argument("--additive-margin", type=float, default=1.0,
                   help="keep additives this far inside the cell wall, in angstrom, "
                        "so no molecule straddles the periodic boundary")
    g.add_argument("--additive-attempts", type=int, default=20000,
                   help="random placement tries per additive molecule before giving up")

    g = p.add_argument_group("realization and output")
    g.add_argument("--seed", type=int, default=None,
                   help="realization seed; drawn and recorded if omitted")
    g.add_argument("--prefix", default=None,
                   help="basename for all outputs; overrides the individual defaults")
    g.add_argument("--outdir", default=".", help="directory for all generated files")
    g.add_argument("--out", default=None,
                   help="final structure (default diamond_all.extxyz)")
    g.add_argument("--core-out", default=None,
                   help="particle-only structure (default diamond_np.extxyz)")
    g.add_argument("--manifest", default=None, help="JSON manifest (default <out>.manifest.json)")
    g.add_argument("--no-wrap", dest="wrap", action="store_false", default=True,
                   help="leave atoms unwrapped instead of folding them into the "
                        "periodic cell")
    g.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    return p


def resolve_paths(cfg):
    defaults = ((f"{cfg.prefix}.extxyz", f"{cfg.prefix}_core.extxyz") if cfg.prefix
                else ("diamond_all.extxyz", "diamond_np.extxyz"))
    cfg.out, cfg.core_out = cfg.out or defaults[0], cfg.core_out or defaults[1]
    if cfg.outdir and cfg.outdir != ".":
        os.makedirs(cfg.outdir, exist_ok=True)
        for attr in ("out", "core_out"):
            if not os.path.isabs(getattr(cfg, attr)):
                setattr(cfg, attr, os.path.join(cfg.outdir, getattr(cfg, attr)))
    cfg.manifest = cfg.manifest or (os.path.splitext(cfg.out)[0] + ".manifest.json")
    return cfg


PHYSICAL_KEYS = [
    "diameter", "lattice_constant", "shape", "gamma_111", "gamma_100", "gamma_110",
    "bond_cutoff", "nv_vacancy", "nv_align", "nv_r", "nv_theta", "nv_phi", "nv_min_depth",
    "substitution_spacing", "contact_scale", "surface_seed",
    "n_water", "box", "water_density", "surface_gap", "water_model", "water_box_clash",
    "ice_repair_sweeps", "additive_gap", "additive_margin", "additive_attempts",
    "additive_seed", "seed",
]


def physical_config(cfg, substitutions, functions, specs):
    inputs = {k: getattr(cfg, k) for k in PHYSICAL_KEYS}
    inputs["water_box"] = os.path.basename(cfg.water_box)
    inputs["water_xyz"] = cfg.water_xyz and os.path.basename(cfg.water_xyz)
    for key, entries in (("substitution", substitutions), ("functionalization", functions)):
        inputs[key] = [dict({"smiles": e["smiles"], "tag": e["tag"],
                             "fraction": "rest" if e["fraction"] is None else e["fraction"]},
                            **({"sites": e["sites"]} if e.get("sites") else {}))
                       for e in entries]
    inputs["additives"] = [{"tag": s["tag"], "template": os.path.basename(s["path"]),
                            "n": s["n"], "n_atoms": s["n_atoms"],
                            "formula": s["formula"]} for s in specs]
    return inputs


def config_hash(inputs):
    blob = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def resolve_cell(cfg, extent, v_excl, v_add):
    if cfg.n_water is None and cfg.box is None:
        die("neither --n-water nor --box was given, so the cell is undetermined.\n"
            "Give --box L to fill that cell with water at --water-density, "
            "--n-water N to derive the box that holds N at that density, both to "
            "cap the count in that cell, or --n-water 0 --box L for a dry structure.")

    if cfg.box is not None:
        return (float(cfg.box) * NM_TO_ANGSTROM, "explicit",
                "derived_from_box" if cfg.n_water is None else "explicit")
    if cfg.n_water == 0:
        die("--n-water 0 was requested, so the box cannot be derived from the solvent "
            "and must be stated explicitly, e.g. --box "
            f"{(extent + cfg.surface_gap) / NM_TO_ANGSTROM:.4f}.")
    return (float((n_water_to_volume(cfg.n_water, cfg.water_density)
                   + v_excl + v_add) ** (1.0 / 3.0)),
            "derived_from_water_count", "explicit")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = build_parser().parse_args(argv)
    cfg.lattice_constant = LATTICE_CONSTANT

    if cfg.n_water is not None and cfg.n_water < 0:
        die("--n-water must be >= 0.")
    for flag, value in (("--gamma-111", cfg.gamma_111), ("--gamma-100", cfg.gamma_100),
                        ("--gamma-110", cfg.gamma_110), ("--contact-scale", cfg.contact_scale)):
        if value is not None and value <= 0:
            die(f"{flag} must be > 0.")
    if cfg.nv_min_depth < 2:
        die("--nv-min-depth must be at least 2, so the nitrogen is never a surface atom.")
    if not 0.0 <= cfg.nv_r <= 1.0:
        die("--nv-r must lie between 0 and 1.")
    if cfg.substitution_spacing < 1:
        die("--substitution-spacing must be at least 1.")

    if cfg.seed is None:
        cfg.seed = int.from_bytes(os.urandom(4), "big")
        if not cfg.quiet:
            print(f"No --seed given; drew seed {cfg.seed} for this realization.")
    if cfg.additive_seed is None:
        cfg.additive_seed = cfg.seed
    if cfg.surface_seed is None:
        cfg.surface_seed = cfg.seed
    cfg = resolve_paths(cfg)

    substitutions = parse_patterns(cfg, "substitution")
    functions = parse_patterns(cfg, "functionalization", default=["*[H]"])
    specs = resolve_additives(cfg)
    entries = substitutions + functions + specs
    shared = sorted({e["tag"] for e in entries if not e["aliased"]
                     and sum(o["tag"] == e["tag"] for o in entries) > 1})
    if shared:
        die(f"the default label tag(s) {', '.join(shared)} fall on more than one "
            f"substitution, functionalization or additive entry. Name them with "
            f"--substitution-alias / --functionalization-alias / --additive-alias; "
            f"entries given the same alias share labels.")
    specs = [s for s in specs if s["n"]]
    v_add = sum(s["n"] * s["volume_angstrom3"] for s in specs)

    r = cfg.diameter * NM_TO_ANGSTROM / 2.0
    ratio_100 = cfg.gamma_100 / cfg.gamma_111 if cfg.gamma_100 else RATIO_OCTAHEDRON
    ratio_110 = cfg.gamma_110 / cfg.gamma_111 if cfg.gamma_110 else RATIO_OCTAHEDRON
    xyz, n_pruned, facets = cut_lattice(r, cfg.shape, ratio_100, ratio_110, cfg.bond_cutoff)
    if len(xyz) < 5:
        die(f"a {cfg.diameter:g} nm cut leaves {len(xyz)} bonded carbons; enlarge "
            f"--diameter.")
    if not cfg.quiet:
        if facets:
            print(f"Wulff core: r111 {facets[0]:.4f} A, r100 {facets[1]:.4f} A, "
                  f"r110 {facets[2]:.4f} A (gamma100:gamma110:gamma111 = "
                  f"{ratio_100:g}:{ratio_110:g}:1)")

    p = Particle(xyz, cfg.bond_cutoff)
    if cfg.nv_vacancy:
        info = place_nv(p, cfg, stream(cfg.seed, "nv"))

    surf = Surface(p)
    env = Environment(p, surf)
    n_sites = substitute(p, surf, env, substitutions, cfg,
                         stream(cfg.surface_seed, "substitution"))
    n_left, n_bare = functionalize(p, surf, env, functions, cfg,
                                   stream(cfg.surface_seed, "functionalization"))
    if not cfg.quiet:
        report_surface(surf, substitutions, functions, n_sites, n_left, n_bare)

    atoms = p.atoms()
    n_lattice = len(p.lattice_atoms())
    labels = [a[0] for a in atoms]
    np_xyz = np.array([a[1:] for a in atoms], float)
    metrics = {"core": size_metrics(labels[:n_lattice], np_xyz[:n_lattice], p.origin()),
               "particle": size_metrics(labels, np_xyz, p.origin())}

    v_excl = excluded_volume(np_xyz, cKDTree(np_xyz), cfg.surface_gap,
                             stream(cfg.seed, "volume"))
    extent = 2.0 * float(np.abs(np_xyz).max())
    box_length, box_source, n_water_source = resolve_cell(cfg, extent, v_excl, v_add)
    wet = cfg.n_water != 0
    if cfg.water_model == "ice" and wet:
        n_cells = ice_cells(box_length, cfg.water_density)
        snapped = n_cells * ice_lattice_constant(cfg.water_density)
        if abs(snapped - box_length) > 1e-9:
            sys.stderr.write(
                f"NOTE: --water-model ice snapped the box from "
                f"{box_length / NM_TO_ANGSTROM:.4f} to {snapped / NM_TO_ANGSTROM:.4f} nm "
                f"so that {n_cells} ice cells tile it exactly at --water-density "
                f"{cfg.water_density:g} g/cm^3.\n")
            box_length = snapped
            box_source += "+ice_snapped"

    def stay_clear(L):
        overlap = image_overlap(np_xyz, L, cfg.surface_gap)
        if overlap:
            d, i, j = overlap
            die(f"a {L / NM_TO_ANGSTROM:.4f} nm box makes the particle overlap its own "
                f"periodic image: {labels[i]} and {labels[j]} come within {d:.2f} A, "
                f"inside the {cfg.surface_gap:g} A --surface-gap. Any box of "
                f"{(extent + cfg.surface_gap) / NM_TO_ANGSTROM:.4f} nm or more is clear.")

    if cfg.box is not None:
        stay_clear(box_length)

    def solutes(L):
        tree = periodic_tree(np_xyz, L)
        added = place_additives(cfg, L, specs, tree) if specs else []
        return tree, added, (periodic_tree(np.array([a[1:] for a in added], float), L)
                             if added else None)

    water_atoms, summary = [], None
    if wet and cfg.water_model == "equilibrated_liquid":
        box_length, additive_atoms, water_atoms, summary = tiled_water(
            cfg, box_length, water_frame(cfg), solutes, cfg.box is None,
            stream(cfg.seed, "water"))
    else:
        np_tree, additive_atoms, additive_tree = solutes(box_length)
        if wet:
            water_atoms, summary = ice_water(cfg, box_length, water_frame(cfg), np_tree,
                                             additive_tree, stream(cfg.seed, "water"))
    if cfg.box is None:
        stay_clear(box_length)
    n_water = len(water_atoms) // 3
    n_asked = cfg.n_water
    if n_asked is None:
        if not n_water:
            die(f"--box {cfg.box:g} nm leaves no room for a single water around the "
                f"particle; enlarge it, or pass --n-water 0 for a dry structure.")
        cfg.n_water = n_water
    elif n_water != n_asked:
        sys.stderr.write(
            f"NOTE: {n_water} waters were placed at --water-density "
            f"{cfg.water_density:g} g/cm^3, not the {n_asked} asked for.\n")

    box_nm = box_length / NM_TO_ANGSTROM
    v_free = box_length ** 3 - v_excl - v_add
    rho_solvent = n_water * MOLAR_MASS_H2O / (AVOGADRO * v_free * ANGSTROM3_TO_CM3)
    cfg.box = box_nm

    inputs = physical_config(cfg, substitutions, functions, specs)
    chash = config_hash(inputs)

    def tally(entries):
        return " ".join(f"{e['tag']}:{e['placed']}" for e in entries) or "none"

    fields = " ".join([
        f"lattice_constant={LATTICE_CONSTANT:.4f}",
        f"diameter_nm={cfg.diameter:.4f}",
        f"rg_angstrom={metrics['particle']['rg_angstrom']:.4f}",
        f"d_equiv_rg_nm={metrics['particle']['d_equiv_rg_nm']:.4f}",
        f"r_max_angstrom={metrics['particle']['r_max_angstrom']:.4f}",
        f"r_inscribed_angstrom={metrics['particle']['r_inscribed_angstrom']:.4f}",
        f"config_hash={chash}",
        f"seed={cfg.seed}",
        f"surface_seed={cfg.surface_seed}",
        f"shape={cfg.shape}",
        f"gamma_100={ratio_100:g}",
        f"gamma_110={ratio_110:g}",
        f'substitution="{tally(substitutions)}"',
        f'functionalization="{tally(functions)}"',
        f"bare={n_bare}",
        f"n_water={cfg.n_water}",
        f"n_water_source={n_water_source}",
        f"box_nm={box_nm:.6f}",
        f"box_source={box_source}",
        f"water_density={cfg.water_density:g}",
        f"water_model={cfg.water_model}",
        f"nv={int(cfg.nv_vacancy)}",
        f"manifest={os.path.basename(cfg.manifest)}",
    ])

    if not cfg.quiet:
        if additive_atoms:
            total = sum(s["n"] for s in specs)
            r = np.linalg.norm(np.array([a[1:] for a in additive_atoms], float), axis=1)
            print("Additives: " + ", ".join(
                f"{s['n']} x {s['tag']}"
                + (f" = {os.path.basename(s['path'])}" if s["aliased"] else "")
                + (f" ({s['n'] / total:.3f} of {total}, asked {s['ratio']:g})"
                   if s["ratio"] is not None else f" ({s['formula']})")
                for s in specs)
                + f"; placed at random, {len(additive_atoms)} atoms "
                  f"{r.min():.2f}-{r.max():.2f} angstrom from the origin")
        if summary:
            print(summary)
        if extent > box_length:
            print(f"Particle spans {extent / NM_TO_ANGSTROM:.4f} nm and wraps around the "
                  f"{box_nm:.4f} nm box, at least {cfg.surface_gap:g} A clear of its own "
                  f"periodic image.")
        print(f"Box edge {box_nm:.4f} nm "
              + ("(given explicitly)" if box_source.startswith("explicit")
                 else "(derived from water count)")
              + f"; {n_water} waters "
              + (f"(what --water-density {cfg.water_density:g} g/cm^3 leaves in the box)"
                 if n_asked is None else "(given explicitly)" if n_water == n_asked
                 else f"(of the {n_asked} asked for)")
              + f"; config {chash}")

    write_text(cfg.core_out, to_extxyz(atoms, box_length, fields)[0])
    final = atoms + additive_atoms + water_atoms
    text, n_wrapped = to_extxyz(final, box_length, fields, wrap=cfg.wrap)
    write_text(cfg.out, text)
    if n_wrapped and not cfg.quiet:
        print(f"Wrapped {n_wrapped} atoms into the periodic cell "
              f"[{-box_nm / 2:.4f}, {box_nm / 2:.4f}] nm on each axis.")

    elements, mass_amu = composition(final)
    rho_system = mass_amu / (AVOGADRO * box_length ** 3 * ANGSTROM3_TO_CM3)

    def record(e):
        keys = ("tag", "input", "smiles", "formula", "stars", "fraction", "requested",
                "placed", "rejected", "eligible") + (
                    ("spaced_out",) if e["kind"] == "substitution" else ("sites", "bonds"))
        return dict({k: e[k] for k in keys},
                    fraction="rest" if e["fraction"] is None else e["fraction"],
                    labels=sorted(set(e["labels"]) | ({e["anchor_label"]}
                                                      if e["kind"] == "substitution" else set())))

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
            "size_metrics": metrics,
            "n_carbons_pruned": int(n_pruned),
            "n_atoms_wrapped": int(n_wrapped),
            "particle_wraps_box": bool(extent > box_length),
            "nv": p.nv or None,
            "box_length_nm": box_nm,
            "box_length_angstrom": box_length,
            "box_source": box_source,
            "n_water_source": n_water_source,
            "free_volume_angstrom3": v_free,
            "excluded_volume_angstrom3": v_excl,
            "additive_volume_angstrom3": v_add,
            "solvent_density_g_cm3": rho_solvent,
            "system_density_g_cm3": rho_system,
            "system_mass_amu": mass_amu,
            "elements": elements,
        },
        "surface": {
            "census": surf.census,
            "substitution": [record(e) for e in substitutions],
            "functionalization": [record(e) for e in functions],
            "dangling_bonds_after_substitution": n_left,
            "dangling_bonds_bare": n_bare,
        },
        "counts": {
            "n_particle_atoms": len(atoms),
            "n_water": n_water,
            "n_additive_molecules": sum(s["n"] for s in specs),
            "n_additive_atoms": len(additive_atoms),
            "n_total_atoms": len(final),
        },
        "additives": [dict({k: v for k, v in s.items() if k not in ("xyz", "aliased")},
                           labels=sorted(set(s["labels"]))) for s in specs],
        "outputs": {"structure": cfg.out, "particle": cfg.core_out},
    }
    write_text(cfg.manifest, json.dumps(manifest, indent=2) + "\n")

    print(f"Total system: {len(final)} atoms ({hill_formula(elements)}) in a "
          f"{box_nm:.4f} nm box")
    print(f"System density: {rho_system:.4f} g/cm^3")
    if not cfg.quiet:
        print(f"{len(atoms)} atoms at D={cfg.diameter:.2f} nm (particle) -> {cfg.core_out}")
        print(f"{len(final)} atoms -> {cfg.out}")
        print(f"provenance -> {cfg.manifest} (config {chash}, seed {cfg.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
