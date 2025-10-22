import numpy as np
import os, subprocess
from scipy.spatial import cKDTree
import random

class DiamondSphereGenerator:
    def __init__(self, a_angstrom=3.567):
        self.a = float(a_angstrom)
        self._fcc_frac = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
        self._diamond_basis = [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]

    def _cell_span(self, r):
        return int(np.ceil((r + self.a * np.sqrt(3)) / self.a))

    def positions(self, r_angstrom, rotate_111_to_z=False):
        r = float(r_angstrom)
        r2 = r * r
        n = self._cell_span(r)
        a = self.a
        A0, A1, A2 = np.array([a, 0, 0]), np.array([0, a, 0]), np.array([0, 0, a])
        if rotate_111_to_z:
            v1, v2 = np.array([1, 1, 1], float), np.array([0, 0, 1], float)
            v1 /= np.linalg.norm(v1)
            c = np.dot(v1, v2)
            if c < 1 - 1e-12:
                if c > -1 + 1e-12:
                    axis = np.cross(v1, v2); axis /= np.linalg.norm(axis)
                    x, y, z = axis; theta = np.arccos(c)
                    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
                else:
                    ref = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
                    axis = np.cross(v1, ref); axis /= np.linalg.norm(axis)
                    x, y, z = axis; theta = np.pi
                    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
                A0, A1, A2 = R @ A0, R @ A1, R @ A2
        atoms = []
        shift = np.array([0, 0, -self.a * np.sqrt(3 / 64)])
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                for k in range(-n, n + 1):
                    O = i * A0 + j * A1 + k * A2 + shift
                    for fx, fy, fz in self._fcc_frac:
                        X0 = O + fx * A0 + fy * A1 + fz * A2
                        for bx, by, bz in self._diamond_basis:
                            X = X0 + bx * A0 + by * A1 + bz * A2
                            if np.dot(X, X) < r2 + 1e-9:
                                atoms.append(tuple(X))
        atoms = list({(round(x, 10), round(y, 10), round(z, 10)) for x, y, z in atoms})
        return sorted(atoms)


    def to_xyz(self, r_angstrom, rotate_111_to_z=False, NV_vacancy=True):
        atoms = self.positions(r_angstrom, rotate_111_to_z)
    
        # --- Base labeling with replacements (C→O, etc.) ---
        labeled = []
        for i, (x, y, z) in enumerate(atoms):
            sym = "C"
            if hasattr(self, "_replacements"):
                sym = self._replacements.get(i, sym)
            labeled.append((sym, x, y, z))
    
        # --- Apply NV vacancy substitution ---
        if NV_vacancy:
            eps = 1e-4
            z0 = self.a * np.sqrt(3 / 64)
            updated = []
            for sym, x, y, z in labeled:
                if abs(x) < eps and abs(y) < eps:
                    if abs(z - z0) < eps:
                        continue  # remove C at +z0 (vacancy)
                    if abs(z + z0) < eps:
                        updated.append(("N", x, y, z))  # replace with N
                        continue
                updated.append((sym, x, y, z))
            labeled = updated
    
        # --- Add any external atoms (e.g., OH or H) ---
        if hasattr(self, "_extra_atoms"):
            labeled.extend(self._extra_atoms)
    
        # --- Compose XYZ file ---
        L = 4.0 * r_angstrom
        header = (
            f'Lattice="{L:.6f} 0.0 0.0  0.0 {L:.6f} 0.0  0.0 0.0 {L:.6f}" '
            f'Origin="{-L/2:.6f} {-L/2:.6f} {-L/2:.6f}" '
            f'Crystal="Diamond" Bravais="FCC" SpaceGroup="227(Fd-3m)" '
            f'a={self.a:.4f}Å diameter={L/20:.2f}nm'
        )
        lines = [str(len(labeled)), header]
        lines += [f"{sym} {x:.6f} {y:.6f} {z:.6f}" for sym, x, y, z in labeled]
    
        print(f"{len(labeled)} atoms at D={L/20:.2f} nm")
        return "\n".join(lines)
    

def hydrogen_functionalization(gen, r_angstrom, bond_tol=0.2):
    C_SP3, C_H_BOND = 4, 1.09
    TETRA = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)

    coords = np.array(gen.positions(r_angstrom, rotate_111_to_z=True))
    tree = cKDTree(coords)
    cutoff = 1.7 + bond_tol
    new_atoms = []

    for i, pos in enumerate(coords):
        neigh = tree.query_ball_point(pos, cutoff)
        neigh = [n for n in neigh if n != i]
        ncoord = len(neigh)
        if ncoord >= 4:
            continue

        missing = 4 - ncoord
        if missing <= 0:
            continue

        rvec = pos / np.linalg.norm(pos)
        vecs = coords[neigh] - pos
        vecs /= np.linalg.norm(vecs, axis=1)[:, None] if len(vecs) else 1

        # choose tetrahedral directions opposite to bonded ones and aligned outward
        remaining_dirs = []
        for d in TETRA:
            if all(np.dot(d, v) < 0.5 for v in vecs):  # opposite to existing bonds
                if np.dot(d, rvec) > 0:  # outward facing
                    remaining_dirs.append(d)

        if not remaining_dirs:
            # fallback: use outward tetra set
            remaining_dirs = [d for d in TETRA if np.dot(d, rvec) > 0]

        for d in remaining_dirs[:missing]:
            H_pos = pos + C_H_BOND * d
            new_atoms.append(("H", *H_pos))

    gen._extra_atoms = getattr(gen, "_extra_atoms", []) + new_atoms
    return gen



def oxygen_mixed_functionalization(gen, r_angstrom, bond_tol=0.2,
                                   ratio_OH_to_O=4.0,
                                   temperature=0.6):
    """
    Mixed hydroxyl/ether oxygen functionalization (Galli et al. 2024 model)

    - Hydroxylates undercoordinated surface carbons with Boltzmann randomness.
    - Converts a subset (~1 per 4 OH) of *2-coordinated* carbons to oxygen atoms.
    - Caps remaining sites with hydrogens.

    Stores substitutions in gen._replacements = {index: "O"}
    so that to_xyz() prints them correctly.
    """
    C_SP3 = 4
    BOND_CC = 1.7
    BOND_CH = 1.09
    BOND_CO = 1.43
    BOND_OH = 0.96

    coords = np.array(gen.positions(r_angstrom, rotate_111_to_z=True))
    tree = cKDTree(coords)
    bonded = tree.query_ball_tree(tree, BOND_CC + bond_tol)
    ncoord = np.array([len(neigh) - 1 for neigh in bonded])
    under_idx = np.where(ncoord < C_SP3)[0]
    n_total = len(coords)

    # empirical scaling (25 OH / 67 C at 1 nm)
    base_fraction = 25 / 67
    scale_factor = 4.0 / r_angstrom
    target_OH = int(base_fraction * n_total * scale_factor)
    target_OH = max(3, min(target_OH, len(under_idx)))

    new_atoms = []
    OH_sites, O_sites = set(), set()

    # Boltzmann probabilities
    weights = np.exp(-(C_SP3 - ncoord[under_idx]) / max(temperature, 1e-3))
    weights /= np.sum(weights)

    # --- Step 1: Hydroxylate undercoordinated carbons ---
    chosen_OH = np.random.choice(under_idx,
                                 size=min(target_OH, len(under_idx)),
                                 replace=False,
                                 p=weights)
    for idx in chosen_OH:
        pos = coords[idx]
        rvec = pos / np.linalg.norm(pos)
        O_pos = pos + BOND_CO * rvec
        H_pos = O_pos + BOND_OH * rvec
        new_atoms.append(("O", *O_pos))
        new_atoms.append(("H", *H_pos))
        OH_sites.add(idx)

    # --- Step 2: choose subset of 2-coordinated Cs to convert to O ---
    two_coord_idx = [i for i in under_idx if ncoord[i] == 2 and i not in OH_sites]
    target_O = int(len(OH_sites) / ratio_OH_to_O)
    if two_coord_idx:
        weights_O = np.exp(-(C_SP3 - ncoord[two_coord_idx]) /
                           (2 * max(temperature, 1e-3)))
        weights_O /= np.sum(weights_O)
        chosen_O = np.random.choice(two_coord_idx,
                                    size=min(target_O, len(two_coord_idx)),
                                    replace=False,
                                    p=weights_O)
        for idx in chosen_O:
            O_sites.add(idx)

    # --- Step 3: hydrogen cap remaining ---
    capped_idx = OH_sites | O_sites
    for idx in under_idx:
        if idx in capped_idx:
            continue
        pos = coords[idx]
        neigh = [n for n in bonded[idx] if n != idx]
        n = len(neigh)
        if n >= 4:
            continue
        missing = 4 - n
        rvec = pos / np.linalg.norm(pos)
        for _ in range(missing):
            H_pos = pos + BOND_CH * rvec
            new_atoms.append(("H", *H_pos))

    # --- record replacements permanently ---
    gen._replacements = getattr(gen, "_replacements", {})
    for idx in O_sites:
        gen._replacements[idx] = "O"

    gen._extra_atoms = getattr(gen, "_extra_atoms", []) + new_atoms
    print(f"Added {len(OH_sites)} OH, replaced {len(O_sites)} C→O, "
          f"remaining {len(under_idx) - len(OH_sites) - len(O_sites)} H-capped.")
    return gen

def add_water_shell(gen, r_angstrom, N_H2O=150, water_xyz_path=None):
    L = 4.0 * r_angstrom
    buffer = 1.0
    core_xyz_path = "core_np.xyz"
    packmol_input_path = "packmol.inp"
    output_path = "diamond_with_water.xyz"

    with open(core_xyz_path, "w") as f:
        f.write(gen.to_xyz(r_angstrom))

    if water_xyz_path is None:
        water_xyz_path = "water.xyz"
        if not os.path.exists(water_xyz_path):
            with open(water_xyz_path, "w") as f:
                f.write("3\nWater\nO 0.000 0.000 0.000\nH 0.757 0.586 0.000\nH -0.757 0.586 0.000\n")

    with open(packmol_input_path, "w") as f:
        f.write(f"""tolerance 2.0
filetype xyz
output {output_path}

structure {core_xyz_path}
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure

structure {water_xyz_path}
  number {N_H2O}
  inside box {-L/2 + buffer} {-L/2 + buffer} {-L/2 + buffer} {L/2 - buffer} {L/2 - buffer} {L/2 - buffer}
  outside sphere 0.0 0.0 0.0 {r_angstrom + buffer}
end structure
""")

    with open(packmol_input_path) as inp:
        subprocess.run(["packmol"], stdin=inp, check=True)

    with open(output_path) as f:
        lines = f.readlines()

    core_np_atoms = len(gen.to_xyz(r_angstrom).splitlines()) - 2
    water_lines = lines[2:][core_np_atoms:]
    water_atoms = [(sym, float(x), float(y), float(z)) for sym, x, y, z in (line.strip().split() for line in water_lines)]
    print(f"Parsed {len(water_atoms)//3} water molecules from Packmol output.")

    gen._extra_atoms = getattr(gen, "_extra_atoms", []) + water_atoms
    return gen

if __name__ == "__main__":
    import sys
    r = float(sys.argv[1]); N = int(sys.argv[2])
    if len(sys.argv) < 3:
        print("Usage: python diamond2.py <radius_angstrom> <num_water_molecules>")
        sys.exit(1)

    gen = DiamondSphereGenerator(3.567)
    gen = oxygen_mixed_functionalization(gen, r)
    gen = add_water_shell(gen, r, N_H2O=N)
    with open("diamond_with_water.xyz", "w") as f:
        f.write(gen.to_xyz(r, rotate_111_to_z=True, NV_vacancy=True))
