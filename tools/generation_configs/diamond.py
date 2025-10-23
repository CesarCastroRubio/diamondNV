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
        L = 3.0*(2.0* r_angstrom)
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
                                   temperature=1.0):
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

    # --- steric overlap helper ---
    MIN_DIST_CH = 1.0
    MIN_DIST_CO = 1.3
    MIN_DIST_HH = 1.0
    MIN_DIST_HO = 1.3
    MIN_DIST_OO = 1.4

    def too_close(new_xyz, sym, existing):
        """Return True if new atom (sym, new_xyz) is too close to any existing atom."""
        for s, x, y, z in existing:
            d = np.linalg.norm(new_xyz - np.array([x, y, z]))
            if d < 0.5:                     # absolute guardrail
                return True
            if s == "C" and sym == "H" and d < MIN_DIST_CH:
                return True
            if s == "C" and sym == "O" and d < MIN_DIST_CO:
                return True
            if s == "H" and sym == "H" and d < MIN_DIST_HH:
                return True
            if (s, sym) in [("H", "O"), ("O", "H")] and d < MIN_DIST_HO:
                return True
            if s == "O" and sym == "O" and d < MIN_DIST_OO:
                return True
        return False

    # Preload coordinates of existing framework atoms
    base_atoms = [("C", *c) for c in coords]
    existing = list(base_atoms) + new_atoms  # update dynamically as new atoms are added

    # --- Step 1: Hydroxylate undercoordinated carbons (local geometry) ---
    capped_idx = set()
    OH_sites = set()
    count_OH = 0

    for idx in under_idx:
        if idx in capped_idx:
            continue

        pos = coords[idx]
        neigh = [n for n in bonded[idx] if n != idx]
        n = len(neigh)
        if n >= 4:
            continue
        if n == 0:
            print(f"[Warning] Atom {idx} has zero coordination — skipping hydroxylation.")
            continue

        # --- build normalized neighbor vectors ---
        neigh_vecs = []
        for j in neigh:
            v = coords[j] - pos
            norm = np.linalg.norm(v)
            if norm > 1e-6:
                neigh_vecs.append(v / norm)
        neigh_vecs = np.array(neigh_vecs)

        rvec = pos / np.linalg.norm(pos)  # outward
        need = 4 - n
        missing_dirs = []

        if n == 3:
            # one missing tetrahedral direction opposite plane of 3 bonds
            normal = np.cross(neigh_vecs[0] - neigh_vecs[1],
                              neigh_vecs[0] - neigh_vecs[2])
            if np.linalg.norm(normal) < 1e-6:
                normal = -np.sum(neigh_vecs, axis=0)
            normal /= np.linalg.norm(normal)
            if np.dot(normal, rvec) < 0:
                normal = -normal
            missing_dirs = [normal]

        elif n == 2:
            continue
            # two missing directions roughly tetrahedral to 2 neighbors
            u, v = neigh_vecs
            cross = np.cross(u, v)
            cross /= np.linalg.norm(cross)
            bis = (u + v)
            bis /= np.linalg.norm(bis)
            theta = np.deg2rad(109.5 / 2)
            d1 = -np.cos(theta) * bis + np.sin(theta) * cross
            d2 = -np.cos(theta) * bis - np.sin(theta) * cross
            if np.dot((d1 + d2) / 2, rvec) < 0:
                d1, d2 = -d1, -d2
            missing_dirs = [d1, d2]

        elif n == 1:
            # three directions around single bond
            a = neigh_vecs[0]
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(a, tmp)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            x = np.cross(a, tmp); x /= np.linalg.norm(x)
            y = np.cross(a, x)
            theta = np.deg2rad(109.5)
            dirs = []
            for phi in [0, 120, 240]:
                phi = np.deg2rad(phi)
                dirs.append(-np.cos(theta) * a +
                            np.sin(theta) * (np.cos(phi) * x + np.sin(phi) * y))
            mean_dir = np.mean(dirs, axis=0)
            if np.dot(mean_dir, rvec) < 0:
                dirs = [-d for d in dirs]
            missing_dirs = dirs

        if len(missing_dirs) == 0:
            continue

        # --- choose one missing direction for O ---
        O_dir = missing_dirs[0]
        H_dirs = missing_dirs[1:]
        O_pos = pos + BOND_CO * O_dir
        if too_close(O_pos, "O", existing):
            continue

        # --- place hydrogens on other missing directions ---
        for d in H_dirs:
            H_pos = pos + BOND_CH * d
            if not too_close(H_pos, "H", existing):
                new_atoms.append(("H", *H_pos))
                existing.append(("H", *H_pos))

        # --- attach hydroxyl H with proper 104.5° C–O–H angle ---
        theta = np.deg2rad(104.5)
        tmp = np.random.randn(3)
        x = np.cross(O_dir, tmp)
        if np.linalg.norm(x) < 1e-6:
            tmp = np.array([1, 0, 0])
            x = np.cross(O_dir, tmp)
        x /= np.linalg.norm(x)
        y = np.cross(O_dir, x)
        phi = np.random.uniform(0, 2 * np.pi)
        rot_dir = np.cos(theta) * (-O_dir) + np.sin(theta) * (np.cos(phi) * x + np.sin(phi) * y)
        rot_dir /= np.linalg.norm(rot_dir)
        H_pos = O_pos + BOND_OH * rot_dir

        if too_close(H_pos, "H", existing):
            continue

        new_atoms.append(("O", *O_pos))
        new_atoms.append(("H", *H_pos))
        existing.append(("O", *O_pos))
        existing.append(("H", *H_pos))
        OH_sites.add(idx)
        count_OH += 1
        if count_OH >= target_OH:
            print(f"[Info] Reached target OH count ({target_OH}); stopping hydroxylation.")
            break


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
        if n == 0:
            print(f"[Warning] Atom {idx} has zero coordination — skipping hydrogenation.")
            continue

        # --- build normalized neighbor vectors ---
        neigh_vecs = []
        for j in neigh:
            v = coords[j] - pos
            norm = np.linalg.norm(v)
            if norm > 1e-6:
                neigh_vecs.append(v / norm)
        neigh_vecs = np.array(neigh_vecs)

        # outward direction (from NP center)
        rvec = pos / np.linalg.norm(pos)

        # --- local tetrahedral completion ---
        # For sp3 geometry, the missing bond directions can be approximated
        # by vectors that minimize the dot product with existing bonds
        # under the constraint of tetrahedral angle (~109.5°)
        need = 4 - n
        missing_dirs = []

        if n == 3:
            # single hydrogen opposite to the plane formed by three neighbors
            normal = np.cross(neigh_vecs[0] - neigh_vecs[1],
                              neigh_vecs[0] - neigh_vecs[2])
            if np.linalg.norm(normal) < 1e-6:
                normal = -np.sum(neigh_vecs, axis=0)
            normal /= np.linalg.norm(normal)
            # flip outward if needed
            if np.dot(normal, rvec) < 0:
                normal = -normal
            missing_dirs = [normal]

        elif n == 2:
            # two hydrogens roughly tetrahedral to two neighbors
            # find vector perpendicular to neighbor plane
            u = neigh_vecs[0]
            v = neigh_vecs[1]
            cross = np.cross(u, v)
            cross /= np.linalg.norm(cross)
            # bisector direction of neighbors
            bis = (u + v)
            bis /= np.linalg.norm(bis)
            # tetrahedral angle offset (~109.5°)
            theta = np.deg2rad(109.5 / 2)
            d1 = -np.cos(theta) * bis + np.sin(theta) * cross
            d2 = -np.cos(theta) * bis - np.sin(theta) * cross
            # orient both outward
            if np.dot((d1 + d2) / 2, rvec) < 0:
                d1, d2 = -d1, -d2
            missing_dirs = [d1, d2]

        elif n == 1:
            # three Hs forming CH3 group around single bond
            a = neigh_vecs[0]
            # generate orthonormal basis perpendicular to a
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(a, tmp)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            x = np.cross(a, tmp)
            x /= np.linalg.norm(x)
            y = np.cross(a, x)
            # three directions rotated by 120° around a
            theta = np.deg2rad(109.5)
            dirs = []
            for phi in [0, 120, 240]:
                phi = np.deg2rad(phi)
                dirs.append(-np.cos(theta) * a +
                            np.sin(theta) * (np.cos(phi) * x + np.sin(phi) * y))
            # orient outward
            mean_dir = np.mean(dirs, axis=0)
            if np.dot(mean_dir, rvec) < 0:
                dirs = [-d for d in dirs]
            missing_dirs = dirs

        # --- place hydrogens ---
        for d in missing_dirs:
            d /= np.linalg.norm(d)
            H_pos = pos + BOND_CH * d
            new_atoms.append(("H", *H_pos))

    # --- record replacements permanently ---
    gen._replacements = getattr(gen, "_replacements", {})
    for idx in O_sites:
        gen._replacements[idx] = "O"

    gen._extra_atoms = getattr(gen, "_extra_atoms", []) + new_atoms
    print(f"Added {len(OH_sites)} OH, replaced {len(O_sites)} C→O, "
          f"remaining {len(under_idx) - len(OH_sites) - len(O_sites)} H-capped.")
    return gen


def add_water_shell(gen, r_angstrom, N_H2O=10, water_xyz_path=None):
    L = 3.0 * (2.0*r_angstrom)
    molar_mass_H2O = 18.01528
    avogadro = 6.02214076e23
    angstrom3_to_cm3 = 1e-24
    volume_cm3 = (L**3 - (4/3)*np.pi*r_angstrom**3) * angstrom3_to_cm3
    mass_g = 1.0 * volume_cm3
    moles = mass_g / molar_mass_H2O
    N_H2O = int(round(moles * avogadro))

    buffer = 1.0
    core_xyz_path = "core_np.xyz"
    packmol_input_path = "packmol.inp"
    output_path = "diamond_with_water.xyz"

    # --- 1. finalize the nanoparticle geometry *before* Packmol ---
    core_str = gen.to_xyz(r_angstrom, rotate_111_to_z=True, NV_vacancy=True)
    core_np_atoms = len(core_str.splitlines()) - 2

    # write finalized core for Packmol
    with open(core_xyz_path, "w") as f:
        f.write(core_str)

    # --- 2. prepare Packmol water file ---
    if water_xyz_path is None:
        water_xyz_path = "water.xyz"
        if not os.path.exists(water_xyz_path):
            with open(water_xyz_path, "w") as f:
                f.write(
                    "3\nWater\nO 0.000 0.000 0.000\n"
                    "H 0.757 0.586 0.000\n"
                    "H -0.757 0.586 0.000\n"
                )

    # --- 3. write Packmol input ---
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

    # --- 4. run Packmol ---
    with open(packmol_input_path) as inp:
        subprocess.run(["packmol"], stdin=inp, check=True)

    # --- 5. parse water molecules from Packmol output ---
    with open(output_path) as f:
        lines = f.readlines()
    atom_lines = lines[2:]
    water_lines = atom_lines[core_np_atoms:]

    water_atoms = []
    for line in water_lines:
        sym, x, y, z = line.strip().split()
        water_atoms.append((sym, float(x), float(y), float(z)))

    print(f"Parsed {len(water_atoms)//3} water molecules from Packmol output.")

    # --- 6. attach waters to gen ---
    gen._extra_atoms = getattr(gen, "_extra_atoms", []) + water_atoms
    return gen

if __name__ == "__main__":
    import sys
    r = float(sys.argv[1])
    if len(sys.argv) < 2:
        print("Usage: python diamond2.py <radius_angstrom> <num_water_molecules>")
        sys.exit(1)

    gen = DiamondSphereGenerator(3.567)
    gen = oxygen_mixed_functionalization(gen, r)
    gen = add_water_shell(gen, r)
    with open("diamond_with_water.xyz", "w") as f:
        f.write(gen.to_xyz(r, rotate_111_to_z=True, NV_vacancy=True))
