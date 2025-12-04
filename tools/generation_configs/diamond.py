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
    
        # === Base labeling with replacements ===
        labeled = []
        for i, (x, y, z) in enumerate(atoms):
            sym = "C"
            if hasattr(self, "_surface_idx") and i in self._surface_idx:
                sym = "CS"
            if hasattr(self, "_replacements"):
                sym = self._replacements.get(i, sym)
            labeled.append((sym, x, y, z))
    
        # === Apply NV vacancy substitution ===
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
    
        # === Add any external atoms ===
        if hasattr(self, "_extra_atoms"):
            labeled.extend(self._extra_atoms)
    
        # === Compose XYZ file ===
        L = 2.25*(2.0* r_angstrom)
        header = (
            f'Lattice="{L:.6f} 0.0 0.0  0.0 {L:.6f} 0.0  0.0 0.0 {L:.6f}" '
            f'Origin="{-L/2:.6f} {-L/2:.6f} {-L/2:.6f}" '
            f'a={self.a:.4f}Å diameter={2.0*r_angstrom/10:.2f}nm'
        )
        lines = [str(len(labeled)), header]
        lines += [f"{sym} {x:.6f} {y:.6f} {z:.6f}" for sym, x, y, z in labeled]
    
        print(f"{len(labeled)} atoms at D={2*r_angstrom/10:.2f} nm")
        return "\n".join(lines)
    

def oxygen_mixed_functionalization(gen, r_angstrom, bond_tol=0.2,
                                   ratio_OH_to_O=4.0,
                                   temperature=100.0):
    """
    Mixed hydroxyl/ether oxygen functionalization (reordered + minimal steric model)

    Step 1: Insert bridging oxygens (C→O) using Boltzmann weighting.
    Step 2: Hydrogenate *all* undercoordinated carbons to satisfy valency (no steric checks).
    Step 3: Randomly sample hydrogens; replace some with hydroxyls (OH) if
            the proposed oxygen is not within 0.6 Å of any existing hydrogen.

    This guarantees all carbons reach sp³ valency and avoids unphysical O–H overlaps.
    """

    # === constants ===
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
    gen._surface_idx = set(under_idx)
    n_total = len(coords)

    base_fraction = 25 / 67
    scale_factor = 4.0 / r_angstrom
    target_OH = int(base_fraction * n_total * scale_factor)
    target_OH = max(3, min(target_OH, len(under_idx)))
    target_O = int(target_OH / ratio_OH_to_O)

    base_atoms = [("C", *c) for c in coords]
    existing = list(base_atoms)
    new_atoms = []
    gen._replacements = getattr(gen, "_replacements", {})

    # === Step 1: bridging oxygens ===
    two_coord_idx = [i for i in under_idx if ncoord[i] == 2]
    chosen_O = []
    if two_coord_idx:
        weights_O = np.exp(-(C_SP3 - ncoord[two_coord_idx]) / max(temperature, 1e-3))
        weights_O /= np.sum(weights_O)
        chosen_O = np.random.choice(two_coord_idx,
                                    size=min(target_O, len(two_coord_idx)),
                                    replace=False, p=weights_O)
        for idx in chosen_O:
            gen._replacements[idx] = "OS"

    # === Step 2: hydrogenation (no steric checks) ===
    count_H = 0
    for idx in under_idx:
        if idx in gen._replacements:
            continue

        pos = coords[idx]
        neigh = [n for n in bonded[idx] if n != idx]
        n = len(neigh)
        if n >= 4:
            continue
        if n == 0:
            continue

        neigh_vecs = []
        for j in neigh:
            v = coords[j] - pos
            norm = np.linalg.norm(v)
            if norm > 1e-6:
                neigh_vecs.append(v / norm)
        neigh_vecs = np.array(neigh_vecs)
        rvec = pos / np.linalg.norm(pos)
        missing_dirs = []

        if n == 3:
            normal = np.cross(neigh_vecs[0] - neigh_vecs[1],
                              neigh_vecs[0] - neigh_vecs[2])
            if np.linalg.norm(normal) < 1e-6:
                normal = -np.sum(neigh_vecs, axis=0)
            normal /= np.linalg.norm(normal)
            if np.dot(normal, rvec) < 0:
                normal = -normal
            missing_dirs = [normal]
            count_H += 1
        elif n == 2:
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
            count_H += 2
        elif n == 1:
            a = neigh_vecs[0]
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(a, tmp)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            x = np.cross(a, tmp)
            x /= np.linalg.norm(x)
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
            count_H += 3

        for d in missing_dirs:
            H_pos = pos + BOND_CH * d / np.linalg.norm(d)
            new_atoms.append(("H", *H_pos))
            existing.append(("H", *H_pos))


    # === Step 3: hydroxyl replacement ===
    H_indices = [i for i, (s, *_) in enumerate(new_atoms) if s == "H"]
    random.shuffle(H_indices)
    count_OH = 0

    for hi in H_indices:
        if count_OH >= target_OH:
            break

        sym, *coords_H = new_atoms[hi]
        H_xyz = np.array(coords_H)
        dists = np.linalg.norm(coords - H_xyz, axis=1)
        nearest_C = np.argmin(dists)
        C_xyz = coords[nearest_C]
        CH_vec = H_xyz - C_xyz
        norm_CH = np.linalg.norm(CH_vec)
        if norm_CH < 1e-6:
            continue
        CH_vec /= norm_CH

        O_pos = C_xyz + BOND_CO * CH_vec

        # === reject if O too close to any existing H or O ===
        too_close_flag = False
        for s, x, y, z in existing:
            if s not in ("H", "O"):
                continue
            if np.allclose([x, y, z], H_xyz, atol=1e-3):
                continue
            d = np.linalg.norm(O_pos - np.array([x, y, z]))
            if (s == "H" and d < 1.0) or (s == "O" and d < 1.2):
                too_close_flag = True
                break
        if too_close_flag:
            continue

        # === orient OH hydrogen via angle sweep to maximize spacing ===
        theta = np.deg2rad(104.5)
        tmp = np.random.randn(3)
        x = np.cross(CH_vec, tmp)
        if np.linalg.norm(x) < 1e-6:
            tmp = np.array([1, 0, 0])
            x = np.cross(CH_vec, tmp)
        x /= np.linalg.norm(x)
        y = np.cross(CH_vec, x)

        best_H, best_min_d = None, -1
        center = np.mean(coords, axis=0)
        out_vec = O_pos - center

        for phi_deg in range(0, 360, 10):
            phi = np.deg2rad(phi_deg)
            rot_dir = np.cos(theta) * (-CH_vec) + np.sin(theta) * (np.cos(phi) * x + np.sin(phi) * y)
            rot_dir /= np.linalg.norm(rot_dir)
            candidate = O_pos + BOND_OH * rot_dir

            # outward bias
            if np.dot(rot_dir, out_vec) < -0.5:
                continue

            dists = [np.linalg.norm(candidate - np.array([xx, yy, zz])) for _, xx, yy, zz in existing]
            min_d = min(dists)
            if min_d > best_min_d:
                best_min_d = min_d
                best_H = candidate

        if best_H is None:
            continue

        # === accept replacement ===
        new_atoms[hi] = ("O", *O_pos)
        new_atoms.append(("H", *best_H))
        existing.append(("O", *O_pos))
        existing.append(("H", *best_H))
        count_OH += 1


    # === finalize ===
    gen._extra_atoms = getattr(gen, "_extra_atoms", []) + new_atoms
    print(f"Inserted {len(chosen_O)} bridging O groups, {count_OH} OH groups, and {count_H-count_OH} hydrogens "
          f"out of {len(under_idx)} possible undercoordinated sites.")
    print("Printing Core:")
    print(f"C{n_total-2-len(chosen_O)} H{count_H-count_OH} (OH){count_OH} O{len(chosen_O)} N")
    return gen


def add_water_shell(gen, r_angstrom, N_H2O=10, water_xyz_path=None):
    L = 2.25 * (2.0*r_angstrom)
    molar_mass_H2O = 18.01528
    avogadro = 6.02214076e23
    angstrom3_to_cm3 = 1e-24
    buffer = 2.0
    volume_cm3 = (L**3 - (4/3)*np.pi*(r_angstrom+buffer)**3) * angstrom3_to_cm3
    mass_g = 1 * volume_cm3
    moles = mass_g / molar_mass_H2O
    N_H2O = int(round(moles * avogadro))
    print(N_H2O)

    core_xyz_path = "diamond_np.xyz"
    packmol_input_path = "packmol.inp"
    output_path = "diamond_water.xyz"

    # === finalize the nanoparticle geometry *before* Packmol ===
    core_str = gen.to_xyz(r_angstrom, rotate_111_to_z=True, NV_vacancy=True)
    core_np_atoms = len(core_str.splitlines()) - 2
    with open(core_xyz_path, "w") as f:
        f.write(core_str)

    # === prepare Packmol water file ===
    if water_xyz_path is None:
        water_xyz_path = "water.xyz"
        if not os.path.exists(water_xyz_path):
            with open(water_xyz_path, "w") as f:
                f.write(
                    "3\nWater\nO 0.000 0.000 0.000\n"
                    "H 0.757 a.586 0.000\n"
                    "H -0.757 0.586 0.000\n"
                )

    # === write Packmol input ===
    with open(packmol_input_path, "w") as f:
        f.write(f"""tolerance 2.0
filetype xyz
output {output_path}

pbc {-L/2} {-L/2} {-L/2} {L/2} {L/2} {L/2}
structure {water_xyz_path}
  number {N_H2O}
  outside sphere 0.0 0.0 0.0 {r_angstrom+buffer}
  radius 1.3
end structure

""")

    # === run Packmol silent ===
    with open(packmol_input_path) as inp:
        subprocess.run(
            ["packmol"],
            stdin=inp,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

    # === parse water molecules from Packmol output ===
    with open(output_path) as f:
        lines = f.readlines()
    atom_lines = lines[2:]
    water_lines = atom_lines 

    water_atoms = []
    for line in water_lines:
        sym, x, y, z = line.strip().split()
        if sym == "O":
            sym = "OW"
        elif sym == "H":
            sym = "HW"
        water_atoms.append((sym, float(x), float(y), float(z)))

    print(f"Parsed {len(water_atoms)//3} water molecules ({len(water_atoms)} atoms) from Packmol output in {volume_cm3/angstrom3_to_cm3:.2f} cubic angstrom")

    # === 6. attach waters to gen ===
    gen._extra_atoms = getattr(gen, "_extra_atoms", []) + water_atoms
    return gen

if __name__ == "__main__":
    import sys
    r = float(sys.argv[1])/2
    if len(sys.argv) < 2:
        print("Usage: python diamond.py <diameter_angstrom> <num_water_molecules>")
        sys.exit(1)

    gen = DiamondSphereGenerator()
    gen = oxygen_mixed_functionalization(gen, r)
    gen = add_water_shell(gen, r)
    print("Printing Total System:")
    with open("diamond_water.xyz", "w") as f:
        f.write(gen.to_xyz(r, rotate_111_to_z=True, NV_vacancy=True))
