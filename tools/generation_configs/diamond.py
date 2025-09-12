#!/usr/bin/env python3
# Diamond sphere generator (XYZ). Units: Å
from math import ceil, sqrt, acos, sin, cos
import numpy as np
from scipy.spatial import cKDTree
import numpy as np


class DiamondSphereGenerator:
    """
    Generate a diamond (FCC + basis) lattice and keep atoms within radius r.
    Lattice constant a_C = 3.567 Å (0.3567 nm).
    Output: XYZ with "C" atoms and explicit Lattice in header.
    """
    def __init__(self, a_angstrom: float = 3.567):
        self.a = float(a_angstrom)

        # Conventional FCC fractional positions within one cubic cell
        self._fcc_frac = [
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        ]
        # Diamond basis relative to each FCC lattice point
        self._diamond_basis = [
            (0.0, 0.0, 0.0),
            (0.25, 0.25, 0.25),
        ]

    def _cell_span(self, r):
        # Conservative span: cover sphere radius plus one body diagonal of a cell
        margin = self.a * sqrt(3)  # larger than needed; ensures coverage
        n = ceil((r + margin) / self.a)
        return n

    def positions(self, r_angstrom: float, rotate_111_to_z: bool = False):
        """
        Return list of Cartesian positions (Å) of C atoms within radius r.
        Sphere centered at origin (0,0,0).
        If rotate_111_to_z is True, rotate the Bravais vectors so that [111] → [001].
        """
        r = float(r_angstrom)
        r2 = r * r
        n = self._cell_span(r)
        a = self.a

        # Base (unrotated) conventional cubic Bravais vectors
        A0 = np.array([a, 0.0, 0.0], dtype=float)
        A1 = np.array([0.0, a, 0.0], dtype=float)
        A2 = np.array([0.0, 0.0, a], dtype=float)

        if rotate_111_to_z:
            v1 = np.array([1.0, 1.0, 1.0], dtype=float)
            v2 = np.array([0.0, 0.0, 1.0], dtype=float)
            v1 /= np.linalg.norm(v1)
            # v2 already unit
            c = float(np.dot(v1, v2))
            if c < 1.0 - 1e-12:
                if c > -1.0 + 1e-12:
                    axis = np.cross(v1, v2)
                    axis /= np.linalg.norm(axis)
                    x, y, z = axis
                    theta = acos(c)
                    K = np.array([[0, -z, y],
                                  [z, 0, -x],
                                  [-y, x, 0]], dtype=float)
                    R = np.eye(3) + sin(theta) * K + (1.0 - cos(theta)) * (K @ K)
                else:
                    # 180°: pick any axis orthogonal to v1
                    ref = np.array([1.0, 0.0, 0.0]) if abs(v1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                    axis = np.cross(v1, ref)
                    axis /= np.linalg.norm(axis)
                    x, y, z = axis
                    theta = np.pi
                    K = np.array([[0, -z, y],
                                  [z, 0, -x],
                                  [-y, x, 0]], dtype=float)
                    R = np.eye(3) + sin(theta) * K + (1.0 - cos(theta)) * (K @ K)
                A0 = R @ A0
                A1 = R @ A1
                A2 = R @ A2
            # if c ~ 1, R = I, nothing to do

        atoms = []
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                for k in range(-n, n + 1):
                    # origin of the conventional cubic cell (possibly rotated basis)
                    O = i * A0 + j * A1 + k * A2 + np.array([0,0,-self.a*np.sqrt(3/64)])
                    for fx, fy, fz in self._fcc_frac:
                        X0 = O + fx * A0 + fy * A1 + fz * A2
                        for bx, by, bz in self._diamond_basis:
                            X = X0 + bx * A0 + by * A1 + bz * A2
                            if float(np.dot(X, X)) < r2 + 1e-9:
                                atoms.append((float(X[0]), float(X[1]), float(X[2])))

        # Deduplicate
        atoms = list({(round(x, 10), round(y, 10), round(z, 10)) for x, y, z in atoms})
        return sorted(atoms)

    def to_xyz(self, r_angstrom: float, rotate_111_to_z: bool = False, NV_vacancy: bool = True):
        """
        Build XYZ string:
          line 1: atom count
          line 2: Lattice="ax 0 0  0 ay 0  0 0 az"  Crystal="Diamond"  a=3.567
          lines: "<Elem>  x  y  z"
        NV_vacancy: if True, place N at (0,0,-a*sqrt(3/64)) and remove C at (0,0,+a*sqrt(3/64)).
        """
        atoms = self.positions(r_angstrom, rotate_111_to_z=rotate_111_to_z)
    
        if NV_vacancy:
            eps = 1e-4
            z0 = self.a * sqrt(3.0 / 64.0)
            labeled = []
            for x, y, z in atoms:
                if abs(x) < eps and abs(y) < eps:
                    if abs(z - z0) < eps:
                        continue  # vacancy at +z0
                    if abs(z + z0) < eps:
                        labeled.append(("N", x, y, z))  # nitrogen at -z0
                        continue
                labeled.append(("C", x, y, z))
        else:
            labeled = [("C", x, y, z) for (x, y, z) in atoms]
    
        if hasattr(self, "_extra_atoms"):
            labeled.extend(self._extra_atoms)
    
        L = 4.0 * float(r_angstrom)
        header = (
            f'Lattice="{L:.6f} 0.0 0.0  0.0 {L:.6f} 0.0  0.0 0.0 {L:.6f}" Origin="{-L/2:.6f} {-L/2:.6f} {-L/2:.6f}" '
            f'Crystal="Diamond" Bravais="FCC" SpaceGroup="227(Fd-3m)" a={self.a:.4f}Å diameter={L/20:.2f}nm'
        )
        lines = [str(len(labeled)), header]
        lines += [f"{sym} {x:.6f} {y:.6f} {z:.6f}" for (sym, x, y, z) in labeled]
        print(f"{len(labeled)} atoms at D={L/20:.2f} nm")
        return "\n".join(lines)
    

def hydrogen_functionalization(gen, r_angstrom, bond_tol=0.2):
    from scipy.spatial import cKDTree
    import numpy as np

    C_SP3 = 4
    C_H_BOND = 1.09
    TETRA_DIRECTIONS = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]) / np.sqrt(3)

    # Step 1: get base carbon positions
    atoms = gen.positions(r_angstrom, rotate_111_to_z=True)
    coords = np.array(atoms)
    tree = cKDTree(coords)
    covalent_radius = 1.54 / 2
    cutoff = 2 * covalent_radius + bond_tol

    bonded_neighbors = tree.query_ball_tree(tree, cutoff)
    num_bonds = [len(neigh) - 1 for neigh in bonded_neighbors]

    new_atoms = []

    for idx, (pos, nb) in enumerate(zip(coords, num_bonds)):
        missing = C_SP3 - nb
        if missing <= 0:
            continue

        neighbor_indices = bonded_neighbors[idx]
        bonded_vecs = [coords[i] - pos for i in neighbor_indices if i != idx]
        bonded_vecs = [v / np.linalg.norm(v) for v in bonded_vecs if np.linalg.norm(v) > 1e-4]

        # Choose from tetrahedral directions those that are most opposite to existing ones
        remaining_dirs = []
        for dir in TETRA_DIRECTIONS:
            if all(np.dot(dir, bv) < 0.5 for bv in bonded_vecs):
                remaining_dirs.append(dir)
        # Place hydrogen atoms in missing directions
        for i in range(min(missing, len(remaining_dirs))):
            vec = remaining_dirs[i]
            H_pos = pos + C_H_BOND * vec
            new_atoms.append(("H", *H_pos))

    gen._extra_atoms = new_atoms
    return gen


if __name__ == "__main__":
    import sys
    r = float(sys.argv[1])
    gen = DiamondSphereGenerator(a_angstrom=3.567)
    gen = hydrogen_functionalization(gen, r)
    with open("diamond.xyz", "w") as f:
        f.write(gen.to_xyz(r, rotate_111_to_z=True, NV_vacancy=True))

