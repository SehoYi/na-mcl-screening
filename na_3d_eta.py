"""
Screening step 3 — Na sublattice percolation dimensionality and isotropy.

Purpose
-------
The screen looks for sodium networks that percolate in three dimensions and are
nearly isotropic. Both properties are set by the topology of the Na-Na hop
network, which is cheap to compute and needs neither bond-valence sums nor
molecular dynamics. Two continuous descriptors are extracted per structure:

  r_3D   largest bottleneck free-radius (Ang) at which a rank-3 (fully 3D
         percolating) Na-Na network still exists.  Blank if the network never
         reaches rank 3.
  eta    lambda_min / lambda_max of the hop tensor of that rank-3 network,
         evaluated at r = r_3D.  1.0 = isotropic, -> 0 = 1D-like.

Continuous descriptors are recorded rather than a pass/fail, so that the
threshold can be placed after the distribution is known.

Method
------
Dimensionality.  Na sites are nodes of a quotient graph; a hop (i, j, image) is
an edge.  BFS assigns each node an accumulated image offset; every non-tree edge
closes a cycle whose image vector is recorded.  The rank of the integer lattice
spanned by those cycle vectors IS the percolation dimensionality of that
connected component (0 isolated cluster, 1 channel, 2 layer, 3 fully 3D).

Rank is computed PER CONNECTED COMPONENT.  Pooling all edges of the cell would
call a cubic structure with three non-intersecting orthogonal 1D channels "3D".

Bottleneck.  Each hop gets a free radius: the minimum over points sampled along
the path of (distance to nearest blocker - blocker ionic radius).  Rather than
fixing an arbitrary cutoff, thresholds are swept from above and the largest one
still admitting a rank-3 component is reported as r_3D.  Adding edges is
monotone in decreasing threshold, so this is a binary search.

Blocker radii.  Six-coordinate Shannon effective ionic radii, keyed by the
formal oxidation state that tag_oxidation.py writes to the `ox_state` field.
  * Sn(2+), which Shannon does not list, takes the machine-learned
    six-coordinate value of 1.025 Ang (Baloch et al., Phys. Rev. Mater. 5,
    043804 (2021)); see ML_EXTENDED below.
  * Six further cation entries are estimates: Al(2+), Ga(2+), Ga(+), Sn(3+),
    Zr(3+), Zr(2+); see ESTIMATED below.
  * Non-integer formal states are interpolated linearly between neighbouring
    integer states; unlisted species take 0.80 Ang.
The three-dimensional count is the one number that depends on the Sn(2+)
value: an earlier estimate of 1.18 Ang gives 457 instead of 460, with the four
survivors unchanged.  The table lives in RADII_OX and provenance.py reads it,
so the description and the numbers cannot drift apart.

Limitations
-----------
  * The path is sampled on the straight line i->j.  The true migration path
    bends around anions, so r_3D underestimates the passage: the screen errs
    toward false negatives, not false positives.
  * eta is built from edge geometry with a pass/fail bottleneck weight.  It
    measures the isotropy of the network shape, not of the diffusivity, and
    can differ from the diffusion tensor obtained from molecular dynamics.
  * The graph is a hopping picture on the stoichiometric Na sublattice.
    Interstitial and concerted mechanisms are not represented.

Usage:
    python na_3d_eta.py --pool out/ternary_tagged.extxyz --out_dir out

    # prune the metallic risks before spending time on topology
    python na_3d_eta.py ... --drop d_electron

    # write the survivor pool at explicit thresholds
    python na_3d_eta.py ... --r_min 0.8 --eta_min 0.5

    # quick check on individual structures, no tagged pool needed
    python na_3d_eta.py --poscar CONTCAR_g0*        # oxidation state inferred
    python na_3d_eta.py --poscar POSCAR --ox_state 2 --rcut 10.0
    python na_3d_eta.py --poscar CONTCAR_* --rcut_sweep 6.5 8 10 12

Author
------
Seho Yi (yiseho207@gmail.com)
Department of Energy and Materials Engineering, Dongguk University, Seoul
"""
__author__ = "Seho Yi"
__email__ = "yiseho207@gmail.com"
__affiliation__ = ("Department of Energy and Materials Engineering, "
                   "Dongguk University, Seoul, Republic of Korea")


import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ase.io import read as ase_read, write as ase_write
from ase.neighborlist import neighbor_list

from provenance import report_block, write_header


MOBILE = "Na"
RCUT = 6.5          # max Na-Na separation admitted as a CANDIDATE hop (Ang).
                    # The manuscript uses 10 Ang, passed by run_all.sh (--rcut);
                    # the retained set is unchanged for any cutoff from 10 to 20.
                    # Deliberately generous: the distance cut must not encode
                    # physics. Na3InCl6 has real hops at 5.22 and 5.36 Ang, and
                    # a 5.0 cut misclassifies it as 2D. Unphysical long pairs
                    # (rocksalt <100>, whose straight line runs through a Cl)
                    # are removed by r_floor instead, not by distance.
R_FLOOR = 0.0       # hops whose free radius is below this are not hops at all:
                    # the Na centre would have to enter the anion hard sphere.
                    # Discarded BEFORE the dimensionality rank is computed.
NSAMP = 21          # points sampled along each hop path (endpoints dropped).
                    # Converged: 21 vs 51 changes r_3D by <0.003 Ang on the
                    # calibration set; NSAMP=5 overestimates by ~0.02.
RANK_TOL = 1e-8
B_DECIMALS = 4      # bin bottleneck radii: symmetry-equivalent hops must not be
                    # split by float noise, or the surviving network at r_3D
                    # loses its point-group symmetry and eta is underestimated

# Blockers are every non-Na site, with six-coordinate Shannon radii (Ang).
# Anions are keyed by element (their state is unambiguous in these chemistries).
ANION_RADII = {"F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20,
               "O": 1.40, "S": 1.84, "Se": 1.98}

# Cations are keyed by (element, formal oxidation state). Values written as a
# one-element list are estimates where Shannon has no CN=6 entry; they shift
# r_3D by a constant offset for that species only.
_RADII_OX = {
    ("Al", 3): 0.535, ("Al", 2): [0.65],
    ("Ga", 3): 0.620, ("Ga", 2): [0.73], ("Ga", 1): [1.20],
    ("Sn", 4): 0.690, ("Sn", 3): [0.90], ("Sn", 2): 1.025,  # not in Shannon: ML_EXTENDED
    ("Ti", 4): 0.605, ("Ti", 3): 0.670, ("Ti", 2): 0.860,
    ("Zr", 4): 0.720, ("Zr", 3): [0.84], ("Zr", 2): [0.95],
    ("Zn", 2): 0.740,  # Table 1 substitution, olivine Na2ZnCl4 reference
    # substitution series into the Na5MCl8 [R-3m] prototype (all M3+)
    ("In", 3): 0.800,
    ("Sc", 3): 0.745,
    ("Y",  3): 0.900,
    ("Er", 3): 0.890,
    # calibration set only
    ("P", 5): 0.380,
}
# Entries written as a one-element list are estimates: Shannon has no CN=6
# value for that (element, state). They are collected here so that provenance
# stamping can report how much of the table is estimated rather than measured.
ESTIMATED = frozenset(k for k, v in _RADII_OX.items() if isinstance(v, list))

# Values Shannon does not list but its machine-learned extension does
# (Baloch et al., Phys. Rev. Mater. 5, 043804 (2021)). Sn2+ matters for the
# screen: an earlier estimate of 1.18 Ang gives 457 three-dimensional networks
# instead of 460, with the four survivors unchanged.
ML_EXTENDED = frozenset({("Sn", 2)})
RADII_OX = {k: (v[0] if isinstance(v, list) else v) for k, v in _RADII_OX.items()}
RADII_DEFAULT = 0.80


def formal_state(atoms):
    """Average framework-cation valence from the composition, on the convention
    used by the composition filter: Na and Cl fixed at +1 and -1, so for
    Na_a M_b Cl_c the framework cation carries (c - a) / b. Returns None when the
    structure is not a single-cation Na-M-Cl ternary."""
    from collections import Counter
    c = Counter(atoms.get_chemical_symbols())
    m = [e for e in c if e not in (MOBILE, "Cl")]
    if len(m) != 1 or MOBILE not in c or "Cl" not in c:
        return None
    return (c["Cl"] - c[MOBILE]) / c[m[0]]


def blocker_radius(elem, ox):
    """Radius for a blocker at a (possibly non-integer) formal state ox."""
    if elem in ANION_RADII:
        return ANION_RADII[elem]
    states = sorted(s for (e, s) in RADII_OX if e == elem)
    if not states:
        return RADII_DEFAULT
    if ox is None:
        # no oxidation info: take the most oxidised (smallest) radius, i.e. the
        # most permissive bottleneck -- errs toward keeping structures
        return min(RADII_OX[(elem, s)] for s in states)
    ox = float(np.clip(ox, states[0], states[-1]))
    lo = max(s for s in states if s <= ox)
    hi = min(s for s in states if s >= ox)
    if lo == hi:
        return RADII_OX[(elem, lo)]
    f = (ox - lo) / (hi - lo)                     # linear in oxidation state
    return (1 - f) * RADII_OX[(elem, lo)] + f * RADII_OX[(elem, hi)]


# ------------------------------------------------------------ geometry helpers
def blocker_supercell(atoms, ox_state):
    """Cartesian coords + radii of all non-mobile atoms across the 3x3x3 block."""
    cell = np.array(atoms.get_cell())
    syms = atoms.get_chemical_symbols()
    frac = atoms.get_scaled_positions()

    keep = [i for i, s in enumerate(syms) if s != MOBILE]
    if not keep:
        return np.zeros((0, 3)), np.zeros(0)
    f = frac[keep]
    r = np.array([blocker_radius(syms[i], ox_state) for i in keep])

    shifts = np.array([[i, j, k] for i in (-1, 0, 1)
                       for j in (-1, 0, 1) for k in (-1, 0, 1)])
    big = (f[None, :, :] + shifts[:, None, :]).reshape(-1, 3) @ cell
    return big, np.tile(r, len(shifts))


def free_radius(p0, p1, blk_xyz, blk_rad):
    """min over interior sampled points of (dist to blocker - r_blocker)."""
    t = np.linspace(0.0, 1.0, NSAMP)[1:-1]
    pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
    d = np.linalg.norm(pts[:, None, :] - blk_xyz[None, :, :], axis=-1)
    return float(np.min(d - blk_rad[None, :]))


def candidate_edges(atoms, ox_state, r_floor=R_FLOOR, mobile_idx=None):
    """[(i, j, image, r_cart, bottleneck)] over Na sites; each undirected hop once.

    mobile_idx restricts the NODES of the hop graph to a subset of the Na atoms
    (e.g. only the tetrahedral, CN<=4 sites). Blockers are unchanged: every
    non-Na atom still blocks, and Na atoms never block, whether or not they are
    in the subset. Restricting the nodes asks "does THIS sublattice percolate
    with open bottlenecks", not "what if the other Na were walls".
    """
    cell = np.array(atoms.get_cell())
    syms = atoms.get_chemical_symbols()
    na = [i for i, s in enumerate(syms) if s == MOBILE]
    if mobile_idx is not None:
        keep = set(int(i) for i in mobile_idx)
        na = [i for i in na if i in keep]
    if len(na) < 2:
        return [], len(na)
    remap = {g: l for l, g in enumerate(na)}

    blk_xyz, blk_rad = blocker_supercell(atoms, ox_state)
    pos = atoms.get_positions()

    I, J, S = neighbor_list("ijS", atoms, RCUT)
    edges, seen = [], set()
    for gi, gj, img in zip(I, J, S):
        gi, gj = int(gi), int(gj)
        if gi not in remap or gj not in remap:
            continue
        i, j, img = remap[gi], remap[gj], tuple(int(v) for v in img)
        key = min((i, j, img), (j, i, tuple(-v for v in img)))
        if key in seen:
            continue
        seen.add(key)

        r = pos[gj] + np.array(img) @ cell - pos[gi]
        b = free_radius(pos[gi], pos[gi] + r, blk_xyz, blk_rad) \
            if len(blk_xyz) else 99.0
        b = round(b, B_DECIMALS)
        if b < r_floor:
            continue
        edges.append((i, j, np.array(img), r, b))
    return edges, len(na)


# --------------------------------------------------- periodic connectivity rank
def components(n, edges):
    """Connected components of the quotient graph, each with its image-lattice rank."""
    adj = defaultdict(list)
    for i, j, img, _, _ in edges:
        adj[i].append((j, img))
        adj[j].append((i, -img))

    off, out = {}, []
    for s in range(n):
        if s in off:
            continue
        off[s] = np.zeros(3, int)
        stack, nodes, cycles = [s], {s}, []
        while stack:
            u = stack.pop()
            for v, img in adj[u]:
                w = off[u] + img
                if v not in off:
                    off[v] = w
                    nodes.add(v)
                    stack.append(v)
                else:
                    c = w - off[v]
                    if np.any(c):
                        cycles.append(c)
        rank = (np.linalg.matrix_rank(np.array(cycles), tol=RANK_TOL)
                if cycles else 0)
        out.append((nodes, rank))
    return out


def hop_tensor(nodes, edges):
    """Normalised sum of r (x) r over edges internal to `nodes`. Trace = 1."""
    T, norm = np.zeros((3, 3)), 0.0
    for i, j, _, r, _ in edges:
        if i in nodes and j in nodes:
            T += np.outer(r, r)
            norm += float(r @ r)
    return None if norm == 0 else T / norm


# ------------------------------------------------------------------- per-frame
def analyse(atoms, r_floor=R_FLOOR, mobile_idx=None):
    """dim, r_1D/r_2D/r_3D and eta for one structure.

    r_kD is the largest bottleneck threshold at which SOME connected component
    still percolates in at least k dimensions. Reporting all three matters: a
    layered conductor (Na2ZrCl6) has no r_3D at all, yet its r_2D is exactly the
    quantity that makes it comparable to the rest of the reference band. Without
    it the most informative anchors are simply blank.

    eta is evaluated on the component achieving `dim`, at r = r_{dim}D, and is
    only meaningful when dim == 3 (a rank-1 or rank-2 network is anisotropic by
    construction, so its lambda_min/lambda_max is near zero by definition).
    """
    ox = atoms.info.get("ox_state")
    edges, n_na = candidate_edges(atoms, ox, r_floor, mobile_idx=mobile_idx)
    blank = dict(n_na=n_na, n_hops=0, dim=0, r_1d=None, r_2d=None,
                 r_3d=None, eta=None)
    if not edges:
        return blank

    bvals = np.unique([e[4] for e in edges])[::-1]      # descending

    def best_at(r, k):
        """(nodes, edges) of a component with rank >= k at threshold r, or None."""
        sub = [e for e in edges if e[4] >= r]
        if not sub:
            return None
        hit = [(nodes, rk) for nodes, rk in components(n_na, sub) if rk >= k]
        if not hit:
            return None
        nodes, _ = max(hit, key=lambda t: t[1])
        return nodes, sub

    def r_perc(k):
        """largest threshold admitting rank >= k; monotone, so binary search."""
        if best_at(bvals[-1], k) is None:
            return None, None
        lo, hi, best = 0, len(bvals) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            res = best_at(bvals[mid], k)
            if res is not None:
                best = (bvals[mid], res)
                hi = mid - 1                            # try a stricter r
            else:
                lo = mid + 1
        return best

    r = {}
    for k in (1, 2, 3):
        r[k] = r_perc(k)

    dim = max((k for k in (1, 2, 3) if r[k][0] is not None), default=0)
    if dim == 0:
        return {**blank, "n_hops": len(edges)}

    rk, (nodes, sub) = r[dim]
    eta = None
    if dim == 3:
        # A rank-1 or rank-2 network has lambda_min ~ 0 by construction, so its
        # "eta" carries no information and would only invite misuse. Report it
        # for fully 3D networks only.
        lam = np.sort(np.linalg.eigvalsh(hop_tensor(nodes, sub)))
        eta = float(lam[0] / lam[2]) if lam[2] > 0 else 0.0

    return dict(n_na=n_na, n_hops=len(sub), dim=dim,
                r_1d=r[1][0], r_2d=r[2][0], r_3d=r[3][0], eta=eta)


FIELDS = ["system", "orig_index", "formula", "cation", "ox_state", "ox_tag",
          "n_Na", "n_hops", "dim", "r_1D", "r_2D", "r_3D", "eta",
          "e_above_hull"]


def run_poscar(paths, ox_arg, cuts, r_floor):
    """Compact path for individual structure files. No tags, no pool, no CSV:
    the point is to be able to check one structure against the same code that
    produced the screening numbers, including how it responds to the neighbour
    cutoff, which is the setting most likely to change a verdict."""
    global RCUT
    rows = {}
    meta = {}
    for f in paths:
        atoms = ase_read(f)
        ox = formal_state(atoms) if ox_arg is None else ox_arg
        meta[f] = (atoms, ox)
        for rc in cuts:
            RCUT = rc
            rows.setdefault(f, {})[rc] = analyse(atoms, r_floor)

    name_w = max(len(Path(f).name) for f in paths)
    print(f"r_floor = {r_floor} Ang;  blocker radii from the formal state\n")
    hdr = (f"{'structure':{name_w}s} {'ox':>5s} {'N_Na':>4s} {'dim':>3s} "
           + " ".join(f"{'r3D@' + format(c, 'g'):>9s}" for c in cuts)
           + f" {'eta':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for f in paths:
        atoms, ox = meta[f]
        last = rows[f][cuts[-1]]
        cells = []
        for c in cuts:
            v = rows[f][c]["r_3d"]
            cells.append("        -" if v is None else f"{v:9.4f}")
        eta = last["eta"]
        oxs = "  -  " if ox is None else f"{ox:+5.2f}"
        print(f"{Path(f).name:{name_w}s} {oxs} {last['n_na']:4d} {last['dim']:3d} "
              + " ".join(cells) + ("        -" if eta is None else f" {eta:7.4f}"))

    # The hop lengths that carry the percolating network decide whether a value
    # is physical. r_3D can be raised without limit by admitting ever longer
    # straight lines, so a descriptor sustained by 12 A "hops" is an artefact of
    # the cutoff and not a transport pathway.
    print(f"\nhop lengths in the percolating network (largest cutoff)")
    RCUT = cuts[-1]
    for f in paths:
        atoms, ox = meta[f]
        edges, n_na = candidate_edges(atoms, ox, r_floor)
        r3 = rows[f][cuts[-1]]["r_3d"]
        if r3 is None or not edges:
            print(f"  {Path(f).name:{name_w}s} no 3D network")
            continue
        sub = [e for e in edges if e[4] >= r3 - 1e-9]
        comp = [nd for nd, rk in components(n_na, sub) if rk >= 3]
        nodes = max(comp, key=len) if comp else set()
        L = sorted(float(np.linalg.norm(e[3])) for e in sub
                   if e[0] in nodes and e[1] in nodes)
        if not L:
            print(f"  {Path(f).name:{name_w}s} no edges in the rank-3 component")
            continue
        print(f"  {Path(f).name:{name_w}s} {len(L):3d} hops   "
              f"min {L[0]:5.2f}  median {L[len(L)//2]:5.2f}  max {L[-1]:5.2f} Ang"
              + ("   [warn] sustained by hops beyond 7 Ang" if L[-1] > 7.0 else ""))

    print(f"\nat the largest cutoff ({cuts[-1]} Ang)")
    for f in paths:
        r = rows[f][cuts[-1]]
        g = lambda x: "  n/a" if x is None else f"{x:5.3f}"
        print(f"  {Path(f).name:{name_w}s} dim={r['dim']}  n_hops={r['n_hops']:4d}  "
              f"r_1D={g(r['r_1d'])}  r_2D={g(r['r_2d'])}  r_3D={g(r['r_3d'])}  "
              f"eta={g(r['eta'])}")

    if len(cuts) > 1:
        unstable = [Path(f).name for f in paths
                    if len({None if rows[f][c]["r_3d"] is None
                            else round(rows[f][c]["r_3d"], 3) for c in cuts}) > 1]
        print()
        if unstable:
            print("[warn] r_3D still moving with the cutoff for: "
                  + ", ".join(unstable))
            print("       the value is not converged; extend the sweep before "
                  "quoting it")
        else:
            print("r_3D converged over the whole sweep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="out/ternary_tagged.extxyz",
                    help="tagged pool written by tag_oxidation.py")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--poscar", nargs="+", default=None,
                    help="POSCAR/CONTCAR/CIF files to check individually; "
                         "bypasses the tagged pool and writes no CSV")
    ap.add_argument("--ox_state", type=float, default=None,
                    help="framework-cation formal state for --poscar mode; "
                         "inferred from the composition when omitted")
    ap.add_argument("--rcut", type=float, default=None,
                    help="override the Na-Na candidate-edge cutoff (Ang)")
    ap.add_argument("--rcut_sweep", nargs="+", type=float, default=None,
                    help="report r_3D at several cutoffs to show convergence")
    ap.add_argument("--drop", default="",
                    help="Comma-separated ox_tag values to exclude before "
                         "analysis (e.g. 'd_electron,non_integer').")
    ap.add_argument("--r_floor", type=float, default=R_FLOOR,
                    help="Discard hops with free radius below this before the "
                         "rank is computed (Ang). 0.0 = the Na centre may not "
                         "enter an anion hard sphere. Set negative only to "
                         "reproduce the unfiltered graph for auditing.")
    ap.add_argument("--r_min", type=float, default=None,
                    help="Bottleneck threshold for the survivor pool (Ang). "
                         "Omit both thresholds to only record descriptors.")
    ap.add_argument("--eta_min", type=float, default=None,
                    help="Isotropy threshold for the survivor pool.")
    args = ap.parse_args()

    global RCUT
    if args.rcut is not None:
        RCUT = args.rcut

    if args.poscar:
        cuts = args.rcut_sweep or [RCUT]
        run_poscar(args.poscar, args.ox_state, sorted(cuts), args.r_floor)
        return
    if not args.pool or not args.out_dir:
        ap.error("--pool and --out_dir are required unless --poscar is used")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drop = {t.strip() for t in args.drop.split(",") if t.strip()}

    frames = ase_read(args.pool, index=":")
    print(f"Loaded {len(frames)} tagged structures")
    if drop:
        frames = [a for a in frames if a.info.get("ox_tag") not in drop]
        print(f"Dropped tags {sorted(drop)} -> {len(frames)} remain")
    print()

    rows, kept = [], []
    dim_by_sys = defaultdict(Counter)
    for k, atoms in enumerate(frames):
        rec = analyse(atoms, args.r_floor)
        sysname = atoms.info.get("system", "?")
        dim_by_sys[sysname][rec["dim"]] += 1

        atoms.info["dim"] = rec["dim"]
        for k in (1, 2, 3):
            if rec[f"r_{k}d"] is not None:
                atoms.info[f"r_{k}D"] = round(rec[f"r_{k}d"], 4)
        if rec["eta"] is not None:
            atoms.info["eta"] = round(rec["eta"], 4)

        rows.append({
            "system": sysname,
            "orig_index": atoms.info.get("orig_index", ""),
            "formula": atoms.get_chemical_formula(empirical=True),
            "cation": atoms.info.get("cation", ""),
            "ox_state": atoms.info.get("ox_state", ""),
            "ox_tag": atoms.info.get("ox_tag", ""),
            "n_Na": rec["n_na"], "n_hops": rec["n_hops"], "dim": rec["dim"],
            "r_1D": "" if rec["r_1d"] is None else round(rec["r_1d"], 4),
            "r_2D": "" if rec["r_2d"] is None else round(rec["r_2d"], 4),
            "r_3D": "" if rec["r_3d"] is None else round(rec["r_3d"], 4),
            "eta": "" if rec["eta"] is None else round(rec["eta"], 4),
            "e_above_hull": atoms.info.get("e_above_hull", ""),
        })

        if rec["dim"] == 3 and \
                (args.r_min is None or rec["r_3d"] >= args.r_min) and \
                (args.eta_min is None or rec["eta"] >= args.eta_min):
            kept.append(atoms)

        if k % 50 == 0:
            rr = "  n/a" if rec["r_3d"] is None else f"{rec['r_3d']:5.2f}"
            r2 = "  n/a" if rec["r_2d"] is None else f"{rec['r_2d']:5.2f}"
            e = "  n/a" if rec["eta"] is None else f"{rec['eta']:5.2f}"
            print(f"  {k:5d}  {rows[-1]['formula']:16s} "
                  f"dim={rec['dim']}  r_2D={r2}  r_3D={rr}  eta={e}", flush=True)

    # ---- outputs ----
    with open(out_dir / "dimensionality_index.csv", "w", newline="") as f:
        write_header(f, "na_3d_eta.py",
                     {"input pool": args.pool,
                      "candidate-edge cutoff": f"{RCUT} Ang",
                      "r_floor": f"{args.r_floor} Ang",
                      "path samples per hop": NSAMP,
                      "bottleneck binning": f"{B_DECIMALS} decimals"})
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    out_pool = None
    if args.r_min is not None or args.eta_min is not None:
        out_pool = out_dir / "ternary_3d.extxyz"
        if kept:
            ase_write(str(out_pool), kept)

    # ---- report ----
    lines = [report_block("na_3d_eta.py",
                          {"input pool": args.pool,
                           "candidate-edge cutoff": f"{RCUT} Ang",
                           "r_floor": f"{args.r_floor} Ang",
                           "path samples per hop": NSAMP,
                           "bottleneck binning": f"{B_DECIMALS} decimals"}), "",
             "Na sublattice percolation dimensionality and isotropy", "=" * 70, "",
             f"RCUT = {RCUT} Ang   r_floor = {args.r_floor} Ang", ""]
    hdr = f"{'system':<12} {'total':>6} " + \
          " ".join(f"{'dim='+str(d):>8}" for d in range(4))
    lines += [hdr, "-" * len(hdr)]
    tot = Counter()
    for s in sorted(dim_by_sys):
        c = dim_by_sys[s]
        tot.update(c)
        lines.append(f"{s:<12} {sum(c.values()):>6} " +
                     " ".join(f"{c.get(d,0):>8}" for d in range(4)))
    lines += ["-" * len(hdr),
              f"{'TOTAL':<12} {sum(tot.values()):>6} " +
              " ".join(f"{tot.get(d,0):>8}" for d in range(4)), ""]
    lines += ["dim = rank of the image lattice spanned by cycles of the",
              "      Na-Na hop graph, computed per connected component.",
              "      0 isolated | 1 channel | 2 layer | 3 percolating 3D", ""]

    a2 = [r["r_2D"] for r in rows if r["dim"] >= 2 and r["r_2D"] != ""]
    if a2:
        v = np.array(a2, float)
        lines.append(f"dim >= 2 structures: {len(a2)}")
        lines.append(f"  r_2D (Ang)  min {v.min():.2f}   median {np.median(v):.2f}"
                     f"   max {v.max():.2f}")
        lines.append("  r_2D lets layered conductors be compared on the same")
        lines.append("  axis as the 3D ones. It is NOT a substitute for r_3D:")
        lines.append("  a high r_2D with no r_3D is a layered material.")
        lines.append("")

    arr3 = [(r["r_3D"], r["eta"]) for r in rows if r["dim"] == 3]
    if arr3:
        r = np.array([a for a, _ in arr3], float)
        e = np.array([b for _, b in arr3], float)
        lines.append(f"3D structures: {len(arr3)}")
        lines.append(f"  r_3D (Ang)  min {r.min():.2f}   median {np.median(r):.2f}   max {r.max():.2f}")
        lines.append(f"  eta         min {e.min():.2f}   median {np.median(e):.2f}   max {e.max():.2f}")
        lines.append("")
        lines.append("  r_3D quantiles: " +
                     "  ".join(f"p{q}={np.percentile(r, q):.2f}" for q in (10, 25, 50, 75, 90)))
        lines.append("  eta  quantiles: " +
                     "  ".join(f"p{q}={np.percentile(e, q):.2f}" for q in (10, 25, 50, 75, 90)))
        lines.append("")
        lines.append("  Joint survivors at candidate thresholds:")
        for rm in (0.4, 0.6, 0.8, 1.0):
            cells = "   ".join(f"eta>={em:.1f} -> {int(((r >= rm) & (e >= em)).sum()):4d}"
                               for em in (0.3, 0.5, 0.7))
            lines.append(f"    r_3D>={rm:.1f} :  {cells}")
        lines.append("")

    lines += ["!! r_3D uses a straight-line path and so UNDERESTIMATES the true",
              "!! bottleneck. Pick the threshold from the distribution above, not",
              "!! from a Shannon-radius argument, and re-check the survivors with",
              "!! first-principles calculations before committing to MD.", ""]

    if out_pool is not None:
        lines.append(f"Survivor pool (r_3D>={args.r_min}, eta>={args.eta_min}): "
                     f"{len(kept)} of {len(frames)}")
    else:
        lines.append("No thresholds given -> descriptors recorded, no pool written.")

    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "dimensionality_report.txt").write_text(report + "\n")

    print()
    print(f"Index  : {out_dir/'dimensionality_index.csv'}")
    print(f"Report : {out_dir/'dimensionality_report.txt'}")
    if out_pool is not None:
        print(f"Pool   : {out_pool}  ({len(kept)} frames)")
    print()
    print("-> Next: first-principles evaluation of the 3D survivors")


if __name__ == "__main__":
    main()
