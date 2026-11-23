"""
Hypothesis v3: bottleneck-gated percolation of the tetrahedral Na sublattice.

History of the hypothesis, each stage refuted by the data:
  v1  "CN4 Na present -> wide channel"        16 low-r_3D structures host CN4.
  v2  "CN4 sublattice percolates by distance" NaAlCl4-types percolate at the
                                              same 5.3-5.7 Ang as the survivors
                                              yet have r_3D ~ 0.05-0.19.
  v3  "CN4 sublattice percolates WITH OPEN BOTTLENECKS"
      The missing ingredient is whether the path between tetrahedral sites is
      blocked by Cl. This script computes r_3D restricted to the CN<=4 Na as
      mobile nodes -- identical machinery to the main screen (same blockers,
      same radii, same threshold sweep), only the graph's nodes change.

Predictions, registered before running:
  survivors (4 with CN4)           r_3D(CN4) ~ global r_3D (0.55-0.75):
                                   the wide network IS the tetrahedral network
  NaAlCl4-types (frac_cn4 = 1.0)   r_3D(CN4) absent or < 0.2: close by
                                   distance, blocked by Cl
  mixed low-band hosts             absent (already fail by distance)

Usage (after run_all.sh; produces the input of Fig. 2b):
    python cn4_percolation.py \
        --index out/dimensionality_index.csv \
        --pool  out/ternary_tagged.extxyz \
        --screen na_3d_eta.py --rcut 10.0
"""

import argparse
import importlib.util

import numpy as np
import pandas as pd
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list

ANIONS = ("Cl", "Br", "I", "S")
CN_SPLIT = 4


def load_screen(path):
    spec = importlib.util.spec_from_file_location("screen", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def na_cn_sites(atoms, rmax=6.0):
    """{atom_index: CN} per Na; first shell at 1.25 x that site's own dmin."""
    syms = atoms.get_chemical_symbols()
    anion = next((x for x in ANIONS if x in syms), None)
    if anion is None:
        return {}
    I, J, D = neighbor_list("ijd", atoms, rmax)
    per = {i: [] for i, x in enumerate(syms) if x == "Na"}
    for k in range(len(I)):
        if syms[I[k]] == "Na" and syms[J[k]] == anion:
            per[I[k]].append(D[k])
    out = {}
    for i, d in per.items():
        if not d:
            out[i] = 0
            continue
        d = np.array(d)
        out[i] = int((d < 1.25 * d.min()).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--screen", default="na_3d_eta.py")
    ap.add_argument("--r_floor", type=float, default=0.0)
    ap.add_argument("--rcut", type=float, default=10.0,
                    help="candidate-hop cutoff (Ang) passed to the screen module; "
                         "must match the cutoff used to build --index")
    ap.add_argument("--out", default="cn4_bottleneck.csv")
    args = ap.parse_args()

    screen = load_screen(args.screen)
    screen.RCUT = args.rcut          # the module default (6.5 Ang) is not the paper's 10 Ang
    print(f"candidate-hop cutoff: {screen.RCUT} Ang")

    df = pd.read_csv(args.index, comment="#")   # skip the provenance header
    d3 = df[(df.dim == 3) & df.r_3D.notna()]
    print(f"{len(d3)} dim=3 structures in the index")

    frames = {}
    for a in ase_read(args.pool, index=":"):
        frames[(str(a.info.get("system")), str(a.info.get("orig_index")))] = a

    rows, missing = [], 0
    for _, r in d3.iterrows():
        a = frames.get((str(r.system), str(r.orig_index)))
        if a is None:
            missing += 1
            continue
        cn = na_cn_sites(a)
        cn4_idx = [i for i, c in cn.items() if c <= CN_SPLIT]
        rec = (screen.analyse(a, args.r_floor, mobile_idx=cn4_idx)
               if len(cn4_idx) >= 2 else None)

        rows.append(dict(
            system=r.system, orig_index=r.orig_index, formula=r.formula,
            ox_tag=r.ox_tag,
            r_3D=r.r_3D, eta=r.eta,
            n_cn4=len(cn4_idx),
            frac_cn4=round(len(cn4_idx) / len(cn), 2) if cn else np.nan,
            dim_cn4="" if rec is None else rec["dim"],
            r_3D_cn4="" if rec is None or rec["r_3d"] is None
                     else round(rec["r_3d"], 4),
            eta_cn4="" if rec is None or rec["eta"] is None
                    else round(rec["eta"], 4),
        ))
    if missing:
        print(f"note: {missing} rows had no matching frame")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    has4 = out[out.n_cn4 > 0].copy()
    has4["r3c"] = pd.to_numeric(has4.r_3D_cn4, errors="coerce")

    # ---------- headline: bands x r_3D(CN4) ----------
    bands = [("r_3D < 0.24", 0.00, 0.24), ("0.24-0.30", 0.24, 0.30),
             ("0.30-0.55 gap", 0.30, 0.55), ("survivors", 0.55, 1.00)]
    hdr = (f"{'band':<16}{'n(CN4)':>7}{'3D open':>9}"
           f"{'r_3D(CN4) values':>44}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, lo, hi in bands:
        b = has4[(has4.r_3D >= lo) & (has4.r_3D < hi)]
        if not len(b):
            print(f"{name:<16}{0:>7}")
            continue
        vals = sorted(b.r3c.dropna().round(2))
        print(f"{name:<16}{len(b):>7}{len(vals):>9}   {vals}")

    # ---------- the decisive comparison ----------
    print("\nsurvivors: does the CN4 network carry the full-network bottleneck?")
    sv = out[(out.r_3D >= 0.55) & (out.n_cn4 > 0)].sort_values("r_3D")
    print(sv[["formula", "r_3D", "r_3D_cn4", "eta", "eta_cn4",
              "n_cn4", "frac_cn4"]].to_string(index=False))

    print("\nNaAlCl4-type (frac_cn4 = 1.0, r_3D < 0.3): close but blocked?")
    iso = out[(out.frac_cn4 == 1.0) & (out.r_3D < 0.3)].sort_values("r_3D")
    if len(iso):
        print(iso[["formula", "r_3D", "r_3D_cn4", "dim_cn4",
                   "n_cn4"]].to_string(index=False))
    else:
        print("  none")

    # ---------- verdict ----------
    print("\n" + "=" * 64)
    sv_vals = pd.to_numeric(sv.r_3D_cn4, errors="coerce").dropna()
    lo4 = has4[has4.r_3D < 0.3]
    lo_vals = lo4.r3c.dropna()
    print(f"survivors:  {len(sv_vals)}/{len(sv)} keep an open 3D CN4 network; "
          f"values {sorted(sv_vals.round(2))}")
    print(f"low band:   {len(lo_vals)}/{len(lo4)} have any open 3D CN4 network"
          + (f"; values {sorted(lo_vals.round(2))}" if len(lo_vals) else ""))
    if len(sv_vals) == len(sv) and \
       (not len(lo_vals) or lo_vals.max() < (sv_vals.min() - 0.2)):
        print("v3 HOLDS: the survivors' wide channel IS the tetrahedral")
        print("sublattice, and the NaAlCl4-types are distance-connected but")
        print("bottleneck-blocked. Compare r_3D_cn4 to r_3D per survivor: if")
        print("they match, the CN4 network is the whole story; if r_3D_cn4 is")
        print("lower, the widest paths also pass through octahedral Na.")
    else:
        print("v3 does NOT separate the bands cleanly. Inspect the exceptions;")
        print("the discriminator is not the CN4 subgraph even with bottlenecks.")

    print(f"\nper-structure table: {args.out}")


if __name__ == "__main__":
    main()
