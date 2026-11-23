"""
Screening step 1 — composition validation of the S.U.N. pool.

chemical_system was a *condition*, not a hard constraint: MatterGen aims at the
element set but can drop an element. Before any dimensionality / eta analysis we
must confirm each structure really is the intended TERNARY Na-M-Cl, and see what
stoichiometries actually emerged.

Checks per structure:
  - element set == {Na, M, Cl}  (exactly 3 species, matching the target system)
  - flags degenerate cases: binary (M missing, e.g. NaCl), missing Na, missing Cl
  - records reduced formula and the Na:M:Cl ratio

Outputs:
  ternary_pool.extxyz     structures that pass (feed this to the 3D filter)
  ternary_index.csv       per-structure record
  composition_report.txt  per-system counts + top stoichiometries

Note on the input pool
----------------------
The S.U.N. pool as collected does not impose the composition-validity
heuristic that the generator's evaluation reports separately: 280 of the 971
structures carry comp_validity=False. That is deliberate. Compositional
admissibility is enforced downstream, in tag_oxidation.py, by an explicit
electron-counting criterion whose bounds are stated and auditable, rather
than by a heuristic whose verdict travels without its definition.

Usage:
    python filter_ternary.py --pool data/sun_structures.extxyz --out_dir out

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

from ase.io import read as ase_read, write as ase_write
from pymatgen.io.ase import AseAtomsAdaptor

from provenance import report_block, write_header

# expected cation for each target system
SYSTEM_CATION = {
    "Na-Al-Cl": "Al",
    "Na-Ga-Cl": "Ga",
    "Na-Zr-Cl": "Zr",
    "Na-Sn-Cl": "Sn",
    "Na-Ti-Cl": "Ti",
}

def classify(elems, target_elems):
    """Return a status string for this structure's element set.

    Missing and extra species are reported independently: a structure that
    drops the target cation and gains another has the same element count as
    the target, and labelling it only 'extra' would hide the substitution.
    """
    if elems == target_elems:
        return "ternary_ok"
    missing = sorted(target_elems - elems)
    extra = sorted(elems - target_elems)
    parts = []
    if missing:
        parts.append("missing_" + "+".join(missing))
    if extra:
        parts.append("extra_" + "+".join(extra))
    return "|".join(parts) if parts else "mismatch"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/sun_structures.extxyz",
                    help="S.U.N. pool written by collect_sun_structures.py")
    ap.add_argument("--out_dir", default="out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = ase_read(args.pool, index=":")
    print(f"Loaded {len(frames)} S.U.N. structures from {args.pool}\n")

    passed = []
    rows = []
    status_by_sys = defaultdict(Counter)
    formula_by_sys = defaultdict(Counter)

    for atoms in frames:
        sysname = atoms.info.get("system", "?")
        cation = SYSTEM_CATION.get(sysname)
        if cation is None:
            # infer: whatever isn't Na or Cl
            others = {s for s in atoms.get_chemical_symbols()} - {"Na", "Cl"}
            cation = sorted(others)[0] if others else "M"
        target = {"Na", cation, "Cl"}

        elems = set(atoms.get_chemical_symbols())
        status = classify(elems, target)
        status_by_sys[sysname][status] += 1

        struct = AseAtomsAdaptor.get_structure(atoms)
        comp = struct.composition
        reduced = comp.reduced_formula

        # Na : M : Cl counts (raw, in the cell)
        counts = Counter(atoms.get_chemical_symbols())
        n_na, n_m, n_cl = counts.get("Na", 0), counts.get(cation, 0), counts.get("Cl", 0)

        rows.append({
            "system": sysname,
            "orig_index": atoms.info.get("orig_index", ""),
            "status": status,
            "formula": reduced,
            "n_Na": n_na, "n_M": n_m, "n_Cl": n_cl,
            "n_atoms": len(atoms),
            "e_above_hull": atoms.info.get("e_above_hull", ""),
        })

        if status == "ternary_ok":
            passed.append(atoms)
            formula_by_sys[sysname][reduced] += 1

    # ---- write outputs ----
    if passed:
        ase_write(str(out_dir / "ternary_pool.extxyz"), passed)

    with open(out_dir / "ternary_index.csv", "w", newline="") as f:
        write_header(f, "filter_ternary.py", {"input pool": args.pool})
        w = csv.DictWriter(f, fieldnames=["system", "orig_index", "status",
                                          "formula", "n_Na", "n_M", "n_Cl",
                                          "n_atoms", "e_above_hull"])
        w.writeheader()
        w.writerows(rows)

    # ---- report ----
    lines = [report_block("filter_ternary.py", {"input pool": args.pool}), ""]
    lines.append("Composition validation of the S.U.N. pool")
    lines.append("=" * 62)
    lines.append("")
    hdr = f"{'system':<12} {'total':>6} {'ternary':>8} {'rate':>7}   other"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    tot = tern = 0
    for sysname in sorted(status_by_sys):
        c = status_by_sys[sysname]
        n = sum(c.values())
        ok = c["ternary_ok"]
        tot += n; tern += ok
        others = ", ".join(f"{k}={v}" for k, v in c.most_common()
                           if k != "ternary_ok") or "-"
        lines.append(f"{sysname:<12} {n:>6} {ok:>8} {ok/n*100 if n else 0:>6.1f}%   {others}")
    lines.append("-" * len(hdr))
    lines.append(f"{'TOTAL':<12} {tot:>6} {tern:>8} {tern/tot*100 if tot else 0:>6.1f}%")
    lines.append("")

    lines.append("Emergent stoichiometries (ternary only, top 6 per system)")
    lines.append("=" * 62)
    for sysname in sorted(formula_by_sys):
        lines.append(f"\n{sysname}:")
        for formula, cnt in formula_by_sys[sysname].most_common(6):
            lines.append(f"   {formula:<16} {cnt:>4}")

    report = "\n".join(lines)
    print(report)
    (out_dir / "composition_report.txt").write_text(report + "\n")

    print()
    print(f"Ternary pool  : {out_dir/'ternary_pool.extxyz'}  ({len(passed)} frames)")
    print(f"Index         : {out_dir/'ternary_index.csv'}")
    print(f"Report        : {out_dir/'composition_report.txt'}")
    print()
    print("-> Next: formal oxidation-state tagging of ternary_pool.extxyz")

if __name__ == "__main__":
    main()
