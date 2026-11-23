"""
Screening step 2 — formal oxidation-state analysis, hard reject + risk tagging.

Rationale
---------
`chemical_system` conditioning fixes the ELEMENTS but not the stoichiometry, so
the emergent formulas imply a range of formal oxidation states for M. That
matters because:

  1. BVSE assumes a definite oxidation state for each species. Our low-cost
     palette was chosen so M has ONE stable state; the emergent formulas break
     that assumption for some structures.
  2. Electronic conduction is fatal in a solid electrolyte. The real risk is a
     partially filled d shell: Ti(3+) is d1 -> metallic / Mott / polaron hopping.
     Main-group ions with odd states (Al2+ = s1) are less "metal" risks than
     phase-stability risks (disproportionation).
  3. Sn(2+) (5s2 lone pair) is a perfectly normal, insulating state -- just not
     the Sn(4+) we targeted.

Two-tier decision
-----------------
HARD REJECT (`impossible`).  A state that violates electron counting cannot
exist regardless of environment, so nothing downstream can rescue it.  The
criterion is a NECESSARY condition only:

        1 <= x <= MAX_VALENCE[M]

  * x > MAX_VALENCE  -- more electrons removed than the atom has valence
    electrons (Ga(5+) would require ionising the 3d shell; Al(4+) the 2p core).
  * x < 1            -- M is not a cation.  x <= 0 means neutral metal or an
    anionic M; 0 < x < 1 means at least one M site must sit at x <= 0.

  For non-integer x the same bounds apply, because a mixed-valence average
  outside [1, MAX_VALENCE] forces at least one site outside it.  An average
  INSIDE the window is kept: Ga avg 2.33 decomposes as Ga(1+)/Ga(3+), which is
  the real, insulating, charge-ordered Ga[GaCl4] motif.

TAG ONLY (everything else).  Rare-but-real states (Al2+, Ga1+, Sn3+) and
d-electron states are flagged, not deleted: the 3D / eta filters remove most
structures anyway, and it is cheaper to judge these on the few survivors.

Formal oxidation state of M is back-computed from charge balance with
Na(+1) and Cl(-1):

        n_Na * (+1) + n_M * x + n_Cl * (-1) = 0
        =>  x = (n_Cl - n_Na) / n_M

Tags
----
  impossible    x outside [1, MAX_VALENCE]  -> REJECTED, written to a side file
  target        x equals the intended state (Al3+, Ga3+, Zr4+, Ti4+, Sn4+)
  known_alt     a different but chemically normal, insulating state (e.g. Sn2+)
  d_electron    partially filled d shell (e.g. Ti3+ d1)  -> METALLIC RISK
  unusual       main-group state that is rare/unstable (e.g. Al2+, Ga1+)
  non_integer   x is not close to an integer -> mixed valence

Relation to the generator's own validity metric
----------------------------------------------
The evaluation shipped with the generator reports a composition-validity
heuristic; 280 of the 971 S.U.N. structures fail it. This script does not use
that verdict. It applies the electron-counting bound above, whose criterion is
stated, whose rejections are written out in full, and which can be audited with
--keep_impossible. A heuristic that travels without its definition is exactly
the kind of quantity this work argues should not be inherited unexamined.

Usage:
    python tag_oxidation.py --pool out/ternary_pool.extxyz --out_dir out

    # also prune the metallic risks up front
    python tag_oxidation.py --drop d_electron

    # keep everything, no rejection (for auditing what would have been cut)
    python tag_oxidation.py --keep_impossible

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

from provenance import report_block, write_header

TOL = 1e-6

# Intended (target) oxidation state per cation.
TARGET_OX = {"Al": 3, "Ga": 3, "Zr": 4, "Ti": 4, "Sn": 4}

# Maximum oxidation state permitted by electron counting = number of valence
# electrons available before breaking into a filled core / filled d shell.
#   Al [Ne]3s2 3p1        -> 3      (Al4+ would ionise the Ne core)
#   Ga [Ar]3d10 4s2 4p1   -> 3      (Ga4+ would ionise the filled 3d shell)
#   Sn [Kr]4d10 5s2 5p2   -> 4
#   Ti [Ar]3d2 4s2        -> 4
#   Zr [Kr]4d2 5s2        -> 4
MAX_VALENCE = {"Al": 3, "Ga": 3, "Sn": 4, "Ti": 4, "Zr": 4}

# All species here are cations in a chloride; x < 1 is not a cation.
MIN_OX = 1

# Chemically normal, insulating alternatives (closed-shell / lone-pair).
KNOWN_ALT = {
    "Sn": {2},     # Sn2+ : 5s2 lone pair, common and insulating
    "Ti": set(),   # Ti2+/Ti3+ are d-electron states -> handled below
    "Zr": set(),
    "Al": set(),
    "Ga": set(),   # Ga1+ exists but is rare in halides -> 'unusual'
}

# Transition metals: a partially filled d shell is the real metallic risk.
# d-count for group-4 metals (Ti, Zr): d_count = 4 - x
TRANSITION_METALS = {"Ti": 4, "Zr": 4}   # element -> group number

def d_count(elem, ox):
    """Number of d electrons for a group-4 transition metal at oxidation state ox."""
    group = TRANSITION_METALS.get(elem)
    if group is None:
        return None
    return group - ox

def check_impossible(elem, x):
    """Necessary condition on the formal state. Returns a reason string, or None.

    Applies to non-integer (mixed-valence) x as well: an average outside the
    window forces at least one site outside it.
    """
    vmax = MAX_VALENCE.get(elem)
    if vmax is None:
        return None
    if x > vmax + TOL:
        if abs(x - round(x)) > TOL:
            return (f"{elem} avg {x:.2f} > max valence {vmax} "
                    f"(some {elem} must exceed {elem}{vmax:+d})")
        return f"{elem}{int(round(x)):+d} exceeds max valence {vmax}"
    if x < MIN_OX - TOL:
        if abs(x - round(x)) > TOL:
            return (f"{elem} avg {x:.2f} < +1 "
                    f"(some {elem} must be neutral or anionic)")
        return f"{elem}{int(round(x)):+d} is not a cation"
    return None

def classify_oxidation(elem, x):
    """Return (tag, note). x is the formal oxidation state of M."""
    # hard reject first: an impossible average must not be excused as
    # 'mixed valence', and an impossible integer must not be excused as 'unusual'
    reason = check_impossible(elem, x)
    if reason is not None:
        return "impossible", f"{reason} -> REJECT"

    # non-integer -> mixed valence
    if abs(x - round(x)) > TOL:
        return "non_integer", f"{elem} avg {x:.2f} (mixed valence)"

    xi = int(round(x))

    if xi == TARGET_OX.get(elem):
        return "target", f"{elem}{xi:+d} (intended)"

    # transition metal with partially filled d shell -> metallic / polaron risk
    dc = d_count(elem, xi)
    if dc is not None and dc > 0:
        return "d_electron", f"{elem}{xi:+d} = d{dc} -> metallic/Mott risk"

    if xi in KNOWN_ALT.get(elem, set()):
        return "known_alt", f"{elem}{xi:+d} (normal, insulating; off-target)"

    return "unusual", f"{elem}{xi:+d} (rare/unstable for {elem})"

RISK_ORDER = ["target", "known_alt", "unusual", "non_integer",
              "d_electron", "impossible"]
RISK_LEVEL = {
    "target":      "safe",
    "known_alt":   "safe (off-target)",
    "unusual":     "phase-stability doubt",
    "non_integer": "mixed valence",
    "d_electron":  "METALLIC RISK",
    "impossible":  "REJECTED (electron count)",
}

FIELDS = ["system", "orig_index", "cation", "n_Na", "n_M", "n_Cl",
          "ox_state", "ox_tag", "risk", "note", "e_above_hull"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="out/ternary_pool.extxyz",
                    help="ternary pool written by filter_ternary.py")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--drop", default="",
                    help="Comma-separated tags to ADDITIONALLY exclude from the "
                         "output pool (e.g. 'd_electron,non_integer'). "
                         "'impossible' is always dropped unless --keep_impossible.")
    ap.add_argument("--keep_impossible", action="store_true",
                    help="Do not reject electron-count-impossible states. "
                         "For auditing only.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    drop = {t.strip() for t in args.drop.split(",") if t.strip()}
    if not args.keep_impossible:
        drop.add("impossible")

    frames = ase_read(args.pool, index=":")
    print(f"Loaded {len(frames)} ternary structures\n")

    kept, rejected = [], []
    rows, rej_rows = [], []
    tag_by_sys = defaultdict(Counter)
    note_counter = Counter()
    rej_note_counter = Counter()

    for atoms in frames:
        sysname = atoms.info.get("system", "?")
        counts = Counter(atoms.get_chemical_symbols())

        cation = next((s for s in counts if s not in ("Na", "Cl")), None)
        if cation is None:
            continue

        n_na = counts.get("Na", 0)
        n_m = counts.get(cation, 0)
        n_cl = counts.get("Cl", 0)
        x = (n_cl - n_na) / n_m   # formal oxidation state of M

        tag, note = classify_oxidation(cation, x)
        tag_by_sys[sysname][tag] += 1
        note_counter[note] += 1

        atoms.info["ox_state"] = round(x, 4)
        atoms.info["ox_tag"] = tag
        atoms.info["cation"] = cation

        row = {
            "system": sysname,
            "orig_index": atoms.info.get("orig_index", ""),
            "cation": cation,
            "n_Na": n_na, "n_M": n_m, "n_Cl": n_cl,
            "ox_state": round(x, 4),
            "ox_tag": tag,
            "risk": RISK_LEVEL[tag],
            "note": note,
            "e_above_hull": atoms.info.get("e_above_hull", ""),
        }
        rows.append(row)

        if tag in drop:
            rejected.append(atoms)
            rej_rows.append(row)
            if tag == "impossible":
                rej_note_counter[note] += 1
        else:
            kept.append(atoms)

    # ---- outputs ----
    out_pool = out_dir / "ternary_tagged.extxyz"
    if kept:
        ase_write(str(out_pool), kept)

    out_rej = out_dir / "ternary_rejected.extxyz"
    if rejected:
        ase_write(str(out_rej), rejected)

    with open(out_dir / "oxidation_index.csv", "w", newline="") as f:
        write_header(f, "tag_oxidation.py", {"input pool": args.pool})
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    if rej_rows:
        with open(out_dir / "rejected_index.csv", "w", newline="") as f:
            write_header(f, "tag_oxidation.py", {"input pool": args.pool})
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rej_rows)

    # ---- report ----
    lines = [report_block("tag_oxidation.py", {"input pool": args.pool}), "",
             "Formal oxidation-state tagging", "=" * 70, ""]
    hdr = f"{'system':<12} {'total':>6} " + " ".join(f"{t:>12}" for t in RISK_ORDER)
    lines.append(hdr)
    lines.append("-" * len(hdr))
    totals = Counter()
    for sysname in sorted(tag_by_sys):
        c = tag_by_sys[sysname]
        n = sum(c.values())
        totals.update(c)
        lines.append(f"{sysname:<12} {n:>6} " +
                     " ".join(f"{c.get(t,0):>12}" for t in RISK_ORDER))
    lines.append("-" * len(hdr))
    lines.append(f"{'TOTAL':<12} {sum(totals.values()):>6} " +
                 " ".join(f"{totals.get(t,0):>12}" for t in RISK_ORDER))
    lines.append("")
    lines.append("Tag meanings:")
    for t in RISK_ORDER:
        lines.append(f"   {t:<12} {RISK_LEVEL[t]}")
    lines.append("")
    lines.append("Max valence used for the hard reject "
                 "(1 <= x <= vmax, applied to averages too):")
    for e in sorted(MAX_VALENCE):
        lines.append(f"   {e:<4} vmax = {MAX_VALENCE[e]}")
    lines.append("")
    lines.append("Distinct oxidation states observed (count):")
    for note, cnt in note_counter.most_common():
        lines.append(f"   {note:<62} {cnt:>5}")
    lines.append("")

    if rej_note_counter:
        n_imp = sum(rej_note_counter.values())
        lines.append(f"HARD REJECT: {n_imp} structures violate electron counting.")
        for note, cnt in rej_note_counter.most_common():
            lines.append(f"   {note:<62} {cnt:>5}")
        lines.append("")

    if totals.get("d_electron"):
        lines.append(f"!! {totals['d_electron']} structures have a partially filled d shell.")
        lines.append("!! These are the genuine metallic-conduction risks; a solid")
        lines.append("!! electrolyte must be an electronic insulator. Confirm with DOS")
        lines.append("!! on any that survive the 3D / eta filters.")
        lines.append("")
    if totals.get("non_integer") and "non_integer" not in drop:
        lines.append(f"!! {totals['non_integer']} structures are mixed-valence but within")
        lines.append("!! the valence window. These are kept because charge-ordered")
        lines.append("!! mixed valence (Ga+ / Ga3+ in Ga[GaCl4]) is insulating, whereas")
        lines.append("!! mixed valence on crystallographically equivalent sites gives")
        lines.append("!! small-polaron hopping. Distinguish with Bader/BVS on survivors.")
        lines.append("")

    lines.append(f"Dropped tags: {sorted(drop) if drop else 'none'}"
                 f"  ->  pool keeps {len(kept)} of {len(frames)}")

    report = "\n".join(lines)
    print(report)
    (out_dir / "oxidation_report.txt").write_text(report + "\n")

    print()
    print(f"Tagged pool   : {out_pool}  ({len(kept)} frames)")
    if rejected:
        print(f"Rejected pool : {out_rej}  ({len(rejected)} frames)")
        print(f"Rejected index: {out_dir/'rejected_index.csv'}")
    print(f"Index         : {out_dir/'oxidation_index.csv'}")
    print(f"Report        : {out_dir/'oxidation_report.txt'}")
    print()
    print("-> Next: dimensionality (3D) filter on ternary_tagged.extxyz")

if __name__ == "__main__":
    main()
