"""
Collect S.U.N.-passing structures from all systems into ONE place.

Why this exists:
  `mattergen-evaluate` gives you
    - metrics_TRI2024.json          : summary percentages only
    - detailed_metrics_TRI2024.json : PER-STRUCTURE stability/novelty/uniqueness
    - relaxed_TRI2024/              : ALL relaxed structures (pass AND fail)
  but it never writes "just the S.U.N. winners". This script does that:
  it reads the per-structure verdicts and splits them into TWO pools.

    sun     Stable AND Unique AND Novel        -> the candidate pool
    known   Stable AND Unique AND NOT Novel    -> compounds the model
                                                  re-discovered, i.e. already
                                                  present in the reference DB

  The `known` pool matters because it is the only set of structures that has
  passed through EXACTLY the same generation and MLIP relaxation as the
  candidates. Reference geometries pulled from Materials Project are PBE
  relaxed, and r_3D depends directly on the lattice, so an anchor taken from MP
  and a candidate taken from MACE do not sit on the same geometric footing.
  The recovered known compounds do.

  Caveats to keep in mind when using `known` as a reference band:
    * "not novel" means "matched something in the reference DB". Much of that
      DB is itself computational. Only entries flagged non-theoretical (ICSD
      provenance) have ever been synthesised, and only a couple of those have a
      measured ionic conductivity. Do not call this a validation set.
    * The compounds a generative model recovers are biased toward common, dense,
      low-energy prototypes, so their r_3D distribution is biased low. Use the
      band to check for composition artefacts, not to set an absolute cutoff.

Splitting S/U/N requires the INDIVIDUAL verdict keys. If the detailed metrics
only ship the combined `novel_unique_stable` flag, the known pool cannot be
built and the script says so rather than guessing.

Inputs (per system, under --gen_base/<system>/):
    detailed_metrics_TRI2024.json
    relaxed_TRI2024/            (relaxed structures; extxyz and/or cif)
    generated_crystals.extxyz   (fallback if relaxed dir is unusable)

Outputs (under --out_dir):
    sun_structures.extxyz       pooled, one frame per S.U.N. structure
    sun_structures/*.cif        one CIF per S.U.N. structure (tagged by system)
    sun_index.csv               system, orig_index, e_hull, formula, cif name

    known_recovered.extxyz      pooled, one frame per S & U & NOT-novel structure
    known_structures/*.cif
    known_index.csv             ... plus matched_id where the metrics expose it

Usage:
    python collect_sun_structures.py \
        --gen_base /Users/seho/work/Na_conductor/lowcost \
        --out_dir  /Users/seho/work/Na_conductor/lowcost/SUN_POOL

    # inspect the detailed-JSON schema first (recommended once):
    python collect_sun_structures.py --gen_base /Users/seho/work/Na_conductor/lowcost --inspect

    # keep recovered known compounds even if they sit above the hull cutoff
    python collect_sun_structures.py --gen_base ... --known_ignore_stability

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
import json
from pathlib import Path

from ase.io import read as ase_read, write as ase_write


DEFAULT_SYSTEMS = ["Na-Al-Cl", "Na-Ga-Cl", "Na-Zr-Cl", "Na-Sn-Cl", "Na-Ti-Cl"]

# Candidate key names (confirmed against mattergen detailed_metrics_TRI2024.json).
# The file conveniently ships a combined verdict key, so prefer it and fall
# back to ANDing the individual ones.
K_SUN    = ["novel_unique_stable"]              # combined S AND U AND N
K_STABLE = ["stable", "stability", "is_stable"]
# Prefer the RAW per-structure booleans (`is_*`) over the possibly
# validity-masked `novel` / `unique`. See the disagreement check in main().
K_UNIQUE = ["is_unique", "unique", "uniqueness"]
K_NOVEL  = ["is_novel", "novel", "novelty"]
# e_hull: prefer the per-atom value (the 0.1 eV/atom criterion is per atom).
K_EHULL  = ["energy_above_hull_per_atom", "energy_above_hull", "e_above_hull"]
# Validity. mattergen ships BOTH a composition check (SMACT charge balance +
# electronegativity ordering) and a structure check (no atoms closer than
# ~0.5 Ang). Neither is part of the S.U.N. verdict, so a structure can be
# "S.U.N." and still be charge-unbalanced. Carry them through and let the
# downstream screen decide.
K_COMPVAL = ["comp_validity"]
K_STRVAL  = ["structure_validity"]
# Diagnostics worth carrying: how far the generated structure moved during MLIP
# relaxation (large -> the generated geometry was poor), and whether the
# COMPOSITION (not the structure) already appears in the reference set.
K_RMSD    = ["rmsd_from_relaxation"]
K_EXPL    = ["is_explored"]
# Identifier of the reference entry a non-novel structure matched against.
# Names vary by mattergen version; record whichever is present so the known
# pool can later be cross-referenced against ICSD provenance and measured
# conductivities. Without this the known pool is an anonymous pile of frames.
K_MATCH  = ["matched_material_id", "matched_id", "match_id", "novelty_match_id",
            "reference_id", "closest_match", "material_id", "mp_id"]


def find_key(d, candidates):
    for k in candidates:
        if k in d:
            return k
    return None


def load_detailed(path):
    with open(path) as f:
        return json.load(f)


def inspect(gen_base, systems):
    """Print the schema of the first detailed JSON so keys can be confirmed."""
    for s in systems:
        p = Path(gen_base) / s / "detailed_metrics_TRI2024.json"
        if not p.exists():
            continue
        d = load_detailed(p)
        print(f"--- {s}: {p.name} ---")
        print(f"top-level type: {type(d).__name__}")
        if isinstance(d, dict):
            print(f"top-level keys: {list(d.keys())[:20]}")
            for k, v in list(d.items())[:6]:
                t = type(v).__name__
                n = len(v) if hasattr(v, "__len__") else "-"
                print(f"  {k}: {t}, len={n}")
                if isinstance(v, list) and v:
                    print(f"     first element: {v[0]!r}")
        elif isinstance(d, list):
            print(f"list of {len(d)}; first element:")
            print(f"  {d[0]!r}")
        print()
        return  # one is enough
    print("No detailed_metrics_TRI2024.json found.")


def per_structure_verdicts(detailed):
    """Normalize detailed metrics into a list of dicts, one per structure.

    Handles two common layouts:
      A) dict of parallel lists: {"stability":[...], "novelty":[...], ...}
      B) list of dicts:         [{"stability":..., "novelty":...}, ...]
    """
    if isinstance(detailed, list):
        return detailed

    if isinstance(detailed, dict):
        ks = find_key(detailed, K_STABLE)
        if ks and isinstance(detailed[ks], list):
            n = len(detailed[ks])
            keys = [k for k, v in detailed.items()
                    if isinstance(v, list) and len(v) == n]
            return [{k: detailed[k][i] for k in keys} for i in range(n)]
        # sometimes nested under a container key
        for cand in ("structures", "per_structure", "results", "detailed"):
            if cand in detailed:
                return per_structure_verdicts(detailed[cand])

    raise ValueError("Unrecognized detailed_metrics layout; run with --inspect")


def truthy(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return False


def structures_from_entries(detailed):
    """Best source: relaxed structures are embedded in detailed_metrics['entry'].

    Each entry is a serialized pymatgen ComputedStructureEntry whose
    ['structure'] is the RELAXED structure (lattice + sites + forces).
    Using these guarantees index alignment with the verdict lists, since they
    come from the same file.

    Returns (list_of_ase_Atoms, source_label) or (None, None) if unavailable.
    """
    entries = detailed.get("entry") if isinstance(detailed, dict) else None
    if not entries or not isinstance(entries, list):
        return None, None
    try:
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError:
        return None, None

    out = []
    for e in entries:
        struct_dict = e.get("structure") if isinstance(e, dict) else None
        if struct_dict is None:
            return None, None
        s = Structure.from_dict(struct_dict)
        out.append(AseAtomsAdaptor.get_atoms(s))
    return out, "relaxed (from detailed_metrics entry)"


def load_relaxed_file(sysdir):
    """Load the relaxed structures written by --structures_output_path.

    That option produces EITHER a directory OR a single extensionless file, so
    handle both. Returns (structures, src) or (None, None) if not found.
    """
    relaxed = sysdir / "relaxed_TRI2024"

    if relaxed.is_dir():
        xyzs = sorted(relaxed.glob("*.extxyz")) + sorted(relaxed.glob("*.xyz"))
        if xyzs:
            return ase_read(str(xyzs[0]), index=":"), f"relaxed dir:{xyzs[0].name}"
        cifs = sorted(relaxed.glob("*.cif"))
        if cifs:
            return [ase_read(str(c)) for c in cifs], f"relaxed dir:{len(cifs)} cifs"

    if relaxed.is_file():
        with open(relaxed, "rb") as f:
            head = f.read(2)
        if head == b"PK":  # zip of cifs
            import tempfile, zipfile
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(relaxed) as z:
                    z.extractall(td)
                cifs = sorted(Path(td).rglob("*.cif"))
                if cifs:
                    return ([ase_read(str(c)) for c in cifs],
                            f"relaxed zip:{len(cifs)} cifs")
            return None, None
        # Extensionless extxyz: ASE cannot infer format, so state it.
        return (ase_read(str(relaxed), index=":", format="extxyz"),
                "relaxed file (extxyz)")

    for cand in (sysdir / "relaxed_TRI2024.extxyz",
                 sysdir / "relaxed_TRI2024.xyz"):
        if cand.exists():
            return ase_read(str(cand), index=":"), f"relaxed file:{cand.name}"

    return None, None


def load_as_generated(sysdir):
    fallback = sysdir / "generated_crystals.extxyz"
    if fallback.exists():
        return ase_read(str(fallback), index=":"), "as-generated (NOT relaxed)"
    raise FileNotFoundError(f"no structures found in {sysdir}")


def load_structures(sysdir, detailed):
    """Get the RELAXED structures for this system.

    Priority:
      1. `relaxed_TRI2024` (from --structures_output_path). Confirmed to be a
         multi-frame extxyz carrying per-frame `structure_index`,
         `total_energy`, forces and stress. Authoritative relaxed geometry, and
         structure_index lets us join exactly to the verdict lists.
      2. Structures embedded in detailed_metrics['entry'] (serialized
         ComputedStructureEntry), same index order as the verdict lists.
      3. The as-generated extxyz — NOT relaxed; last resort, biases BVSE.
    """
    structs, src = load_relaxed_file(sysdir)
    if structs:
        return structs, src

    structs, src = structures_from_entries(detailed)
    if structs:
        return structs, src

    return load_as_generated(sysdir)


def dedup_known(rows, atoms_list):
    """Keep one representative per matched reference entry.

    mattergen computes uniqueness WITHIN the generated set, so a compound the
    model rediscovered thirty times already contributes a single S&U frame.
    This is a safety net for the cases where that does not hold (e.g. the same
    reference entry reached from two different systems, or a uniqueness
    tolerance looser than the novelty tolerance). Ties are broken on e_hull.
    """
    if not any(r.get("matched_id") for r in rows):
        # No reference identifier in the metrics. Falling back to formula would
        # merge distinct polymorphs of the same composition, which is exactly
        # what the reference band needs to keep. Uniqueness already removed the
        # within-set duplicates, so leave the pool alone.
        return rows, atoms_list

    best = {}
    for row, atoms in zip(rows, atoms_list):
        key = row.get("matched_id") or f"{row['system']}|{row['orig_index']}"
        try:
            e = float(row.get("e_above_hull"))
        except (TypeError, ValueError):
            e = float("inf")
        if key not in best or e < best[key][0]:
            best[key] = (e, row, atoms)
    keep = [(r, a) for _, r, a in best.values()]
    return [r for r, _ in keep], [a for _, a in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_base", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    ap.add_argument("--inspect", action="store_true",
                    help="Print detailed-JSON schema and exit.")
    ap.add_argument("--known_ignore_stability", action="store_true",
                    help="Collect U & NOT-novel regardless of the hull cutoff. "
                         "Useful when a recovered known polymorph is metastable "
                         "and would otherwise be dropped from the reference band.")
    ap.add_argument("--no_known", action="store_true",
                    help="Only write the S.U.N. pool (original behaviour).")
    ap.add_argument("--require_validity", action="store_true",
                    help="Drop structures failing comp_validity or "
                         "structure_validity from BOTH pools. Off by default so "
                         "the S.U.N. pool keeps its usual definition, but the "
                         "counts are always reported.")
    args = ap.parse_args()

    gen_base = Path(args.gen_base)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    if args.inspect:
        inspect(gen_base, systems)
        return

    out_dir = Path(args.out_dir or (gen_base / "SUN_POOL"))
    cif_dir = out_dir / "sun_structures"
    known_cif_dir = out_dir / "known_structures"
    out_dir.mkdir(parents=True, exist_ok=True)
    cif_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_known:
        known_cif_dir.mkdir(parents=True, exist_ok=True)

    pooled, index_rows = [], []
    known_pooled, known_rows = [], []
    totals = {}
    known_possible = True

    for sysname in systems:
        sysdir = gen_base / sysname
        dpath = sysdir / "detailed_metrics_TRI2024.json"
        if not dpath.exists():
            print(f"[{sysname}] no detailed metrics, skipping")
            continue

        detailed = load_detailed(dpath)
        verdicts = per_structure_verdicts(detailed)
        structures, src = load_structures(sysdir, detailed)

        # The relaxed extxyz carries `structure_index` per frame (the index into
        # the original generated set). Use it for an EXACT join with the verdict
        # lists, rather than trusting positional order -- relaxation can drop or
        # reorder frames.
        by_index = {}
        for atoms in structures:
            si = atoms.info.get("structure_index")
            if si is not None:
                by_index[int(si)] = atoms

        if by_index and len(by_index) == len(structures):
            match_mode = "structure_index"
        else:
            match_mode = "positional"
            if len(verdicts) != len(structures):
                print(f"[{sysname}] WARNING: {len(verdicts)} verdicts vs "
                      f"{len(structures)} structures ({src}), and no "
                      f"structure_index to join on. Using positional min().")

        if "NOT relaxed" in src:
            print(f"[{sysname}] !! using UNRELAXED structures. BVSE / bottleneck")
            print(f"[{sysname}] !! analysis will be biased. Investigate relaxed output.")

        n = len(verdicts)

        k_sun = find_key(verdicts[0], K_SUN)
        ks = find_key(verdicts[0], K_STABLE)
        ku = find_key(verdicts[0], K_UNIQUE)
        kn = find_key(verdicts[0], K_NOVEL)
        ke = find_key(verdicts[0], K_EHULL)
        km = find_key(verdicts[0], K_MATCH)
        kcv = find_key(verdicts[0], K_COMPVAL)
        ksv = find_key(verdicts[0], K_STRVAL)
        krm = find_key(verdicts[0], K_RMSD)
        kex = find_key(verdicts[0], K_EXPL)

        # `novel`/`unique` and `is_novel`/`is_unique` can disagree: in several
        # mattergen versions the un-prefixed arrays are masked to False for
        # structures that failed validity, while the `is_` arrays are raw.
        # Using a masked `novel` to define "not novel" would sweep every
        # INVALID structure into the known pool. Report the disagreement rather
        # than silently picking one.
        for a, b in (("novel", "is_novel"), ("unique", "is_unique")):
            if a in verdicts[0] and b in verdicts[0]:
                dis = sum(1 for v in verdicts if truthy(v.get(a)) != truthy(v.get(b)))
                if dis:
                    na = sum(1 for v in verdicts if truthy(v.get(a)))
                    nb = sum(1 for v in verdicts if truthy(v.get(b)))
                    print(f"[{sysname}] {a}={na} vs {b}={nb}, disagree on {dis}"
                          f"/{n}. Using '{b}' (raw) for the known pool; '{a}' is"
                          f" likely masked by validity.")

        if k_sun is None and not (ks and ku and kn):
            raise SystemExit(f"[{sysname}] could not find S.U.N. keys. "
                             f"Available: {list(verdicts[0].keys())}\n"
                             f"Run with --inspect and update the K_* lists.")

        # The known pool is S & U & NOT-novel, which cannot be reconstructed
        # from the combined verdict alone: `not novel_unique_stable` conflates
        # "known" with "unstable" and "duplicate". Individual keys required.
        sys_known_ok = bool(ku and kn) and not args.no_known
        if not sys_known_ok and not args.no_known:
            known_possible = False
            print(f"[{sysname}] cannot build the known pool: individual "
                  f"unique/novel keys absent (have {list(verdicts[0].keys())})")

        def is_sun(v):
            if k_sun is not None:
                return truthy(v.get(k_sun))
            return truthy(v.get(ks)) and truthy(v.get(ku)) and truthy(v.get(kn))

        def is_known(v):
            if not truthy(v.get(ku)) or truthy(v.get(kn)):
                return False
            if args.known_ignore_stability:
                return True
            return truthy(v.get(ks)) if ks else True

        def fetch(i):
            if match_mode == "structure_index":
                return by_index.get(i)
            return structures[i] if i < len(structures) else None

        n_pass = n_known = n_missing = n_invalid = 0
        for i in range(n):
            v = verdicts[i]
            sun, known = is_sun(v), (sys_known_ok and is_known(v))
            if not (sun or known):
                continue

            atoms = fetch(i)
            if atoms is None:
                n_missing += 1
                continue

            cval = truthy(v.get(kcv)) if kcv else True
            sval = truthy(v.get(ksv)) if ksv else True
            if not (cval and sval):
                n_invalid += 1
                if args.require_validity:
                    continue

            atoms = atoms.copy()
            atoms.info["system"] = sysname
            atoms.info["orig_index"] = i
            atoms.info["bucket"] = "sun" if sun else "known"
            atoms.info["comp_validity"] = bool(cval)
            atoms.info["structure_validity"] = bool(sval)
            if ke is not None and v.get(ke) is not None:
                atoms.info["e_above_hull"] = v[ke]
            if krm is not None and v.get(krm) is not None:
                atoms.info["rmsd_from_relaxation"] = v[krm]
            if kex is not None and v.get(kex) is not None:
                atoms.info["is_explored"] = bool(truthy(v.get(kex)))
            matched = v.get(km) if km else None
            if matched is not None:
                atoms.info["matched_id"] = str(matched)

            formula = atoms.get_chemical_formula(mode="metal", empirical=True)
            row = {
                "system": sysname,
                "orig_index": i,
                "formula": formula,
                "e_above_hull": v.get(ke) if ke else "",
                "comp_validity": cval,
                "structure_validity": sval,
                "rmsd": v.get(krm) if krm else "",
            }

            if sun:
                cif_name = f"{sysname}_{i:05d}_{formula}.cif"
                ase_write(str(cif_dir / cif_name), atoms)
                pooled.append(atoms)
                index_rows.append({**row, "cif": cif_name})
                n_pass += 1
            else:
                cif_name = f"{sysname}_{i:05d}_{formula}.cif"
                known_pooled.append(atoms)
                known_rows.append({**row, "matched_id": str(matched) if matched
                                   else "", "cif": cif_name})
                n_known += 1

        if n_missing:
            print(f"[{sysname}] note: {n_missing} verdicts had no matching "
                  f"structure (skipped)")

        totals[sysname] = (n, n_pass, n_known, src)
        print(f"[{sysname}] S.U.N. {n_pass}/{n}   known {n_known}/{n}   "
              f"({src}, join={match_mode})")
        if n_invalid:
            verb = "dropped" if args.require_validity else "KEPT"
            print(f"[{sysname}] !! {n_invalid} selected structures fail "
                  f"comp/structure validity ({verb}). Charge-unbalanced "
                  f"compositions are not electrolytes.")
        if n <= 1:
            print(f"  !! ONLY {n} STRUCTURE EVALUATED for {sysname}.")
            print(f"  !! Expected ~2000. mattergen-evaluate likely read only the")
            print(f"  !! first frame. Check the frame count in")
            print(f"  !!   {sysdir}/generated_crystals.extxyz")

    if not pooled:
        raise SystemExit("No S.U.N. structures found.")

    pooled_xyz = out_dir / "sun_structures.extxyz"
    ase_write(str(pooled_xyz), pooled)

    idx_csv = out_dir / "sun_index.csv"
    with open(idx_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "orig_index", "formula",
                                          "e_above_hull", "comp_validity",
                                          "structure_validity", "rmsd", "cif"])
        w.writeheader()
        w.writerows(index_rows)

    known_xyz = known_csv = None
    if known_pooled:
        before = len(known_pooled)
        known_rows, known_pooled = dedup_known(known_rows, known_pooled)
        if len(known_pooled) != before:
            print(f"\nknown pool: {before} -> {len(known_pooled)} after "
                  f"collapsing repeated reference matches")
        for a, r in zip(known_pooled, known_rows):
            ase_write(str(known_cif_dir / r["cif"]), a)
        known_xyz = out_dir / "known_recovered.extxyz"
        ase_write(str(known_xyz), known_pooled)
        known_csv = out_dir / "known_index.csv"
        with open(known_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["system", "orig_index", "formula",
                                              "e_above_hull", "comp_validity",
                                              "structure_validity", "rmsd",
                                              "matched_id", "cif"])
            w.writeheader()
            w.writerows(known_rows)

    print()
    print(f"{'system':<12} {'N':>7} {'S.U.N.':>8} {'known':>7} {'SUN rate':>10}")
    print("-" * 48)
    tot_n = tot_p = tot_k = 0
    for s, (n, p, k, _) in totals.items():
        tot_n += n; tot_p += p; tot_k += k
        print(f"{s:<12} {n:>7} {p:>8} {k:>7} {p/n*100 if n else 0:>9.1f}%")
    print("-" * 48)
    print(f"{'TOTAL':<12} {tot_n:>7} {tot_p:>8} {tot_k:>7} "
          f"{tot_p/tot_n*100 if tot_n else 0:>9.1f}%")
    print()
    print(f"S.U.N. pool     : {pooled_xyz}  ({len(pooled)} frames)")
    print(f"S.U.N. CIFs     : {cif_dir}")
    print(f"S.U.N. index    : {idx_csv}")
    if known_xyz:
        n_id = sum(1 for r in known_rows if r["matched_id"])
        print(f"Known pool      : {known_xyz}  ({len(known_pooled)} frames)")
        print(f"Known index     : {known_csv}"
              f"   [{n_id}/{len(known_rows)} carry a matched_id]")
        if not n_id:
            print("  !! No matched_id in the detailed metrics. The known pool is")
            print("  !! usable as a reference band but cannot be cross-checked")
            print("  !! against ICSD provenance or measured conductivities.")
    elif not args.no_known:
        if not known_possible:
            print("Known pool      : NOT WRITTEN (individual U/N keys missing)")
        else:
            print("Known pool      : empty (the model recovered nothing known)")
    print()
    print("-> S.U.N. pool     : tag_oxidation.py -> na_3d_eta.py  (candidates)")
    if known_xyz:
        print("-> known pool      : the same two scripts, separate --out_dir.")
        print("   Use it as a reference band only. Never merge it into the")
        print("   candidate pool, and do not call it a validation set unless the")
        print("   matched entry has ICSD provenance and a measured conductivity.")


if __name__ == "__main__":
    main()
