#!/usr/bin/env bash
# Reproduce the screening funnel of the manuscript from the S.U.N. pool.
#
# Author: Seho Yi (yiseho207@gmail.com), Dongguk University
#
# Step 0 (collect_sun_structures.py) is not run here: it needs the generator's
# own evaluation output, which is far larger than this archive. Its product,
# data/sun_structures.extxyz, is included so that everything downstream of it
# reproduces without regenerating anything.
#
#   usage:  bash run_all.sh [OUT_DIR]

set -euo pipefail

OUT="${1:-out}"
POOL="data/sun_structures.extxyz"
RCUT=10.0          # candidate-edge cutoff; the retained set is invariant
                   # over 10 to 20 Ang (Supplementary Note 3)
THRESH=0.65        # r_3D threshold, placed in the unpopulated interval

[[ -f "$POOL" ]] || { echo "missing $POOL"; exit 1; }
mkdir -p "$OUT"

echo "=== 1/3  composition validation"
python filter_ternary.py --pool "$POOL" --out_dir "$OUT"

echo "=== 2/3  formal oxidation state"
python tag_oxidation.py --pool "$OUT/ternary_pool.extxyz" --out_dir "$OUT"

echo "=== 3/3  percolation dimensionality, r_3D and eta"
python na_3d_eta.py --pool "$OUT/ternary_tagged.extxyz" --out_dir "$OUT" \
    --rcut "$RCUT"

echo
echo "=== finalists at r_3D >= $THRESH Ang"
python - "$OUT/dimensionality_index.csv" "$THRESH" <<'PY'
import csv, io, sys
path, thresh = sys.argv[1], float(sys.argv[2])
rows = list(csv.DictReader(io.StringIO(
    "".join(l for l in open(path) if not l.startswith("#")))))
sel = [r for r in rows if r["r_3D"] and float(r["r_3D"]) >= thresh]
for r in sorted(sel, key=lambda r: -float(r["r_3D"])):
    print(f"  {r['formula']:12s} {r['system']:10s} "
          f"r_3D {float(r['r_3D']):.4f}  eta {float(r['eta']):.4f}")
print(f"  -> {len(sel)} structures")
PY

cat <<'EOF2'

Expected, as reported in the manuscript:
  971 S.U.N. -> 948 exact ternary -> 844 admissible valence
  -> 460 three-dimensionally percolating -> 4 above the r_3D threshold
  Na5SnCl8 0.7444 | Na2SnCl4 0.7297 | Na5TiCl8 0.7111 | Na5AlCl8 0.6983
EOF2
