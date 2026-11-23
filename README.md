# Physics-guided screening of generated Na–M–Cl structures

Code and data for the screening funnel of

> **Cation-controlled site topology enables intrinsic sodium-ion transport in
> stoichiometric Na₂SnCl₄**

The pipeline starts from structures proposed by a generative model (MatterGen),
keeps the intended Na–M–Cl ternaries, rejects compositions whose formal valence
cannot exist, and ranks the rest by the geometry of the sodium network. It ends
where the first-principles work of the manuscript begins: four structures, one
of which is Na₂SnCl₄.

Author: Seho Yi (yiseho207@gmail.com), Department of Energy and Materials
Engineering, Dongguk University, Seoul.

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh          # writes everything to out/
```

This runs in a few minutes on one core and prints each stage. The counts it
should reproduce, as reported in the manuscript (Fig. 2a, Supplementary
Table S1):

| stage | script | structures |
|---|---|---|
| stable, unique, novel pool | `collect_sun_structures.py` | 971 |
| exact Na–M–Cl ternary | `filter_ternary.py` | 948 |
| admissible formal valence | `tag_oxidation.py` | 844 |
| three-dimensional Na network | `na_3d_eta.py` | 460 |
| r₃D ≥ 0.65 Å | `na_3d_eta.py` | 4 |

The four survivors (Table 2 of the manuscript, quoted there to three decimals):

| composition | r₃D (Å) | η |
|---|---|---|
| Na₅SnCl₈ | 0.7444 | 0.9115 |
| Na₂SnCl₄ | 0.7297 | 0.9422 |
| Na₅TiCl₈ | 0.7111 | 0.9271 |
| Na₅AlCl₈ | 0.6983 | 0.9243 |

## Repository contents

| file | produces | used for |
|---|---|---|
| `data/sun_structures.extxyz` | — | input: the 971-structure pool (see below) |
| `collect_sun_structures.py` | `sun_structures.extxyz`, `sun_index.csv` | builds the pool from the generator's evaluation output (not run by `run_all.sh`) |
| `filter_ternary.py` | `ternary_pool.extxyz`, `ternary_index.csv`, `composition_report.txt` | Fig. 2a stage 2; Supplementary Table S1 |
| `tag_oxidation.py` | `ternary_tagged.extxyz`, `oxidation_index.csv`, `rejected_index.csv`, `oxidation_report.txt` | Fig. 2a stage 3; Supplementary Table S2 |
| `na_3d_eta.py` | `dimensionality_index.csv`, `dimensionality_report.txt` | Fig. 2a stages 4–5 and Fig. 2c; Tables 1 and 2; Supplementary Table S1, Note 3 and Fig. S2 |
| `cn4_percolation.py` | `cn4_bottleneck.csv` | Fig. 2b (tetrahedral-sodium sublattice) |
| `provenance.py` | — | stamps the conventions below into the header of every output file |
| `run_all.sh` | `out/` | runs the three screening stages and lists the survivors |

`collect_sun_structures.py` needs the generator's own evaluation directory,
which is far larger than this repository; its product is included as
`data/sun_structures.extxyz` so that everything downstream reproduces without
regenerating anything.

To reproduce Fig. 2b after `run_all.sh`:

```bash
python cn4_percolation.py --index out/dimensionality_index.csv \
    --pool out/ternary_tagged.extxyz --screen na_3d_eta.py --rcut 10.0
```

Single structures (for example the DFT-relaxed polymorphs of Table 1) can be
analysed directly:

```bash
python na_3d_eta.py --poscar CONTCAR_* --ox_state 2 --rcut 10.0
```

## Conventions

Every number here depends on a choice the number itself does not carry. These
choices are written into the header of each output file.

**Novelty reference set.** The pool was judged against the reference bundle
distributed with the generator at commit `a245cf2`. That set holds a single
Na₂SnCl₄ entry, so two structures already deposited in Alexandria pass as new.
Both counts are reported in the manuscript.

**Energy correction.** Screening energies carry the `TRI110PBE` scheme as
distributed at that commit, in which the adjustment is computed from an
already anion-corrected energy. This was reported as issue #251 and a
correcting pull request has since been opened upstream. The numbers here
predate that correction; hull distances used for any conclusion were
recomputed from first principles.

**Compositional validity.** The pool does not impose the generator's
composition-validity heuristic (280 of the 971 structures fail it).
Admissibility is decided instead by the explicit electron-counting bound in
`tag_oxidation.py`, whose rejections are written out in full and can be
inspected with `--keep_impossible`.

**Blocker radii.** Six-coordinate Shannon effective ionic radii, keyed by
formal oxidation state.

- Sn²⁺, which Shannon does not list, takes the machine-learned six-coordinate
  value of 1.025 Å (Baloch et al., *Phys. Rev. Mater.* **5**, 043804 (2021)).
- Six further cation entries are estimates: Al²⁺, Ga²⁺, Ga⁺, Sn³⁺, Zr³⁺, Zr²⁺.
- Non-integer formal states are interpolated linearly between neighbouring
  integer states; unlisted species take 0.80 Å.

The three-dimensional count is the one number that depends on the Sn²⁺ value:
an earlier estimate of 1.18 Å gives 457 instead of 460, with the four survivors
unchanged. The table lives in `na_3d_eta.RADII_OX` and `provenance.py` reads
it, so the description and the numbers cannot drift apart.

**Candidate-edge cutoff.** 10 Å, passed by `run_all.sh` (the module default is
6.5 Å). The retained set is unchanged for any cutoff from 10 to 20 Å, and no
structure falls in 0.62 < r₃D < 0.70 Å, so the 0.65 Å threshold is not a tuned
parameter. Sweep the cutoff with `--rcut_sweep`.

## Limits worth knowing before reusing this

- r₃D samples the straight line between sodium sites, so it underestimates the
  passage actually used: in the surviving structure, sodium reaches the next
  site through an intervening vacant octahedron rather than across that line.
- η measures the isotropy of the network shape, not of the diffusivity.
- The graph is a hopping picture on the stoichiometric sodium sublattice and
  represents neither interstitial nor concerted motion.
- The screen was built for precision rather than recall: failing a gate is a
  reason not to spend first-principles time on a candidate, not evidence that
  it lacks function.

## Citing

If you use this code, please cite the manuscript and this archived release:
https://doi.org/10.5281/zenodo.24460375

Released under the MIT License.
