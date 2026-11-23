"""
Provenance stamping for the screening scripts.

Every number this pipeline produces depends on a convention that is not
implied by the number itself: which reference set novelty was judged
against, which correction scheme the screening energies carry, which
ionic-radius table sets the blockers, and which library versions did the
symmetry and structure matching. The manuscript argues that such
quantities should be reported alongside the result; this module makes the
code do the same, by writing the conventions into every output file.

Author
------
Seho Yi (yiseho207@gmail.com)
Department of Energy and Materials Engineering, Dongguk University, Seoul
"""

from __future__ import annotations

__author__ = "Seho Yi"
__email__ = "yiseho207@gmail.com"
__affiliation__ = ("Department of Energy and Materials Engineering, "
                   "Dongguk University, Seoul, Republic of Korea")

import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

# --- conventions this pipeline was run with -------------------------------
# Update these when the underlying inputs change; they are recorded in every
# output file so that a rerun with different inputs is never mistaken for a
# failed reproduction.
GENERATOR_COMMIT = "a245cf2"          # microsoft/mattergen
REFERENCE_DATASET = "alex_mp_20240607 (TRI2024 bundle, pre-PR#253)"
# The radii themselves live in na_3d_eta.py, which is the only consumer; this
# module imports them at stamp time so the description and the numbers can
# never drift apart. Import is lazy to keep provenance dependency-free.
IONIC_RADII_TABLE = ("Shannon effective ionic radii, six-coordinate values, "
                     "taken per formal oxidation state")


def radii_provenance() -> str:
    """One line describing the blocker radii actually in use, read from the
    table that the geometric analysis applies."""
    try:
        from na_3d_eta import (ANION_RADII, RADII_OX, RADII_DEFAULT,
                               ESTIMATED, ML_EXTENDED)
    except ImportError:
        return IONIC_RADII_TABLE + " (table not importable here)"
    n_est = len(ESTIMATED)
    n_ml = len(ML_EXTENDED)
    return (f"{IONIC_RADII_TABLE}; {len(RADII_OX)} cation entries "
            f"({n_est} estimated where Shannon has no CN=6 value, "
            f"{n_ml} from the machine-learned extension of Baloch et al. 2021), "
            f"{len(ANION_RADII)} anion entries, default {RADII_DEFAULT} Ang "
            f"for anything unlisted (na_3d_eta.RADII_OX)")


ENERGY_CORRECTION = "TRI110PBE as distributed with the generator at the commit above"

_PACKAGES = ("pymatgen", "ase", "numpy", "scipy")

def _pkg_versions() -> dict[str, str]:
    out = {}
    for p in _PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "not installed"
    return out

def _git_revision() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if rev.returncode == 0:
            return rev.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"

def stamp(script: str, extra: dict | None = None) -> list[str]:
    """Provenance lines, each already prefixed with '#'."""
    lines = [
        f"script: {script}",
        f"author: {__author__} <{__email__}>, {__affiliation__}",
        f"analysis code revision: {_git_revision()}",
        f"run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"python: {platform.python_version()}",
        "packages: " + ", ".join(f"{k} {v}" for k, v in _pkg_versions().items()),
        f"generator commit: {GENERATOR_COMMIT}",
        f"novelty reference set: {REFERENCE_DATASET}",
        f"energy correction: {ENERGY_CORRECTION}",
        f"ionic radii: {radii_provenance()}",
    ]
    for k, v in (extra or {}).items():
        lines.append(f"{k}: {v}")
    return ["# " + l for l in lines]

def write_header(fh, script: str, extra: dict | None = None) -> None:
    fh.write("\n".join(stamp(script, extra)) + "\n")

def report_block(script: str, extra: dict | None = None) -> str:
    return "\n".join(l[2:] for l in stamp(script, extra))
