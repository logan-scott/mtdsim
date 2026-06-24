"""Run manifest — records exactly what produced a set of outputs.

For reproducibility and auditability, every ``run_all`` invocation writes a JSON
manifest capturing: the resolved configuration, the master seed, the git commit
(if available), timestamps, the environment (Python + pinned package versions),
and a SHA-256 hash of every generated figure/table file. Re-running with the
same config should reproduce identical aggregate numbers; the manifest lets a
reviewer verify that the artifacts on disk match a given run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__


def _git_commit() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (subprocess.SubprocessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for mod in ("numpy", "pandas", "matplotlib", "yaml"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            versions[mod] = "missing"
    return versions


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    *,
    config_fingerprint: dict[str, Any],
    config_path: str | None,
    seed: int,
    experiments: dict[str, Any],
    output_files: list[str],
) -> dict[str, Any]:
    """Assemble the manifest dict (does not write it)."""
    hashes = {}
    for f in sorted(set(output_files)):
        if Path(f).exists():
            hashes[f] = file_sha256(f)
    return {
        "tool": "mtdsim",
        "version": __version__,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config_path": config_path,
        "seed": seed,
        "git": _git_commit(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "config": config_fingerprint,
        "experiments": experiments,
        "output_files": hashes,
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return str(p)
