"""Ingestion entry point: one call turns a dossier directory into a Dossier.

Each stage is isolated: a failure lands in dossier.unparsed instead of
killing the run. A missing file costs one lens, a crash costs everything.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts import Dossier


def load_dossier(path: str | Path) -> Dossier:
    root = Path(path)
    dossier = Dossier(name=root.name)
    if not root.is_dir():
        dossier.unparsed.append((str(root), "dossier directory not found"))
        return dossier

    from .gdpdu import load_gdpdu
    try:
        load_gdpdu(root, dossier)
    except Exception as e:
        dossier.unparsed.append(("<gdpdu>", f"stage failed: {e}"))

    try:
        from .begleit import load_begleit
        load_begleit(root, dossier)
    except Exception as e:
        dossier.unparsed.append(("<begleit>", f"stage failed: {e}"))

    try:
        from .docs import load_docs
        load_docs(root, dossier)
    except Exception as e:
        dossier.unparsed.append(("<docs>", f"stage failed: {e}"))

    return dossier
