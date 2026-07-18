"""Lens package: importing it imports every lens module, which registers
itself via the @register decorator. A module that fails to import is skipped
and reported, never fatal -- one broken lens must not kill the pipeline.
"""

from __future__ import annotations

import importlib
import pkgutil

FAILED: list[tuple[str, str]] = []  # (module, reason), shown in the coverage panel

for _info in pkgutil.iter_modules(__path__):
    if _info.name.startswith("_"):
        continue
    try:
        importlib.import_module(f"{__name__}.{_info.name}")
    except Exception as e:
        FAILED.append((_info.name, f"{type(e).__name__}: {e}"))
