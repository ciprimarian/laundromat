"""Run the whole pipeline over one dossier directory.

Usage: python -m laundromat.pipeline data/practice
"""

from __future__ import annotations

import sys
from collections import Counter

from .contracts import REGISTRY, Dossier, Finding, Flag


def run_lenses(dossier: Dossier) -> tuple[list[Flag], dict[str, str]]:
    """Run every registered lens; a crashing lens is reported, not fatal."""
    from . import lenses  # noqa: F401  (import registers all lens modules)

    flags: list[Flag] = []
    errors: dict[str, str] = {}
    for lens_id, lens in sorted(REGISTRY.items()):
        try:
            flags.extend(lens.run(dossier))
        except Exception as e:
            errors[lens_id] = f"{type(e).__name__}: {e}"
    return flags, errors


def run(path: str) -> tuple[Dossier, list[Flag], list[Finding]]:
    from .ingest import load_dossier

    dossier = load_dossier(path)
    flags, errors = run_lenses(dossier)
    for lens_id, err in errors.items():
        dossier.unparsed.append((f"<lens:{lens_id}>", err))

    findings: list[Finding] = []
    try:
        from .scoring import score_all

        findings = score_all(flags)
    except NotImplementedError:
        pass  # scoring not written yet; flags still usable
    return dossier, flags, findings


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.strip())
        return 2
    dossier, flags, findings = run(argv[0])

    from .lenses import FAILED

    print(f"{dossier.name}: {len(dossier.postings)} postings, "
          f"{len(dossier.entities)} entities, {len(dossier.documents)} documents")
    for mod, reason in FAILED:
        print(f"lens module failed to import: {mod}: {reason}")
    for file, reason in dossier.unparsed:
        print(f"unparsed: {file}: {reason}")

    by_lens = Counter(f.lens_id for f in flags)
    print(f"\nflags: {len(flags)}")
    for lens_id, n in sorted(by_lens.items()):
        print(f"  {lens_id}: {n}")

    if findings:
        by_tier = Counter(f.tier.value for f in findings)
        print(f"\nfindings: {len(findings)}  {dict(by_tier)}")
    else:
        print("\nfindings: scoring not implemented yet")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
