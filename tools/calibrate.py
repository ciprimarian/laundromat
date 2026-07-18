#!/usr/bin/env python3
"""Run all registered lenses on a dossier and print fire-rate + family overlap.

Usage:
  python tools/calibrate.py [path]
  python -m tools.calibrate data/practice

Default path: data/practice
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

# allow running from repo root without install
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = argv[0] if argv else "data/practice"

    from laundromat.contracts import REGISTRY, Flag, LensFamily
    from laundromat.ingest import load_dossier
    from laundromat import lenses as _lenses  # noqa: F401  registers all

    print(f"loading {path} ...")
    dossier = load_dossier(path)
    n_post = len(dossier.postings)
    n_ent = len(dossier.entities)
    n_doc = len(dossier.documents)
    print(
        f"dossier: postings={n_post} entities={n_ent} documents={n_doc} "
        f"unparsed={len(dossier.unparsed)}"
    )
    if _lenses.FAILED:
        print("lens import failures:")
        for mod, reason in _lenses.FAILED:
            print(f"  {mod}: {reason}")

    counts: dict[str, int] = {}
    family_of: dict[str, LensFamily] = {}
    errors: dict[str, str] = {}
    all_flags: list[Flag] = []

    for lens_id, lens in sorted(REGISTRY.items()):
        family_of[lens_id] = lens.family
        try:
            flags = list(lens.run(dossier))
        except Exception as e:
            errors[lens_id] = f"{type(e).__name__}: {e}"
            counts[lens_id] = -1
            continue
        counts[lens_id] = len(flags)
        all_flags.extend(flags)

    print()
    print("=== per-lens flag counts ===")
    print(f"{'lens_id':32s} {'family':16s} {'flags':>6s} {'%rows':>8s}  note")
    print("-" * 78)
    by_family: Counter[str] = Counter()
    for lid in sorted(counts):
        c = counts[lid]
        fam = family_of.get(lid)
        fam_s = fam.value if fam else "?"
        if c < 0:
            print(f"{lid:32s} {fam_s:16s} {'ERR':>6s} {'':>8s}  {errors.get(lid, '')}")
            continue
        pct = (100.0 * c / n_post) if n_post else 0.0
        note = ""
        if pct > 2.0:
            note = "HIGH"
        elif c == 0:
            note = "silent"
        by_family[fam_s] += c
        print(f"{lid:32s} {fam_s:16s} {c:6d} {pct:7.2f}%  {note}")

    print()
    print("=== flags by family ===")
    for fam, c in sorted(by_family.items(), key=lambda x: -x[1]):
        print(f"  {fam:16s} {c:5d}")
    print(f"  {'TOTAL':16s} {sum(by_family.values()):5d}")

    # subject = entity_id or doc_no
    subject_families: dict[str, set[str]] = defaultdict(set)
    subject_flags: dict[str, list[Flag]] = defaultdict(list)
    for f in all_flags:
        subj = f.entity_id or f.doc_no
        if not subj:
            # lens-level / partition without subject: skip overlap table
            continue
        subject_families[subj].add(f.family.value)
        subject_flags[subj].append(f)

    hist: Counter[int] = Counter()
    for subj, fams in subject_families.items():
        hist[len(fams)] += 1

    print()
    print("=== family overlap (subjects with entity_id or doc_no) ===")
    print(f"subjects with >=1 flag: {len(subject_families)}")
    for k in sorted(hist):
        label = f"{k} famil{'y' if k == 1 else 'ies'}"
        print(f"  {label:12s} {hist[k]:5d} subjects")

    multi = [(s, fams) for s, fams in subject_families.items() if len(fams) >= 2]
    multi.sort(key=lambda x: (-len(x[1]), x[0]))
    if multi:
        print()
        print("=== top multi-family subjects (up to 25) ===")
        for subj, fams in multi[:25]:
            fl = subject_flags[subj]
            amts = [f.amount for f in fl if f.amount is not None]
            max_amt = max(amts) if amts else None
            lenses = sorted({f.lens_id for f in fl})
            print(
                f"  {subj:16s} families={sorted(fams)} "
                f"n_flags={len(fl)} lenses={lenses}"
                + (f" max_amt={max_amt}" if max_amt is not None else "")
            )

    if errors:
        print()
        print("=== errors ===")
        for lid, err in errors.items():
            print(f"  {lid}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
