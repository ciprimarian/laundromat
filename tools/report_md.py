#!/usr/bin/env python3
"""Render scored findings to a German audit memo (Pruefbericht.md).

Usage:
  python tools/report_md.py [dossier_path] [-o path]
Default output: Pruefbericht.md in the current directory (not committed).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from laundromat.contracts import Finding, Tier  # noqa: E402
from laundromat.pipeline import run  # noqa: E402


def _fmt_amt(x: Decimal | None) -> str:
    if x is None:
        return "–"
    s = f"{abs(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} EUR"


def _tier_de(t: Tier) -> str:
    return {
        Tier.HIGH: "HOCH",
        Tier.MEDIUM: "MITTEL",
        Tier.REVIEW: "PRUEFEN",
        Tier.DISMISSED: "VERWORFEN",
    }.get(t, t.value)


def render(dossier, flags, findings: list[Finding]) -> str:
    by_tier = Counter(f.tier.value for f in findings)
    by_family = Counter()
    for fl in flags:
        by_family[fl.family.value] += 1

    lines: list[str] = []
    lines.append("# Pruefbericht")
    lines.append("")
    lines.append(f"**Dossier:** {dossier.name}")
    lines.append("")
    lines.append("## Abdeckung")
    lines.append("")
    lines.append(f"- Buchungen: **{len(dossier.postings)}**")
    lines.append(f"- Entitaeten: **{len(dossier.entities)}**")
    lines.append(f"- Dokumente: **{len(dossier.documents)}**")
    lines.append(f"- Flags (Rohsignale): **{len(flags)}**")
    lines.append(f"- Feststellungen (bewertet): **{len(findings)}**")
    lines.append(
        f"- Stufen: HOCH={by_tier.get('high', 0)}, "
        f"MITTEL={by_tier.get('medium', 0)}, "
        f"PRUEFEN={by_tier.get('review', 0)}, "
        f"VERWORFEN={by_tier.get('dismissed', 0)}"
    )
    if by_family:
        fam = ", ".join(f"{k}={v}" for k, v in sorted(by_family.items()))
        lines.append(f"- Flags nach Familie: {fam}")
    if dossier.unparsed:
        lines.append(f"- Unverarbeitet: {len(dossier.unparsed)} Datei(en)")
    lines.append("")
    lines.append(
        "Bewertung basiert auf unabhaengigen Lens-Familien. "
        "HOCH erfordert in der Regel mindestens drei Familien; "
        "Einzelmeinungen bleiben in PRUEFEN."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Feststellungen (nach Score)")
    lines.append("")

    # omit dismissed from the printable memo (still counted above)
    reportable = [f for f in findings if f.tier != Tier.DISMISSED]
    if not reportable:
        lines.append("_Keine meldepflichtigen Feststellungen._")
        lines.append("")
        return "\n".join(lines)

    for i, f in enumerate(reportable, 1):
        name = ""
        if f.subject_kind == "entity" and f.subject_id in dossier.entities:
            name = dossier.entities[f.subject_id].name
        title = f.subject_id + (f" – {name}" if name else "")
        fams = ", ".join(sorted(x.value for x in f.families))
        lines.append(f"### {i}. {title}")
        lines.append("")
        lines.append(
            f"- **Stufe:** {_tier_de(f.tier)}  "
            f"| **Score:** {f.score:.2f}  "
            f"| **Betrag:** {_fmt_amt(f.max_amount)}  "
            f"| **Art:** {f.subject_kind}  "
            f"| **Familien:** {fams}"
        )
        lines.append("")

        # one-paragraph rationale from flags
        bits = []
        for fl in f.flags:
            bits.append(f"{fl.title} ({fl.lens_id}, {fl.family.value}): {fl.rationale}")
        if f.defense_note:
            bits.append(f"Entlastungshinweis: {f.defense_note}")
        para = " ".join(bits)
        if len(para) > 1200:
            para = para[:1197] + "..."
        lines.append(para)
        lines.append("")
        lines.append("**Nachweise:**")
        lines.append("")
        seen: set[tuple] = set()
        for fl in f.flags:
            for ev in fl.evidence:
                key = (ev.file, ev.line, ev.page, ev.excerpt[:80])
                if key in seen:
                    continue
                seen.add(key)
                loc = ev.file
                if ev.line is not None:
                    loc += f":{ev.line}"
                if ev.page is not None:
                    loc += f" S.{ev.page}"
                if ev.sheet:
                    loc += f" [{ev.sheet}]"
                excerpt = (ev.excerpt or "").replace("\n", " ").strip()
                if len(excerpt) > 240:
                    excerpt = excerpt[:237] + "..."
                lines.append(f"- `{loc}`")
                if excerpt:
                    lines.append(f"  > {excerpt}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("_Ende des Berichts._")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dossier", nargs="?", default="data/practice")
    p.add_argument("-o", "--output", default="Pruefbericht.md")
    args = p.parse_args(argv)

    print(f"running pipeline on {args.dossier} ...")
    dossier, flags, findings = run(args.dossier)
    text = render(dossier, flags, findings)
    out = Path(args.output)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(findings)} findings, {len(flags)} flags)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
