#!/usr/bin/env python3
"""Unseen-dossier insurance: rename every path, re-run the pipeline, compare.

Copies data/practice (or given path) to a temp tree with directories and files
renamed to English-ish placeholders. GDPdU index.xml URL entries are rewritten
to match. Runs the full flag pipeline on original and renamed trees and flags
any lens whose count drops to zero on the renamed copy while non-zero on the
original (filename dependency).

Usage:
  python tools/generalize_check.py [dossier_path]
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# Deterministic renames. Unlisted names get a generic english token + counter.
_DIR_MAP = {
    "kreditoren": "vendors",
    "debitoren": "customers",
    "sachkonten": "general_ledger",
    "begleitdokumente": "supporting_docs",
    "anlagen": "fixed_assets",
    "av": "fixed_assets",
    "steuercodes": "tax_codes",
    "practice": "renamed_dossier",
}

_FILE_STEM_MAP = {
    "lieferanten": "suppliers",
    "lieferantenbuchungen": "supplier_postings",
    "kunden": "customers_master",
    "kundenbuchungen": "customer_postings",
    "sachkonten": "accounts",
    "sachkontobuchungen": "journal_entries",
    "anlagen": "assets_master",
    "anlagenbuchungen": "asset_postings",
    "wareneingangsliste_2025": "goods_receipts",
    "warenausgangsliste_2025": "goods_issues",
    "freigabe-log_journale_2025": "approval_log",
    "stammdatenaenderungen_2025": "master_data_changes",
    "buchungen_folgeperiode_2026": "next_period_postings",
    "fakturajournal_2025": "sales_invoice_journal",
    "fakturajournal_januar_2026_kreditoren": "purchase_invoice_journal",
    "kreditlimitliste_debitoren_2025": "credit_limits",
    "gesellschafterliste_beteiligungen": "shareholders",
    "saldenliste_2025": "trial_balance",
    "saldenliste_2024_vorjahr": "trial_balance_prior",
    "op-liste_debitoren_2025": "open_items_ar",
    "op-liste_kreditoren_2025": "open_items_ap",
    "berechtigungsauswertung_2025": "permissions",
    "abstimmung_nebenbuecher_hb_2025": "reconciliation_bridge",
    "ja-entwurf_2025_auszug_bilanz_guv": "draft_financials",
    "exportprotokoll_gdpdu_2025": "export_protocol",
    "it-bestaetigung_vollstaendigkeit_2025": "it_confirmation",
    "pruefungsplanung_jet_2025": "audit_plan",
}

_GENERIC = {
    ".txt": "table",
    ".csv": "sheet",
    ".xlsx": "workbook",
    ".pdf": "document",
    ".docx": "brief",
    ".xml": "index",
    ".dtd": "schema",
}


def _key(name: str) -> str:
    return name.casefold().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")


def _new_stem(stem: str, counters: Counter) -> str:
    k = _key(stem)
    if k in _FILE_STEM_MAP:
        return _FILE_STEM_MAP[k]
    # strip year/suffix noise for partial match
    base = re.sub(r"[_-]?\d{4}.*$", "", k)
    if base in _FILE_STEM_MAP:
        return _FILE_STEM_MAP[base]
    for known, eng in _FILE_STEM_MAP.items():
        if known in k or k in known:
            return eng
    counters[stem] += 1
    return f"file_{counters[stem]:03d}"


def _new_dirname(name: str) -> str:
    k = _key(name)
    if k in _DIR_MAP:
        return _DIR_MAP[k]
    return f"dir_{k[:12]}" if k else "dir_x"


def build_renamed_tree(src: Path, dst: Path) -> dict[str, str]:
    """Copy src -> dst with renamed paths. Returns old_rel -> new_rel map."""
    src = src.resolve()
    mapping: dict[str, str] = {}
    counters: Counter = Counter()

    # walk bottom-up-ish: first collect all paths, then copy with new names
    all_dirs: list[Path] = []
    all_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(src, followlinks=True):
        p = Path(dirpath)
        all_dirs.append(p)
        for fn in filenames:
            if fn.startswith("."):
                continue
            all_files.append(p / fn)

    # map each relative directory
    dir_map: dict[Path, Path] = {src: dst}
    for d in sorted(all_dirs, key=lambda p: len(p.parts)):
        if d == src:
            continue
        parent_new = dir_map[d.parent]
        new_name = _new_dirname(d.name)
        # avoid collisions
        candidate = parent_new / new_name
        n = 2
        while candidate.exists() or candidate in dir_map.values():
            candidate = parent_new / f"{new_name}_{n}"
            n += 1
        dir_map[d] = candidate

    for d, nd in dir_map.items():
        nd.mkdir(parents=True, exist_ok=True)
        if d != src:
            mapping[str(d.relative_to(src))] = str(nd.relative_to(dst))

    for f in all_files:
        parent_new = dir_map[f.parent]
        stem, suf = f.stem, f.suffix
        # keep index.xml / dtd recognizable but still renamed where possible
        if f.name.lower() == "index.xml":
            new_name = "index.xml"  # must stay for GDPdU discovery
        elif f.suffix.lower() == ".dtd":
            new_name = f"schema{f.suffix.lower()}"
        else:
            new_stem = _new_stem(stem, counters)
            new_name = f"{new_stem}{suf.lower() if suf else ''}"
        candidate = parent_new / new_name
        n = 2
        while candidate.exists():
            candidate = parent_new / f"{Path(new_name).stem}_{n}{Path(new_name).suffix}"
            n += 1
        shutil.copy2(f, candidate, follow_symlinks=True)
        mapping[str(f.relative_to(src))] = str(candidate.relative_to(dst))

    # rewrite index.xml URL entries to renamed files
    for dirpath, _, filenames in os.walk(dst, followlinks=True):
        if "index.xml" not in filenames:
            continue
        idx = Path(dirpath) / "index.xml"
        _rewrite_index(idx, Path(dirpath), mapping, src, dst)

    return mapping


def _rewrite_index(idx: Path, table_dir: Path, mapping: dict[str, str], src: Path, dst: Path) -> None:
    """Point each <URL> at the renamed sibling file in the same directory."""
    try:
        tree = ET.parse(idx)
    except ET.ParseError:
        return
    root = tree.getroot()
    # build local old->new basenames from mapping for this directory
    # original dir relative path
    try:
        rel_dir = str(table_dir.relative_to(dst))
    except ValueError:
        rel_dir = ""

    # invert: for files that lived under the original counterpart of table_dir
    # find by scanning mapping values that sit in this new dir
    new_files = {p.name: p for p in table_dir.iterdir() if p.is_file()}
    # map old basename -> new basename among siblings
    old_to_new_base: dict[str, str] = {}
    for old_rel, new_rel in mapping.items():
        op, np = Path(old_rel), Path(new_rel)
        if str(np.parent).replace("\\", "/") == rel_dir.replace("\\", "/") or (
            rel_dir == "" and str(np.parent) in (".", "")
        ):
            old_to_new_base[op.name] = np.name
        # also match when parent path equals via endswith
        if np.parent == Path(rel_dir) or (rel_dir in ("", ".") and len(np.parts) == 1):
            old_to_new_base[op.name] = np.name

    # fallback: unique extension match when only one data file
    changed = False
    for url_el in root.iter("URL"):
        old = (url_el.text or "").strip()
        if not old:
            continue
        base = Path(old).name
        if base in old_to_new_base:
            new_base = old_to_new_base[base]
            if url_el.text != new_base:
                url_el.text = new_base
                changed = True
            continue
        # try case-insensitive
        for ob, nb in old_to_new_base.items():
            if ob.casefold() == base.casefold():
                url_el.text = nb
                changed = True
                break
        else:
            # last resort: if the old basename is not on disk, pick sole non-xml/dtd file
            data = [n for n in new_files if n.lower() not in {"index.xml"} and not n.lower().endswith(".dtd")]
            if len(data) == 1:
                url_el.text = data[0]
                changed = True
    if changed:
        tree.write(idx, encoding="utf-8", xml_declaration=True)


def run_counts(path: str) -> tuple[Counter, int, int, list[tuple[str, str]]]:
    from laundromat.pipeline import run_lenses
    from laundromat.ingest import load_dossier

    dossier = load_dossier(path)
    flags, errors = run_lenses(dossier)
    counts = Counter(f.lens_id for f in flags)
    for lid, err in errors.items():
        counts[f"ERR:{lid}"] = -1
        dossier.unparsed.append((f"<lens:{lid}>", err))
    return counts, len(dossier.postings), len(dossier.documents), dossier.unparsed


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    src = Path(argv[0] if argv else "data/practice")
    if not src.is_dir():
        print(f"FAIL: dossier not found: {src}")
        return 2

    print(f"original: {src}")
    print("running pipeline on original ...")
    orig_counts, orig_posts, orig_docs, orig_unp = run_counts(str(src))
    print(f"  postings={orig_posts} documents={orig_docs} flags={sum(orig_counts.values())}")

    with tempfile.TemporaryDirectory(prefix="laundromat_gen_") as tmp:
        dst = Path(tmp) / "renamed_dossier"
        print(f"building renamed tree at {dst} ...")
        mapping = build_renamed_tree(src, dst)
        print(f"  renamed {len(mapping)} paths")
        # sample
        for i, (a, b) in enumerate(sorted(mapping.items())):
            if i >= 8:
                print("  ...")
                break
            print(f"  {a} -> {b}")

        print("running pipeline on renamed copy ...")
        new_counts, new_posts, new_docs, new_unp = run_counts(str(dst))
        print(f"  postings={new_posts} documents={new_docs} flags={sum(v for v in new_counts.values() if v > 0)}")

    print()
    print("=== lens flag comparison ===")
    print(f"{'lens_id':32s} {'orig':>6s} {'renamed':>8s} {'status':>10s}")
    print("-" * 60)
    all_lenses = sorted(set(orig_counts) | set(new_counts))
    failures: list[str] = []
    warnings: list[str] = []
    for lid in all_lenses:
        if lid.startswith("ERR:"):
            continue
        o = orig_counts.get(lid, 0)
        n = new_counts.get(lid, 0)
        if o > 0 and n == 0:
            status = "FAIL"
            failures.append(lid)
        elif o > 0 and n < o * 0.5:
            status = "WARN"
            warnings.append(lid)
        elif o == 0 and n == 0:
            status = "silent"
        else:
            status = "ok"
        print(f"{lid:32s} {o:6d} {n:8d} {status:>10s}")

    print()
    print("=== ingest volume ===")
    print(f"  postings:  {orig_posts} -> {new_posts}")
    print(f"  documents: {orig_docs} -> {new_docs}")
    if orig_posts and new_posts < orig_posts * 0.9:
        print("  WARN: posting count dropped >10% on rename (ingest path issue)")
        warnings.append("postings_volume")
    if orig_docs and new_docs < orig_docs * 0.5:
        print("  WARN: document count dropped >50% on rename (begleit name/header issue)")
        warnings.append("documents_volume")

    if new_unp:
        print(f"  renamed unparsed ({len(new_unp)}):")
        for f, r in new_unp[:12]:
            print(f"    {f}: {r}")

    print()
    if failures:
        print("FAIL: lenses that went silent on renamed tree (filename dependency):")
        for lid in failures:
            print(f"  - {lid} (was {orig_counts[lid]})")
        print(f"SUMMARY: FAIL ({len(failures)} lens(es), {len(warnings)} warning(s))")
        return 1
    if warnings:
        print(f"SUMMARY: PASS with warnings ({len(warnings)}): {', '.join(warnings)}")
        return 0
    print("SUMMARY: PASS (all active lenses survive rename)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
