#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copy SerapiCore marker YAMLs from a source (folder OR zip) into a destination markers tree.
- Families: ATO, SEM, CLU, CLU_INTUITION
- Reads marker names from a --bundle YAML (SerapiCore_1.0.yaml) if provided,
  otherwise uses the built-in SerapiCore set defined below.
- Works 100% offline. No external downloads.
"""

import argparse
import json
import os
import re
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

BUILTIN_SET: Dict[str, List[str]] = {
    "ATO": [
        "ATO_PAUSE_MICRO",
        "ATO_PAUSE_LONG",
        "ATO_RISING_INTONATION",
        "ATO_TEMPO_SLOW",
        "ATO_TEMPO_FAST",
        "ATO_SPEECH_RATE_VARIATION",
        "ATO_PITCH_SPREAD_WIDE",
        "ATO_PITCH_SPREAD_NARROW",
        "ATO_LAUGHTER_EVENT",
        "ATO_SIGH",
        "ATO_JITTER_SHIMMER_HIGH",
        "ATO_LONG_RESPONSE_GAP",
        "ATO_HEDGING_CUE",
    ],
    "SEM": [
        "SEM_VALIDATION_OF_FEELING",
        "SEM_VALIDATION_SEEKING",
        "SEM_BOUNDARY_SETTING",
        "SEM_REFLECTIVE_BOUNDARY_SETTING",
        "SEM_SHARED_GOAL_FRAMING",
        "SEM_COMMITMENT_REQUEST",
        "SEM_SOFT_COMMITMENT_MARKER",
        "SEM_DOUBT_UNCERTAINTY",
        "SEM_GUILT_FRAMING",
        "SEM_BLAME_SHIFTING",
    ],
    "CLU": [
        "CLU_EMOTIONAL_SUPPORT",
        "CLU_GROWING_CONNECTION_CLUSTER",
        "CLU_GOTTMAN_TOXICITY_CLUSTER",
        "CLU_INDIRECT_CONFLICT_AVOIDANCE",
        "CLU_DEFENSIVE_RETREAT",
        "CLU_PROCRASTINATION_LOOP",
        "CLU_NEEDINESS_GUILT_BIND",
        "CLU_MISUNDERSTANDING_AUDIO",
    ],
    "CLU_INTUITION": [
        "CLU_INTUITION_SUPPORT",
        "CLU_INTUITION_CONFLICT",
        "CLU_INTUITION_COMMITMENT",
        "CLU_INTUITION_UNCERTAINTY",
        "CLU_INTUITION_GRIEF",
    ],
}

def load_bundle(bundle_path: Optional[Path]) -> Dict[str, List[str]]:
    """Load includes list from SerapiCore_1.0.yaml. Falls back to BUILTIN_SET."""
    if not bundle_path:
        return BUILTIN_SET
    text = Path(bundle_path).read_text(encoding="utf-8")
    # Try to parse includes section with a light-weight approach (no PyYAML needed).
    # Expect structure:
    # includes:
    #   ATO:
    #     - NAME
    #   SEM:
    #     - NAME
    includes: Dict[str, List[str]] = {"ATO": [], "SEM": [], "CLU": [], "CLU_INTUITION": []}
    current_key = None
    in_includes = False
    for line in text.splitlines():
        if re.match(r"^\s*includes\s*:\s*$", line):
            in_includes = True
            continue
        if in_includes:
            m_key = re.match(r"^\s*([A-Z_]+)\s*:\s*$", line)
            if m_key:
                key = m_key.group(1)
                if key in includes:
                    current_key = key
                else:
                    current_key = None
                continue
            m_item = re.match(r"^\s*-\s*([A-Za-z0-9_]+)\s*$", line)
            if m_item and current_key:
                includes[current_key].append(m_item.group(1))
            # stop conditions (another top-level key)
            if re.match(r"^\S", line) and not re.match(r"^\s", line) and not re.match(r"^\s*includes", line):
                # another top-level section began
                break
    # Fallback if parse failed
    if not any(includes.values()):
        return BUILTIN_SET
    return includes

def build_stem_index(root: Path, extensions=(".yaml", ".yml")) -> Dict[str, List[Path]]:
    """Create a mapping from filename stem to candidate paths (recursive)."""
    idx: Dict[str, List[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            stem = p.stem  # filename without suffix
            idx.setdefault(stem, []).append(p)
    return idx

def select_best_candidate(candidates: List[Path], family: str) -> Path:
    """If multiple files match, prefer those that contain the family name in their parents."""
    # priority 1: any parent folder equals family (case-insensitive)
    for c in candidates:
        parts = [pp.lower() for pp in c.parts]
        if family.lower() in parts:
            return c
        # relaxed: substring on directory names
        if any(family.lower() in pp.lower() for pp in c.parts):
            return c
    # priority 2: shortest path (fewer directories)
    return sorted(candidates, key=lambda x: len(x.parts))[0]

def read_text_safe(p: Path, max_bytes=200_000) -> Optional[str]:
    try:
        size = p.stat().st_size
        if size > max_bytes:
            return None
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

def search_by_id_field(root: Path, marker_name: str, extensions=(".yaml", ".yml")) -> Optional[Path]:
    """Slow fallback: look into files and search for `id: <marker_name>`."""
    pat = re.compile(rf"^\s*id\s*:\s*{re.escape(marker_name)}\s*$", re.IGNORECASE | re.MULTILINE)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            txt = read_text_safe(p)
            if txt and pat.search(txt):
                return p
    return None

def copy_file(src: Path, dst: Path, overwrite: bool) -> Tuple[bool, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return False, f"exists: {dst}"
    shutil.copy2(src, dst)
    return True, f"copied: {src} -> {dst}"

# ---- ZIP support ----
import zipfile

class ZipSearcher:
    def __init__(self, zip_path: Path):
        self.z = zipfile.ZipFile(zip_path, "r")
        self.members = [m for m in self.z.infolist() if not m.is_dir() and (m.filename.lower().endswith(".yaml") or m.filename.lower().endswith(".yml"))]
        self.index: Dict[str, List[zipfile.ZipInfo]] = {}
        for m in self.members:
            stem = Path(m.filename).stem
            self.index.setdefault(stem, []).append(m)

    def select_best(self, markers: List[zipfile.ZipInfo], family: str) -> zipfile.ZipInfo:
        # prefer ones with family name in path
        for m in markers:
            parts = [p.lower() for p in Path(m.filename).parts]
            if family.lower() in parts or any(family.lower() in pp for pp in parts):
                return m
        # else shortest path
        return sorted(markers, key=lambda m: len(Path(m.filename).parts))[0]

    def search_by_id(self, marker_name: str) -> Optional[zipfile.ZipInfo]:
        pat = re.compile(rf"^\s*id\s*:\s*{re.escape(marker_name)}\s*$", re.IGNORECASE | re.MULTILINE)
        for m in self.members:
            try:
                data = self.z.read(m)
                if len(data) > 200_000:
                    continue
                txt = data.decode("utf-8", errors="ignore")
                if pat.search(txt):
                    return m
            except Exception:
                continue
        return None

    def extract_to(self, member: zipfile.ZipInfo, dst: Path, overwrite: bool) -> Tuple[bool, str]:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not overwrite:
            return False, f"exists: {dst}"
        with self.z.open(member, "r") as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out)
        return True, f"copied(zip): {member.filename} -> {dst}"

def run_from_folder(src_root: Path, dst_root: Path, includes: Dict[str, List[str]], overwrite: bool, dry_run: bool) -> Dict:
    index = build_stem_index(src_root)
    report = {"source": str(src_root), "dest": str(dst_root), "copied": [], "skipped": [], "missing": [], "duplicates": []}
    for family, names in includes.items():
        for name in names:
            candidates = index.get(name, [])
            chosen = None
            if len(candidates) == 1:
                chosen = candidates[0]
            elif len(candidates) > 1:
                chosen = select_best_candidate(candidates, family)
                report["duplicates"].append({"marker": name, "candidates": [str(c) for c in candidates], "chosen": str(chosen)})
            else:
                # fallback: search by 'id:' field
                alt = search_by_id_field(src_root, name)
                if alt:
                    chosen = alt
            if not chosen:
                report["missing"].append({"marker": name, "family": family})
                continue
            dst = dst_root / family / f"{name}.yaml"
            if dry_run:
                report["copied"].append({"marker": name, "from": str(chosen), "to": str(dst), "dry_run": True})
            else:
                ok, msg = copy_file(chosen, dst, overwrite)
                (report["copied"] if ok else report["skipped"]).append({"marker": name, "from": str(chosen), "to": str(dst), "msg": msg})
    return report

def run_from_zip(src_zip: Path, dst_root: Path, includes: Dict[str, List[str]], overwrite: bool, dry_run: bool) -> Dict:
    zs = ZipSearcher(src_zip)
    report = {"source_zip": str(src_zip), "dest": str(dst_root), "copied": [], "skipped": [], "missing": [], "duplicates": []}
    for family, names in includes.items():
        for name in names:
            candidates = zs.index.get(name, [])
            chosen = None
            if len(candidates) == 1:
                chosen = candidates[0]
            elif len(candidates) > 1:
                chosen = zs.select_best(candidates, family)
                report["duplicates"].append({"marker": name, "candidates": [c.filename for c in candidates], "chosen": chosen.filename})
            else:
                alt = zs.search_by_id(name)
                if alt:
                    chosen = alt
            if not chosen:
                report["missing"].append({"marker": name, "family": family})
                continue
            dst = dst_root / family / f"{name}.yaml"
            if dry_run:
                report["copied"].append({"marker": name, "from": chosen.filename, "to": str(dst), "dry_run": True})
            else:
                ok, msg = zs.extract_to(chosen, dst, overwrite)
                (report["copied"] if ok else report["skipped"]).append({"marker": name, "from": chosen.filename, "to": str(dst), "msg": msg})
    return report

def main():
    ap = argparse.ArgumentParser(description="Copy SerapiCore markers from folder or zip into ./markers/* structure.")
    ap.add_argument("--src", required=True, help="Pfad zum Quellordner ODER zur ZIP (z.B. Marker_5.0_VOICE.zip)")
    ap.add_argument("--dst", default="./markers", help="Ziel-Wurzelordner (Standard: ./markers)")
    ap.add_argument("--bundle", default=None, help="Optional: Pfad zu SerapiCore_1.0.yaml; sonst Built-In Liste")
    ap.add_argument("--overwrite", action="store_true", help="Vorhandene Dateien überschreiben")
    ap.add_argument("--dry-run", action="store_true", help="Nur anzeigen, nichts kopieren")
    ap.add_argument("--report", default="copy_report.json", help="Pfad für JSON-Report")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    bundle = Path(args.bundle) if args.bundle else None

    includes = load_bundle(bundle)

    if src.is_file() and src.suffix.lower() == ".zip":
        report = run_from_zip(src, dst, includes, overwrite=args.overwrite, dry_run=args.dry_run)
    elif src.is_dir():
        report = run_from_folder(src, dst, includes, overwrite=args.overwrite, dry_run=args.dry_run)
    else:
        print(f"[ERROR] --src ist weder Ordner noch .zip: {src}", file=sys.stderr)
        sys.exit(2)

    # write report
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary to stdout
    copied = len(report["copied"])
    missing = len(report["missing"])
    skipped = len(report["skipped"])
    dups = len(report["duplicates"])
    print(f"[OK] copied={copied}, skipped={skipped}, missing={missing}, duplicates={dups}")
    if missing:
        print("Missing markers:")
        for m in report["missing"]:
            print(f"  - {m['family']}/{m['marker']}")

if __name__ == "__main__":
    main()
