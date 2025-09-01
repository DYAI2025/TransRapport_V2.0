#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seed/augment marker YAMLs with an `examples` block (positive/negative).

- Reads bundle `bundles/SerapiCore_1.0.yaml` for the whitelist.
- Resolves `markers_root` from `config/app.yaml` (falls back to ./markers).
- For each marker, ensures a YAML exists at ./markers/<FAMILY>/<NAME>.yaml
  (or at configured root) and contains:

  examples:
    positive:
      - "..."
    negative:
      - "..."

Usage:
  python tools/seed_marker_examples.py --dry-run    # print actions
  python tools/seed_marker_examples.py              # write changes

Optional:
  --min-pos, --min-neg: thresholds for validation (default 20/20)
  Returns non-zero exit code if counts are below thresholds (unless --no-validate).
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def read_yaml(p: Path) -> dict:
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def write_yaml(p: Path, data: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_bundle_includes(bundle_path: Path) -> Dict[str, List[str]]:
    data = read_yaml(bundle_path)
    incs = (data or {}).get("includes", {})
    out: Dict[str, List[str]] = {"ATO": [], "SEM": [], "CLU": [], "CLU_INTUITION": []}
    for fam in out.keys():
        out[fam] = list(incs.get(fam, []) or [])
    return out


def resolve_markers_root() -> Path:
    cfg = read_yaml(Path("config/app.yaml"))
    p = (cfg.get("paths", {}) or {}).get("markers_root")
    if p:
        return Path(p)
    return Path("./markers")


def ensure_examples_block(doc: dict) -> dict:
    ex = doc.get("examples")
    if not isinstance(ex, dict):
        doc["examples"] = {"positive": [], "negative": []}
    else:
        if "positive" not in ex or not isinstance(ex["positive"], list):
            ex["positive"] = []
        if "negative" not in ex or not isinstance(ex["negative"], list):
            ex["negative"] = []
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="bundles/SerapiCore_1.0.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--min-neg", type=int, default=20)
    args = ap.parse_args()

    bundle = Path(args.bundle)
    includes = load_bundle_includes(bundle)
    root = resolve_markers_root()
    actions: List[str] = []
    below: List[Tuple[str, int, int]] = []

    for fam, names in includes.items():
        for name in names:
            dst = root / fam / f"{name}.yaml"
            if dst.exists():
                doc = read_yaml(dst)
            else:
                doc = {"id": name, "family": fam, "description": "", "notes": ""}
            before = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
            doc = ensure_examples_block(doc)
            pos = len(doc["examples"]["positive"])  # type: ignore[index]
            neg = len(doc["examples"]["negative"])  # type: ignore[index]
            if args.dry_run:
                if not dst.exists():
                    actions.append(f"create {dst}")
                if before != yaml.safe_dump(doc, allow_unicode=True, sort_keys=False):
                    actions.append(f"update {dst} (ensure examples block)")
            else:
                write_yaml(dst, doc)
            if not args.no_validate and (pos < args.min_pos or neg < args.min_neg):
                below.append((f"{fam}/{name}", pos, neg))

    if args.dry_run:
        for a in actions:
            print(a)

    if below and not args.no_validate:
        print("Markers below thresholds:")
        for m, p, n in below:
            print(f"  - {m}: pos={p} neg={n}")
        raise SystemExit(3)

    print("OK: seed/validate complete.")


if __name__ == "__main__":
    main()

