#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert LRIA-Follow Excel files into strict IF labeled test JSON.

Outputs:
- data/qwen3_strict_internal_if_labeled.json
- data/qwen3_strict_internal_if_manifest.json

This script intentionally keeps rows with empty `gt` out of the labeled set
and records them in the manifest so auditors can see what was excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _norm_text(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def _load_sheet(path: Path, source_label: str, language: str) -> Tuple[List[Dict], Dict]:
    df = pd.read_excel(path)
    rows: List[Dict] = []
    stats = {
        "rows_total": int(len(df)),
        "rows_missing_prompt": 0,
        "rows_missing_gt": 0,
        "rows_kept_labeled": 0,
    }

    for idx, rec in enumerate(df.to_dict(orient="records")):
        prompt = _norm_text(rec.get("Prompt"))
        gt = _norm_text(rec.get("gt"))
        if not prompt:
            stats["rows_missing_prompt"] += 1
            continue
        if not gt:
            stats["rows_missing_gt"] += 1
            continue

        row = {
            "instruction": prompt,
            "input": "",
            "output": gt,
            "task_type": "instruction_following",
            "source": source_label,
            "language": language,
            "meta": {
                "id": rec.get("id"),
                "L1": _norm_text(rec.get("L1")),
                "L2": _norm_text(rec.get("L2")),
                "judge_method": _norm_text(rec.get("judge_method")),
                "new_add": _norm_text(rec.get("new_add")),
                "note": _norm_text(rec.get("Note")),
            },
            "_src_file": str(path),
            "_src_index": idx,
            "_src_loader": "lria_follow_xlsx",
            "_src_source_name": source_label,
            "_record_uid": f"lria_follow|{path.name}|{idx}",
        }
        rows.append(row)

    stats["rows_kept_labeled"] = len(rows)
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LRIA-Follow xlsx files to strict labeled IF json.")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--output-name", type=str, default="qwen3_strict_internal_if_labeled.json")
    parser.add_argument("--manifest-name", type=str, default="qwen3_strict_internal_if_manifest.json")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    dataset_dir = base / args.dataset_dir
    output_dir = base / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    en_path = dataset_dir / "LRIA-Follow_EN" / "LRIA-Follow_v3_EN.xlsx"
    zh_path = dataset_dir / "LRIA-Follow_ZH" / "LRIA-Follow_v3_ZH.xlsx"

    rows: List[Dict] = []
    sources = []
    file_stats = {}

    if en_path.exists():
        en_rows, en_stats = _load_sheet(en_path, "lria_follow_en", "en")
        rows.extend(en_rows)
        sources.append(str(en_path))
        file_stats[str(en_path)] = en_stats
    if zh_path.exists():
        zh_rows, zh_stats = _load_sheet(zh_path, "lria_follow_zh", "zh")
        rows.extend(zh_rows)
        sources.append(str(zh_path))
        file_stats[str(zh_path)] = zh_stats

    out_path = output_dir / args.output_name
    out_manifest = output_dir / args.manifest_name

    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "output_file": str(out_path),
        "count": len(rows),
        "source_distribution": {
            "lria_follow_en": sum(1 for r in rows if r.get("source") == "lria_follow_en"),
            "lria_follow_zh": sum(1 for r in rows if r.get("source") == "lria_follow_zh"),
        },
        "language_distribution": {
            "en": sum(1 for r in rows if r.get("language") == "en"),
            "zh": sum(1 for r in rows if r.get("language") == "zh"),
        },
        "sources": sources,
        "file_stats": file_stats,
    }
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("LRIA-Follow strict IF labeled conversion complete")
    print("=" * 72)
    print(f"output:    {out_path} ({len(rows)})")
    print(f"manifest:  {out_manifest}")


if __name__ == "__main__":
    main()

