#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Derive dual-adapter training datasets from qwen3_strict mother dataset.

Outputs (default):
- data/qwen3_strict_train_ts_v1.json
- data/qwen3_strict_val_ts_v1.json
- data/qwen3_strict_train_if_v1.json
- data/qwen3_strict_val_if_v1.json
- data/qwen3_strict_dual_adapter_manifest_v1.json

This keeps the strict split policy intact by deriving from already-built
`qwen3_strict_train.json` / `qwen3_strict_val.json` only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_by(items: List[Dict], key: str) -> Dict[str, int]:
    c = Counter(str(x.get(key, "missing")) for x in items)
    return dict(c)


def summarize(items: List[Dict]) -> Dict:
    return {
        "count": len(items),
        "task_distribution": count_by(items, "task_type"),
        "source_distribution": count_by(items, "source"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build qwen3_strict dual-adapter datasets from strict mother train/val.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--train-file", type=str, default="qwen3_strict_train.json")
    parser.add_argument("--val-file", type=str, default="qwen3_strict_val.json")
    parser.add_argument("--version-tag", type=str, default="v1")
    parser.add_argument("--ts-name", type=str, default="qwen3_strict")
    parser.add_argument("--if-name", type=str, default="qwen3_strict")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    data_dir = base / args.data_dir
    train_path = data_dir / args.train_file
    val_path = data_dir / args.val_file

    train = json.loads(train_path.read_text(encoding="utf-8"))
    val = json.loads(val_path.read_text(encoding="utf-8"))

    if not isinstance(train, list) or not isinstance(val, list):
        raise ValueError("Strict mother train/val must be JSON lists")

    # Task routing from strict mother dataset
    ts_tasks = {"translation", "summarization"}
    if_task = "instruction_following"

    train_ts = [x for x in train if str(x.get("task_type")) in ts_tasks]
    val_ts = [x for x in val if str(x.get("task_type")) in ts_tasks]
    train_if = [x for x in train if str(x.get("task_type")) == if_task]
    val_if = [x for x in val if str(x.get("task_type")) == if_task]

    tag = args.version_tag
    out_train_ts = data_dir / f"{args.ts_name}_train_ts_{tag}.json"
    out_val_ts = data_dir / f"{args.ts_name}_val_ts_{tag}.json"
    out_train_if = data_dir / f"{args.if_name}_train_if_{tag}.json"
    out_val_if = data_dir / f"{args.if_name}_val_if_{tag}.json"
    out_manifest = data_dir / f"qwen3_strict_dual_adapter_manifest_{tag}.json"

    for p, payload in [
        (out_train_ts, train_ts),
        (out_val_ts, val_ts),
        (out_train_if, train_if),
        (out_val_if, val_if),
    ]:
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "series": "qwen3_strict_dual_adapter",
        "version_tag": tag,
        "parent": {
            "train_file": str(train_path),
            "val_file": str(val_path),
            "train_sha256": sha256_file(train_path),
            "val_sha256": sha256_file(val_path),
        },
        "routing_policy": {
            "ts_adapter_tasks": sorted(ts_tasks),
            "if_adapter_tasks": [if_task],
        },
        "outputs": {
            "train_ts": str(out_train_ts),
            "val_ts": str(out_val_ts),
            "train_if": str(out_train_if),
            "val_if": str(out_val_if),
        },
        "stats": {
            "parent_train": summarize(train),
            "parent_val": summarize(val),
            "train_ts": summarize(train_ts),
            "val_ts": summarize(val_ts),
            "train_if": summarize(train_if),
            "val_if": summarize(val_if),
        },
        "consistency_checks": {
            "train_partition_complete": len(train_ts) + len(train_if) == len(train),
            "val_partition_complete": len(val_ts) + len(val_if) == len(val),
            "train_overlap_count": sum(1 for x in train if str(x.get("task_type")) in ts_tasks and str(x.get("task_type")) == if_task),
            "val_overlap_count": sum(1 for x in val if str(x.get("task_type")) in ts_tasks and str(x.get("task_type")) == if_task),
        },
        "notes": [
            "Derived from qwen3_strict mother train/val only; strict test sets are unchanged.",
            "Use strict test_labeled/test_if_unlabeled for all downstream adapter evaluation.",
        ],
    }
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("qwen3_strict dual-adapter dataset derivation complete")
    print("=" * 72)
    print(f"train_ts: {out_train_ts} ({len(train_ts)})")
    print(f"val_ts:   {out_val_ts} ({len(val_ts)})")
    print(f"train_if: {out_train_if} ({len(train_if)})")
    print(f"val_if:   {out_val_if} ({len(val_if)})")
    print(f"manifest: {out_manifest}")


if __name__ == "__main__":
    main()

