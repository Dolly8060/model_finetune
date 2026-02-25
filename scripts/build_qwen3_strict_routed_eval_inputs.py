#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split qwen3_strict labeled eval set into routed subsets for Plan A.

Route policy:
- TS adapter: translation + summarization
- IF adapter: instruction_following

Outputs:
- evaluation/output_data/planA_routing/qwen3_strict_test_labeled_ts.json
- evaluation/output_data/planA_routing/qwen3_strict_test_labeled_if.json
- evaluation/output_data/planA_routing/qwen3_strict_test_labeled_route_manifest.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def dedup_key(sample: Dict) -> str:
    return f"{(sample.get('instruction') or '')[:240]}|||{(sample.get('input') or '')[:120]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build routed eval subsets for qwen3_strict labeled test.")
    parser.add_argument("--input-file", type=str, default="data/qwen3_strict_test_labeled.json")
    parser.add_argument("--output-dir", type=str, default="evaluation/output_data/planA_routing")
    parser.add_argument("--ts-name", type=str, default="qwen3_strict_test_labeled_ts.json")
    parser.add_argument("--if-name", type=str, default="qwen3_strict_test_labeled_if.json")
    parser.add_argument("--manifest-name", type=str, default="qwen3_strict_test_labeled_route_manifest.json")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    in_path = base / args.input_file
    out_dir = base / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input eval file must be a JSON list")

    ts_tasks = {"translation", "summarization"}
    ts: List[Dict] = []
    iff: List[Dict] = []
    other: List[Dict] = []
    seen = set()
    duplicates = 0

    for idx, s in enumerate(data):
        if not isinstance(s, dict):
            continue
        x = dict(s)
        x["_route_index"] = idx
        x["_route_key"] = dedup_key(x)
        if x["_route_key"] in seen:
            duplicates += 1
        seen.add(x["_route_key"])
        task = str(x.get("task_type", "unknown"))
        if task in ts_tasks:
            ts.append(x)
        elif task == "instruction_following":
            iff.append(x)
        else:
            other.append(x)

    out_ts = out_dir / args.ts_name
    out_if = out_dir / args.if_name
    out_manifest = out_dir / args.manifest_name

    out_ts.write_text(json.dumps(ts, ensure_ascii=False, indent=2), encoding="utf-8")
    out_if.write_text(json.dumps(iff, ensure_ascii=False, indent=2), encoding="utf-8")
    out_manifest.write_text(
        json.dumps(
            {
                "input_file": str(in_path),
                "total": len(data),
                "routed_ts": len(ts),
                "routed_if": len(iff),
                "other_unrouted": len(other),
                "duplicates_by_route_key": duplicates,
                "task_distribution_total": dict(Counter(str(x.get("task_type", "unknown")) for x in data if isinstance(x, dict))),
                "task_distribution_ts": dict(Counter(str(x.get("task_type", "unknown")) for x in ts)),
                "task_distribution_if": dict(Counter(str(x.get("task_type", "unknown")) for x in iff)),
                "outputs": {
                    "ts": str(out_ts),
                    "if": str(out_if),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("Plan A routed eval inputs built")
    print("=" * 72)
    print(f"ts: {out_ts} ({len(ts)})")
    print(f"if: {out_if} ({len(iff)})")
    print(f"manifest: {out_manifest}")


if __name__ == "__main__":
    main()

