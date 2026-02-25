#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate strict 3-seed evaluation results (mean/std) by version.

Reads:
- evaluation/strict_3seed/<version>_s<seed>_labeled/eval_results.json
- evaluation/strict_3seed/<version>_s<seed>_if_unlabeled/eval_results.json

Produces:
- evaluation/strict_3seed/aggregate_3seed_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def flatten_numeric(obj, prefix="") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_numeric(v, key))
    elif isinstance(obj, list):
        return out
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def mean_std(vals: Iterable[float]) -> Tuple[float, float]:
    xs = list(vals)
    if not xs:
        return (math.nan, math.nan)
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return (m, 0.0)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return (m, math.sqrt(var))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate qwen3 strict 3-seed eval results.")
    parser.add_argument("--eval-dir", type=str, default="evaluation/strict_3seed")
    parser.add_argument("--seeds", type=int, nargs="*", default=[2026, 2027, 2028])
    parser.add_argument("--versions", type=str, nargs="*", default=["v1a", "v1b", "v1c"])
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    eval_dir = base / args.eval_dir
    out_path = Path(args.output) if args.output else (eval_dir / "aggregate_3seed_summary.json")

    summary = {
        "eval_dir": str(eval_dir),
        "versions": args.versions,
        "seeds": args.seeds,
        "splits": {},
    }

    for split in ["labeled", "if_unlabeled"]:
        split_data = {}
        for v in args.versions:
            per_seed_flat: Dict[int, Dict[str, float]] = {}
            missing = []
            for s in args.seeds:
                p = eval_dir / f"{v}_s{s}_{split}" / "eval_results.json"
                if not p.exists():
                    missing.append(str(p))
                    continue
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    per_seed_flat[s] = flatten_numeric(obj)
                except Exception as e:
                    missing.append(f"{p} (parse_error={e})")

            keys = sorted(set().union(*(d.keys() for d in per_seed_flat.values()))) if per_seed_flat else []
            metrics = {}
            for k in keys:
                vals = [d[k] for d in per_seed_flat.values() if k in d]
                m, sd = mean_std(vals)
                metrics[k] = {
                    "n": len(vals),
                    "mean": m,
                    "std": sd,
                    "values": {str(seed): per_seed_flat[seed].get(k) for seed in sorted(per_seed_flat)},
                }

            split_data[v] = {
                "available_seeds": sorted(per_seed_flat.keys()),
                "missing": missing,
                "metrics": metrics,
            }
        summary["splits"][split] = split_data

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved 3-seed aggregate summary: {out_path}")


if __name__ == "__main__":
    main()

