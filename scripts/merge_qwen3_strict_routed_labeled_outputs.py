#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge routed Plan A labeled generation outputs back into strict labeled order.

Inputs:
- original strict labeled eval file (list)
- TS adapter generate.py output (json with top-level `samples`)
- IF adapter generate.py output (json with top-level `samples`)

Output:
- generate.py-compatible merged output json (top-level `samples`) for score.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def route_key(instruction: str, input_text: str) -> str:
    return f"{(instruction or '')[:240]}|||{(input_text or '')[:120]}"


def load_generate_output(path: Path) -> Dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "samples" not in obj or not isinstance(obj["samples"], list):
        raise ValueError(f"Invalid generate output format: {path}")
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge routed TS/IF labeled outputs into one generate-style file.")
    parser.add_argument("--original-eval-file", type=str, required=True)
    parser.add_argument("--ts-output", type=str, required=True)
    parser.add_argument("--if-output", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="PlanA_Routed")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--adapter-path", type=str, default="TS+IF routed")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    original_path = base / args.original_eval_file
    ts_path = base / args.ts_output
    if_path = base / args.if_output
    out_path = base / args.output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    original = json.loads(original_path.read_text(encoding="utf-8"))
    if not isinstance(original, list):
        raise ValueError("Original eval file must be a JSON list")

    ts_obj = load_generate_output(ts_path)
    if_obj = load_generate_output(if_path)
    ts_samples = ts_obj["samples"]
    if_samples = if_obj["samples"]

    # Build keyed predictions from routed outputs
    pred_by_key: Dict[str, Dict] = {}
    duplicate_keys = []
    source_counts = Counter()
    for routed_name, samples in [("ts", ts_samples), ("if", if_samples)]:
        for s in samples:
            if not isinstance(s, dict):
                continue
            k = route_key(s.get("instruction", ""), s.get("input", ""))
            if k in pred_by_key:
                duplicate_keys.append(k)
            pred_by_key[k] = {
                "prediction": s.get("prediction", ""),
                "task_type": s.get("task_type", ""),
                "source": s.get("source", ""),
                "language": s.get("language", ""),
                "reference": s.get("reference", ""),
                "_routed_from": routed_name,
            }
            source_counts[routed_name] += 1

    merged_samples: List[Dict] = []
    missing = []
    routed_from_counts = Counter()

    for s in original:
        if not isinstance(s, dict):
            continue
        ins = s.get("instruction", "")
        inp = s.get("input", "")
        ref = s.get("output", "")
        k = route_key(ins, inp)
        rec = pred_by_key.get(k)
        if rec is None:
            missing.append(k)
            pred = "[ERROR] missing routed prediction"
            routed_from = "missing"
        else:
            pred = rec.get("prediction", "")
            routed_from = rec.get("_routed_from", "unknown")

        merged_samples.append(
            {
                "instruction": ins,
                "input": inp,
                "reference": ref,
                "prediction": pred,
                "task_type": s.get("task_type", ""),
                "source": s.get("source", ""),
                "language": s.get("language", ""),
                "routed_from": routed_from,
            }
        )
        routed_from_counts[routed_from] += 1

    output = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "eval_file": str(original_path),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_samples": len(merged_samples),
        "samples": merged_samples,
        "routing_meta": {
            "ts_output": str(ts_path),
            "if_output": str(if_path),
            "routed_output_counts": dict(source_counts),
            "merged_routed_from_counts": dict(routed_from_counts),
            "missing_predictions": len(missing),
            "duplicate_routed_keys": len(duplicate_keys),
        },
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Merged Plan A routed labeled outputs")
    print("=" * 72)
    print(f"output: {out_path}")
    print(f"total_samples: {len(merged_samples)}")
    print(f"missing_predictions: {len(missing)}")
    print(f"routed_from: {dict(routed_from_counts)}")

    if missing:
        raise SystemExit("Missing routed predictions detected; merged output saved for debugging.")


if __name__ == "__main__":
    main()

