#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split metadata-preserving public data into strict train/val/test source files.

Expected input:
- data/public_eval_with_meta.json (from scripts/download_public_datasets.py)

Outputs:
- data/public_train_strict_with_meta.json
- data/public_val_strict_with_meta.json
- data/public_test_strict_with_meta.json
- data/public_strict_split_manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def dedup_key(sample: Dict) -> str:
    return f"{(sample.get('instruction') or '')[:240]}|||{(sample.get('input') or '')[:120]}"


def looks_chinese(text: str) -> bool:
    text = text or ""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    zh = sum(1 for c in chars if "\u4e00" <= c <= "\u9fff")
    return zh / len(chars) >= 0.25


def infer_direction(sample: Dict) -> str:
    ins = (sample.get("instruction") or "").lower()
    if "to chinese" in ins or "into chinese" in ins or "英译中" in ins:
        return "en2zh"
    if "to english" in ins or "into english" in ins or "中译英" in ins:
        return "zh2en"
    return "zh2en" if looks_chinese(sample.get("input", "")) else "en2zh"


def strat_key(sample: Dict) -> str:
    task = (sample.get("task_type") or "unknown").lower()
    if task == "translation":
        return f"translation:{infer_direction(sample)}"
    return task


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare public strict source splits from metadata-preserving public data.")
    parser.add_argument("--input", type=str, default="data/public_eval_with_meta.json")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    base = Path(__file__).resolve().parents[1]
    in_path = base / args.input
    out_dir = base / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Input file must be a JSON list")

    # Filter malformed and de-dup
    clean: List[Dict] = []
    seen = set()
    malformed = 0
    for idx, s in enumerate(raw):
        if not isinstance(s, dict):
            malformed += 1
            continue
        if not (s.get("instruction") and s.get("output")):
            malformed += 1
            continue
        x = dict(s)
        x.setdefault("source", "public_unknown")
        x.setdefault("_src_file", str(in_path))
        x.setdefault("_src_index", idx)
        x.setdefault("_src_loader", "public_meta_json")
        x.setdefault("_src_source_name", x.get("source", "public_unknown"))
        x.setdefault("_record_uid", f"public_meta|{in_path.name}|{idx}")
        k = dedup_key(x)
        if k in seen:
            continue
        seen.add(k)
        clean.append(x)

    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for s in clean:
        buckets[strat_key(s)].append(s)

    rng = random.Random(args.seed)
    train: List[Dict] = []
    val: List[Dict] = []
    test: List[Dict] = []

    for key, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        n_test = n - n_train - n_val
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:n_train + n_val + n_test])

    # Final shuffle for each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    out_train = out_dir / "public_train_strict_with_meta.json"
    out_val = out_dir / "public_val_strict_with_meta.json"
    out_test = out_dir / "public_test_strict_with_meta.json"
    out_manifest = out_dir / "public_strict_split_manifest.json"

    out_train.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    out_val.write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")
    out_test.write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")

    def task_dist(items: List[Dict]) -> Dict[str, int]:
        return dict(Counter((x.get("task_type") or "unknown") for x in items))

    def strat_dist(items: List[Dict]) -> Dict[str, int]:
        return dict(Counter(strat_key(x) for x in items))

    manifest = {
        "input": str(in_path),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "counts": {
            "input_raw": len(raw),
            "input_clean_dedup": len(clean),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "malformed_or_missing_required": malformed,
        },
        "task_distribution": {
            "train": task_dist(train),
            "val": task_dist(val),
            "test": task_dist(test),
        },
        "strat_distribution": {
            "train": strat_dist(train),
            "val": strat_dist(val),
            "test": strat_dist(test),
        },
        "outputs": {
            "train": str(out_train),
            "val": str(out_val),
            "test": str(out_test),
        },
    }
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Public strict source split complete")
    print("=" * 72)
    print(f"train: {out_train} ({len(train)})")
    print(f"val:   {out_val} ({len(val)})")
    print(f"test:  {out_test} ({len(test)})")
    print(f"manifest: {out_manifest}")


if __name__ == "__main__":
    main()

