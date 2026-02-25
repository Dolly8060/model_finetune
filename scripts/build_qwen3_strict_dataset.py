#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build audit-friendly Qwen3 strict datasets (new series: v1a/v1b/v1c).

Runtime direct outputs:
- data/qwen3_strict_train.json            # v1a training dataset
- data/qwen3_strict_val.json
- data/qwen3_strict_test_labeled.json
- data/qwen3_strict_test_if_unlabeled.json
- data/qwen3_strict_train_v1b.json        # IF-priority train variant (old v2b direction)
- data/qwen3_strict_train_v1c.json        # default same data as v1b (old v2c direction; LR differs in config)
- data/qwen3_strict_manifest.json

This script is designed for an audit-grade rebuild:
- clear direct inputs
- leakage checks
- lineage/hash audit artifacts (via --audit-mode)

Public translation/summarization data is expected as metadata-preserving json list
files (see source_registry template and build docs).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from build_qwen3_rigorous_dataset import (  # type: ignore
    add_think_prefix,
    build_lineage_records,
    count_by_source,
    count_by_source_file,
    count_by_task,
    dedup_key,
    dedup_samples,
    infer_task_type,
    infer_translation_direction,
    load_ifeval_jsonl,
    load_json_list,
    load_mifeval_dir,
    sha256_file,
    split_val_then_train,
)


def sample_take(pool: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    if n <= 0 or not pool:
        return []
    x = list(pool)
    rng.shuffle(x)
    return x[: min(n, len(x))]


def sample_take_with_replacement(pool: List[Dict], n: int, rng: random.Random) -> Tuple[List[Dict], int]:
    """Sample n items, with replacement if pool is too small.

    Returns (samples, duplicate_extra_count).
    """
    if n <= 0 or not pool:
        return [], 0
    if len(pool) >= n:
        x = list(pool)
        rng.shuffle(x)
        return x[:n], 0

    out = list(pool)
    duplicate_extra = n - len(pool)
    out.extend(rng.choice(pool) for _ in range(duplicate_extra))
    rng.shuffle(out)
    return out, duplicate_extra


def format_sample(sample: Dict, think_prefix: bool) -> Dict:
    out = {
        "instruction": sample.get("instruction", ""),
        "input": sample.get("input", ""),
        "output": add_think_prefix(sample.get("output", ""), think_prefix),
        "task_type": sample.get("task_type", "unknown"),
        "source": sample.get("source", "unknown"),
    }
    if "direction" in sample:
        out["direction"] = sample.get("direction")
    if "language" in sample:
        out["language"] = sample.get("language")
    if "meta" in sample and isinstance(sample.get("meta"), dict):
        out["meta"] = sample.get("meta")
    return out


def format_test_sample(sample: Dict) -> Dict:
    return format_sample(sample, think_prefix=False)


def classify_and_route_for_labeled_test(samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    test_trans_en2zh: List[Dict] = []
    test_trans_zh2en: List[Dict] = []
    test_summ: List[Dict] = []
    test_if_labeled: List[Dict] = []

    for s in samples:
        task = infer_task_type(s)
        s["task_type"] = task
        has_output = bool((s.get("output") or "").strip())
        if not has_output:
            continue
        if task == "translation":
            direction = infer_translation_direction(s)
            s["direction"] = direction
            if direction == "en2zh":
                test_trans_en2zh.append(s)
            else:
                test_trans_zh2en.append(s)
        elif task == "summarization":
            test_summ.append(s)
        elif task == "instruction_following":
            test_if_labeled.append(s)

    return (
        dedup_samples(test_trans_en2zh),
        dedup_samples(test_trans_zh2en),
        dedup_samples(test_summ),
        dedup_samples(test_if_labeled),
    )


def build_if_priority_variant(
    train_base: List[Dict],
    total_target: int,
    rng: random.Random,
    if_ratio: float = 0.50,
    trans_ratio: float = 0.25,
    summ_ratio: float = 0.25,
) -> Tuple[List[Dict], Dict]:
    # Partition base train
    trans_en2zh = [x for x in train_base if x.get("task_type") == "translation" and x.get("direction") == "en2zh"]
    trans_zh2en = [x for x in train_base if x.get("task_type") == "translation" and x.get("direction") == "zh2en"]
    summ = [x for x in train_base if x.get("task_type") == "summarization"]
    if_set = [x for x in train_base if x.get("task_type") == "instruction_following"]

    if_n = int(total_target * if_ratio)
    trans_n = int(total_target * trans_ratio)
    summ_n = total_target - if_n - trans_n
    trans_e2z_n = trans_n // 2
    trans_z2e_n = trans_n - trans_e2z_n

    out_if, dup_if = sample_take_with_replacement(if_set, if_n, rng)
    out_e2z, dup_e2z = sample_take_with_replacement(trans_en2zh, trans_e2z_n, rng)
    out_z2e, dup_z2e = sample_take_with_replacement(trans_zh2en, trans_z2e_n, rng)
    out_summ, dup_summ = sample_take_with_replacement(summ, summ_n, rng)

    combined = out_if + out_e2z + out_z2e + out_summ
    rng.shuffle(combined)

    # Duplicate stats based on dedup_key in selected variant
    key_counts = Counter(dedup_key(x) for x in combined)
    duplicate_extra_total = sum(v - 1 for v in key_counts.values() if v > 1)
    duplicate_groups = sum(1 for v in key_counts.values() if v > 1)

    stats = {
        "target_total": total_target,
        "target_if": if_n,
        "target_translation": trans_n,
        "target_summarization": summ_n,
        "selected": len(combined),
        "duplicates_by_pool_shortage": {
            "instruction_following": dup_if,
            "translation_en2zh": dup_e2z,
            "translation_zh2en": dup_z2e,
            "summarization": dup_summ,
        },
        "duplicates_observed_in_variant": {
            "duplicate_extra_instances": duplicate_extra_total,
            "duplicate_groups": duplicate_groups,
        },
        "task_distribution": count_by_task(combined),
        "source_distribution": count_by_source(combined),
    }
    return combined, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build qwen3_strict datasets with audit-friendly lineage.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--output-prefix", type=str, default="qwen3_strict")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-target", type=int, default=3600)
    parser.add_argument("--val-target", type=int, default=360)
    parser.add_argument("--test-target", type=int, default=600)
    parser.add_argument("--if-unlabeled-target", type=int, default=1200)
    parser.add_argument("--no-think-prefix", action="store_true")
    parser.add_argument("--audit-mode", action="store_true")
    parser.add_argument("--audit-dir", type=str, default="")
    parser.add_argument("--source-registry", type=str, default="data/source_registry_qwen3_strict.template.json")

    # Direct atomic/approved sources (expected)
    parser.add_argument("--self-train", type=str, default="data/train.json")
    parser.add_argument("--self-val", type=str, default="data/val.json")
    parser.add_argument("--argilla-ifeval", type=str, default="data/argilla_ifeval.json")
    parser.add_argument("--chinese-ifeval", type=str, default="data/ifeval_full_with_meta.json")
    parser.add_argument("--public-train-meta", type=str, default="data/public_train_strict_with_meta.json")
    parser.add_argument("--public-val-meta", type=str, default="data/public_val_strict_with_meta.json")
    parser.add_argument("--public-test-meta", type=str, default="data/public_test_strict_with_meta.json")
    parser.add_argument("--internal-if-labeled", type=str, default="data/qwen3_strict_internal_if_labeled.json")

    # Optional controls
    parser.add_argument("--exclude-chinese-ifeval-train", action="store_true")
    parser.add_argument("--include-public-val-in-train", action="store_true")
    parser.add_argument("--v1c-copy-v1b", action="store_true", default=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base = Path(__file__).resolve().parents[1]
    data_dir = base / args.data_dir
    dataset_dir = base / args.dataset_dir
    think_prefix = not args.no_think_prefix

    # ----------------------------
    # 0) Direct inputs
    # ----------------------------
    train_sources: List[Tuple[Path, str]] = [
        (base / args.self_train, "self_generated"),
        (base / args.self_val, "self_generated"),
        (base / args.argilla_ifeval, "argilla_ifeval"),
        (base / args.public_train_meta, "public_train_meta"),
    ]
    if not args.exclude_chinese_ifeval_train:
        train_sources.append((base / args.chinese_ifeval, "chinese_generated"))
    if args.include_public_val_in_train:
        train_sources.append((base / args.public_val_meta, "public_val_meta_train"))

    eval_labeled_sources: List[Tuple[Path, str]] = [
        (base / args.internal_if_labeled, "internal_if_labeled"),
        (base / args.public_test_meta, "public_test_meta"),
    ]

    # ----------------------------
    # 1) Build labeled test pools
    # ----------------------------
    raw_eval_labeled: List[Dict] = []
    for path, source_label in eval_labeled_sources:
        raw_eval_labeled.extend(load_json_list(path, source_label))

    test_trans_en2zh, test_trans_zh2en, test_summ, test_if_labeled = classify_and_route_for_labeled_test(raw_eval_labeled)

    # build prompt-only IF benchmark from internal dataset/ raw files
    test_if_unlabeled: List[Dict] = []
    test_if_unlabeled.extend(load_mifeval_dir(dataset_dir / "m-ifeval"))
    test_if_unlabeled.extend(load_ifeval_jsonl(dataset_dir / "IFEval" / "input_data.jsonl"))
    test_if_unlabeled = dedup_samples(test_if_unlabeled)

    test_per_task = max(args.test_target // 3, 1)
    test_trans_each = max(test_per_task // 2, 1)
    selected_test = []
    selected_test.extend(sample_take(test_trans_en2zh, test_trans_each, rng))
    selected_test.extend(sample_take(test_trans_zh2en, test_trans_each, rng))
    selected_test.extend(sample_take(test_summ, test_per_task, rng))
    selected_test.extend(sample_take(test_if_labeled, test_per_task, rng))
    selected_test = dedup_samples(selected_test)

    test_if_unlabeled = sample_take(test_if_unlabeled, args.if_unlabeled_target, rng)
    test_if_unlabeled = dedup_samples(test_if_unlabeled)

    test_keys = {dedup_key(x) for x in selected_test} | {dedup_key(x) for x in test_if_unlabeled}

    # ----------------------------
    # 2) Build train/val pools (atomic sources only)
    # ----------------------------
    train_trans_en2zh: List[Dict] = []
    train_trans_zh2en: List[Dict] = []
    train_summ: List[Dict] = []
    train_if: List[Dict] = []
    skipped = {
        "missing_output": 0,
        "task_other": 0,
        "test_overlap": 0,
        "missing_file_sources": [],
    }

    for path, source_label in train_sources:
        if not path.exists():
            skipped["missing_file_sources"].append(str(path))
            continue
        for s in load_json_list(path, source_label):
            if not (s.get("output") or "").strip():
                skipped["missing_output"] += 1
                continue
            if dedup_key(s) in test_keys:
                skipped["test_overlap"] += 1
                continue
            task = infer_task_type(s)
            s["task_type"] = task
            if task == "translation":
                direction = infer_translation_direction(s)
                s["direction"] = direction
                if direction == "en2zh":
                    train_trans_en2zh.append(s)
                else:
                    train_trans_zh2en.append(s)
            elif task == "summarization":
                train_summ.append(s)
            elif task == "instruction_following":
                train_if.append(s)
            else:
                skipped["task_other"] += 1

    train_trans_en2zh = dedup_samples(train_trans_en2zh)
    train_trans_zh2en = dedup_samples(train_trans_zh2en)
    train_summ = dedup_samples(train_summ)
    train_if = dedup_samples(train_if)

    train_per_task = max(args.train_target // 3, 1)
    val_per_task = max(args.val_target // 3, 1)
    train_trans_each = max(train_per_task // 2, 1)
    val_trans_each = max(val_per_task // 2, 1)

    tr_e2z, va_e2z = split_val_then_train(train_trans_en2zh, val_trans_each, train_trans_each, rng)
    tr_z2e, va_z2e = split_val_then_train(train_trans_zh2en, val_trans_each, train_trans_each, rng)
    tr_sum, va_sum = split_val_then_train(train_summ, val_per_task, train_per_task, rng)
    tr_if, va_if = split_val_then_train(train_if, val_per_task, train_per_task, rng)

    train_set = dedup_samples(tr_e2z + tr_z2e + tr_sum + tr_if)
    val_set = dedup_samples(va_e2z + va_z2e + va_sum + va_if)
    val_keys = {dedup_key(x) for x in val_set}
    train_set = [x for x in train_set if dedup_key(x) not in val_keys]

    # ----------------------------
    # 3) Build v1b/v1c variants (IF-priority)
    # ----------------------------
    variant_seed = args.seed + 101
    v_rng = random.Random(variant_seed)
    train_v1b_set, train_v1b_stats = build_if_priority_variant(train_set, len(train_set), v_rng)

    if args.v1c_copy_v1b:
        train_v1c_set = list(train_v1b_set)
        train_v1c_stats = {
            "copied_from": "v1b",
            "seed": variant_seed,
            "task_distribution": count_by_task(train_v1c_set),
            "source_distribution": count_by_source(train_v1c_set),
            "selected": len(train_v1c_set),
        }
    else:
        train_v1c_set, train_v1c_stats = build_if_priority_variant(train_set, len(train_set), random.Random(args.seed + 202))

    # ----------------------------
    # 4) Format and save
    # ----------------------------
    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(selected_test)
    rng.shuffle(test_if_unlabeled)
    random.Random(args.seed + 303).shuffle(train_v1b_set)
    random.Random(args.seed + 404).shuffle(train_v1c_set)

    train_out = [format_sample(x, think_prefix) for x in train_set]
    val_out = [format_sample(x, think_prefix) for x in val_set]
    test_labeled_out = [format_test_sample(x) for x in selected_test]
    test_if_unlabeled_out = [format_test_sample(x) for x in test_if_unlabeled]
    train_v1b_out = [format_sample(x, think_prefix) for x in train_v1b_set]
    train_v1c_out = [format_sample(x, think_prefix) for x in train_v1c_set]

    out_train = data_dir / f"{args.output_prefix}_train.json"
    out_val = data_dir / f"{args.output_prefix}_val.json"
    out_test = data_dir / f"{args.output_prefix}_test_labeled.json"
    out_if_unlabeled = data_dir / f"{args.output_prefix}_test_if_unlabeled.json"
    out_train_v1b = data_dir / f"{args.output_prefix}_train_v1b.json"
    out_train_v1c = data_dir / f"{args.output_prefix}_train_v1c.json"
    out_manifest = data_dir / f"{args.output_prefix}_manifest.json"

    for p, payload in [
        (out_train, train_out),
        (out_val, val_out),
        (out_test, test_labeled_out),
        (out_if_unlabeled, test_if_unlabeled_out),
        (out_train_v1b, train_v1b_out),
        (out_train_v1c, train_v1c_out),
    ]:
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Leakage checks
    k_train = {dedup_key(x) for x in train_out}
    k_val = {dedup_key(x) for x in val_out}
    k_test = {dedup_key(x) for x in test_labeled_out}
    k_if = {dedup_key(x) for x in test_if_unlabeled_out}
    k_v1b = [dedup_key(x) for x in train_v1b_out]
    k_v1c = [dedup_key(x) for x in train_v1c_out]

    leakage = {
        "train_val": len(k_train & k_val),
        "train_test_labeled": len(k_train & k_test),
        "val_test_labeled": len(k_val & k_test),
        "train_test_if_unlabeled": len(k_train & k_if),
        "val_test_if_unlabeled": len(k_val & k_if),
        "test_labeled_vs_test_if_unlabeled": len(k_test & k_if),
        "train_v1b_vs_val": len(set(k_v1b) & k_val),
        "train_v1c_vs_val": len(set(k_v1c) & k_val),
        "train_v1b_vs_test_labeled": len(set(k_v1b) & k_test),
        "train_v1c_vs_test_labeled": len(set(k_v1c) & k_test),
    }

    manifest = {
        "series": "qwen3_strict",
        "version_mapping": {
            "v1a": "balanced strict training set (old v1 direction)",
            "v1b": "IF-priority strict training set (old v2b direction)",
            "v1c": "IF-priority strict training set + lower LR config (old v2c direction)",
        },
        "args": vars(args),
        "counts": {
            "train_v1a": len(train_out),
            "val": len(val_out),
            "test_labeled": len(test_labeled_out),
            "test_if_unlabeled": len(test_if_unlabeled_out),
            "train_v1b": len(train_v1b_out),
            "train_v1c": len(train_v1c_out),
        },
        "task_distribution": {
            "train_v1a": count_by_task(train_out),
            "train_v1b": count_by_task(train_v1b_out),
            "train_v1c": count_by_task(train_v1c_out),
            "val": count_by_task(val_out),
            "test_labeled": count_by_task(test_labeled_out),
            "test_if_unlabeled": count_by_task(test_if_unlabeled_out),
        },
        "source_distribution": {
            "train_v1a": count_by_source(train_out),
            "train_v1b": count_by_source(train_v1b_out),
            "train_v1c": count_by_source(train_v1c_out),
            "val": count_by_source(val_out),
            "test_labeled": count_by_source(test_labeled_out),
            "test_if_unlabeled": count_by_source(test_if_unlabeled_out),
        },
        "pool_stats": {
            "skipped": skipped,
            "train_pool_translation_en2zh": len(train_trans_en2zh),
            "train_pool_translation_zh2en": len(train_trans_zh2en),
            "train_pool_summarization": len(train_summ),
            "train_pool_instruction_following": len(train_if),
            "labeled_test_pool_translation_en2zh": len(test_trans_en2zh),
            "labeled_test_pool_translation_zh2en": len(test_trans_zh2en),
            "labeled_test_pool_summarization": len(test_summ),
            "labeled_test_pool_instruction_following": len(test_if_labeled),
        },
        "variant_stats": {
            "v1b": train_v1b_stats,
            "v1c": train_v1c_stats,
        },
        "leakage": leakage,
        "direct_input_files": [str(p) for p, _ in train_sources + eval_labeled_sources],
        "notes": [
            "train.json and val.json are user-confirmed API-generated sources.",
            "dataset/IFEval and dataset/m-ifeval are used for prompt-only IF benchmark.",
            "dataset/LRIA-Follow_* is used via converted internal_if_labeled file.",
            "public_*_strict_with_meta.json files are expected metadata-preserving public datasets.",
        ],
    }
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ----------------------------
    # 5) Audit artifacts
    # ----------------------------
    if args.audit_mode:
        audit_dir = Path(args.audit_dir) if args.audit_dir else (data_dir / "audit")
        audit_dir.mkdir(parents=True, exist_ok=True)
        lineage_path = audit_dir / f"{args.output_prefix}_lineage.jsonl"
        snapshot_path = audit_dir / f"{args.output_prefix}_source_snapshot.json"
        hashes_path = audit_dir / f"{args.output_prefix}_hashes.json"

        lineage_records: List[Dict] = []
        lineage_records.extend(build_lineage_records(train_set, "train_v1a"))
        lineage_records.extend(build_lineage_records(val_set, "val"))
        lineage_records.extend(build_lineage_records(selected_test, "test_labeled"))
        lineage_records.extend(build_lineage_records(test_if_unlabeled, "test_if_unlabeled"))
        lineage_records.extend(build_lineage_records(train_v1b_set, "train_v1b"))
        lineage_records.extend(build_lineage_records(train_v1c_set, "train_v1c"))
        with lineage_path.open("w", encoding="utf-8") as f:
            for rec in lineage_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        raw_if_sources = [dataset_dir / "m-ifeval", dataset_dir / "IFEval" / "input_data.jsonl"]
        direct_paths: List[Path] = []
        for p, _ in train_sources + eval_labeled_sources:
            if p.exists():
                direct_paths.append(p)
        for src in raw_if_sources:
            if src.is_dir():
                direct_paths.extend(sorted(x for x in src.glob("*") if x.is_file()))
            elif src.exists():
                direct_paths.append(src)

        registry_path = base / args.source_registry
        if registry_path.exists():
            direct_paths.append(registry_path)

        uniq: List[Path] = []
        seen: Set[str] = set()
        for p in direct_paths:
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            uniq.append(p)

        snapshot = {
            "output_prefix": args.output_prefix,
            "source_registry_path": str(registry_path),
            "confirmed_user_generated": ["data/train.json", "data/val.json"],
            "direct_input_files": [{"path": str(p), "size_bytes": p.stat().st_size} for p in uniq if p.exists()],
            "final_split_source_labels": {
                "train_v1a": count_by_source(train_out),
                "train_v1b": count_by_source(train_v1b_out),
                "train_v1c": count_by_source(train_v1c_out),
                "val": count_by_source(val_out),
                "test_labeled": count_by_source(test_labeled_out),
                "test_if_unlabeled": count_by_source(test_if_unlabeled_out),
            },
            "final_split_source_files": {
                "train_v1a": count_by_source_file(train_set),
                "train_v1b": count_by_source_file(train_v1b_set),
                "train_v1c": count_by_source_file(train_v1c_set),
                "val": count_by_source_file(val_set),
                "test_labeled": count_by_source_file(selected_test),
                "test_if_unlabeled": count_by_source_file(test_if_unlabeled),
            },
            "lineage_record_count": len(lineage_records),
        }
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

        hash_entries = {}
        files_to_hash = uniq + [
            out_train,
            out_val,
            out_test,
            out_if_unlabeled,
            out_train_v1b,
            out_train_v1c,
            out_manifest,
            Path(__file__).resolve(),
        ]
        for p in files_to_hash:
            if p.exists() and p.is_file():
                hash_entries[str(p)] = {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
        hashes_path.write_text(
            json.dumps({"output_prefix": args.output_prefix, "files": hash_entries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("=" * 72)
    print("qwen3_strict dataset build complete")
    print("=" * 72)
    print(f"v1a train: {out_train} ({len(train_out)})")
    print(f"val:       {out_val} ({len(val_out)})")
    print(f"test:      {out_test} ({len(test_labeled_out)})")
    print(f"test IF:   {out_if_unlabeled} ({len(test_if_unlabeled_out)})")
    print(f"v1b train: {out_train_v1b} ({len(train_v1b_out)})")
    print(f"v1c train: {out_train_v1c} ({len(train_v1c_out)})")
    print(f"manifest:  {out_manifest}")
    if args.audit_mode:
        print(f"audit dir: {audit_dir}")


if __name__ == "__main__":
    main()

