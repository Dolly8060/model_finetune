#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build rigorous Qwen3 SFT datasets with strict train/val/test isolation.

Outputs:
- data/qwen3_rigorous_train.json
- data/qwen3_rigorous_val.json
- data/qwen3_rigorous_test_labeled.json
- data/qwen3_rigorous_test_if_unlabeled.json
- data/qwen3_rigorous_manifest.json
"""

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

THINK_PREFIX = "<think>\\n\\n</think>\\n\\n"

TRANSLATION_KEYWORDS = [
    "translate", "translation", "english translation", "chinese translation",
    "to chinese", "to english", "into chinese", "into english",
    "\u7ffb\u8bd1", "\u4e2d\u8bd1\u82f1", "\u82f1\u8bd1\u4e2d", "\u4e2d\u6587\u7ffb\u8bd1",
    "\u8bd1\u4e3a", "\u8bd1\u6210",
]

SUMMARIZATION_KEYWORDS = [
    "summary", "summarize", "extract the main", "main points", "key points",
    "\u603b\u7ed3", "\u6458\u8981", "\u6982\u62ec", "\u63d0\u70bc", "\u5f52\u7eb3", "\u6838\u5fc3\u89c2\u70b9",
]

EN2ZH_HINTS = [
    "to chinese", "into chinese", "chinese translation",
    "\u82f1\u8bd1\u4e2d", "\u7ffb\u8bd1\u6210\u4e2d\u6587", "\u8bd1\u4e3a\u4e2d\u6587", "\u7ffb\u6210\u4e2d\u6587",
]
ZH2EN_HINTS = [
    "to english", "into english", "english translation",
    "\u4e2d\u8bd1\u82f1", "\u7ffb\u8bd1\u6210\u82f1\u6587", "\u8bd1\u4e3a\u82f1\u6587", "\u7ffb\u6210\u82f1\u6587",
]

IF_CONSTRAINT_PATTERNS = [
    r"at least\\s+\\d+\\s+(?:words?|sentences?|paragraphs?)",
    r"at most\\s+\\d+\\s+(?:words?|sentences?|paragraphs?)",
    r"no more than\\s+\\d+\\s+(?:words?|sentences?|paragraphs?)",
    r"exactly\\s+\\d+\\s+(?:words?|sentences?|paragraphs?)",
    r"must\\s+(?:include|contain)",
    r"do not\\s+(?:use|include)",
    r"avoid\\s+using",
    r"json format",
    r"markdown",
    r"bullet point",
    r"numbered list",
    r"\u81f3\u5c11\\d+",
    r"\u4e0d\u8d85\u8fc7\\d+",
    r"\u6070\u597d\\d+",
    r"\u5fc5\u987b\u5305\u542b",
    r"\u4e0d\u8981\u4f7f\u7528",
    r"\u5217\u8868\u5f62\u5f0f",
]


def dedup_key(sample: Dict, key_len: int = 240) -> str:
    return f"{(sample.get('instruction') or '')[:key_len]}|||{(sample.get('input') or '')[:120]}"


def looks_chinese(text: str, threshold: float = 0.25) -> bool:
    text = text or ""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    zh = sum(1 for c in chars if "\\u4e00" <= c <= "\\u9fff")
    return zh / max(len(chars), 1) >= threshold


def infer_task_type(sample: Dict) -> str:
    explicit = (sample.get("task_type") or "").strip().lower()
    if explicit in {"translation", "summarization", "instruction_following"}:
        return explicit

    instruction = (sample.get("instruction") or "").lower()
    if any(k in instruction for k in TRANSLATION_KEYWORDS):
        return "translation"
    if any(k in instruction for k in SUMMARIZATION_KEYWORDS):
        return "summarization"
    if any(re.search(p, instruction, re.IGNORECASE) for p in IF_CONSTRAINT_PATTERNS):
        return "instruction_following"

    return "other"


def infer_translation_direction(sample: Dict) -> str:
    instruction = (sample.get("instruction") or "").lower()
    if any(h in instruction for h in EN2ZH_HINTS):
        return "en2zh"
    if any(h in instruction for h in ZH2EN_HINTS):
        return "zh2en"

    src = sample.get("input") or ""
    if looks_chinese(src):
        return "zh2en"
    return "en2zh"


def load_json_list(path: Path, source_name: str) -> List[Dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    out = []
    for idx, s in enumerate(data):
        if not isinstance(s, dict):
            continue
        x = dict(s)
        x.setdefault("source", source_name)
        x.setdefault("_src_file", str(path))
        x.setdefault("_src_index", idx)
        x.setdefault("_src_loader", "json_list")
        x.setdefault("_src_source_name", source_name)
        x.setdefault("_record_uid", f"json_list|{path.name}|{idx}")
        out.append(x)
    return out


def load_mifeval_dir(path: Path) -> List[Dict]:
    out: List[Dict] = []
    if not path.exists():
        return out

    for file in sorted(path.glob("PMMEval-mifeval-*.json")):
        lang = file.stem.split("-")[-1]
        raw = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        for item_key, item in raw.items():
            prompt = ""
            op = item.get("origin_prompt")
            if isinstance(op, list) and op:
                p0 = op[0]
                if isinstance(p0, dict):
                    prompt = p0.get("prompt", "")
            elif isinstance(op, str):
                prompt = op
            if not prompt:
                continue
            out.append({
                "instruction": prompt,
                "input": "",
                "output": "",
                "task_type": "instruction_following",
                "source": f"mifeval_{lang}",
                "language": lang,
                "_src_file": str(file),
                "_src_index": item_key,
                "_src_loader": "mifeval_json",
                "_src_source_name": f"mifeval_{lang}",
                "_record_uid": f"mifeval|{file.name}|{item_key}",
            })
    return out


def load_ifeval_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = obj.get("prompt", "")
            if not prompt:
                continue
            out.append({
                "instruction": prompt,
                "input": "",
                "output": "",
                "task_type": "instruction_following",
                "source": "ifeval_prompt_only",
                "language": "en",
                "_src_file": str(path),
                "_src_index": line_no,
                "_src_loader": "ifeval_jsonl",
                "_src_source_name": "ifeval_prompt_only",
                "_record_uid": f"ifeval_jsonl|{path.name}|{line_no}",
            })
    return out


def dedup_samples(samples: List[Dict]) -> List[Dict]:
    seen: Set[str] = set()
    out: List[Dict] = []
    for s in samples:
        k = dedup_key(s)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def sample_take(pool: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    if n <= 0 or not pool:
        return []
    if len(pool) <= n:
        x = list(pool)
        rng.shuffle(x)
        return x
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    return [pool[i] for i in idx[:n]]


def split_val_then_train(pool: List[Dict], val_n: int, train_n: int, rng: random.Random) -> Tuple[List[Dict], List[Dict]]:
    x = list(pool)
    rng.shuffle(x)
    val = x[:min(val_n, len(x))]
    remain = x[len(val):]
    train = remain[:min(train_n, len(remain))]
    return train, val


def add_think_prefix(text: str, enabled: bool) -> str:
    text = text or ""
    if not enabled:
        return text
    if text.startswith("<think>"):
        return text
    return THINK_PREFIX + text


def count_by_task(samples: List[Dict]) -> Dict[str, int]:
    c = Counter((s.get("task_type") or "missing") for s in samples)
    return dict(c)


def count_by_source(samples: List[Dict]) -> Dict[str, int]:
    c = Counter((s.get("source") or "missing") for s in samples)
    return dict(c)


def count_by_source_file(samples: List[Dict]) -> Dict[str, int]:
    c = Counter((s.get("_src_file") or "missing") for s in samples)
    return dict(c)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_lineage_records(samples: List[Dict], split_name: str) -> List[Dict]:
    out: List[Dict] = []
    for pos, s in enumerate(samples):
        out.append({
            "split": split_name,
            "position": pos,
            "record_uid": s.get("_record_uid"),
            "source_label": s.get("source"),
            "source_file": s.get("_src_file"),
            "source_index": s.get("_src_index"),
            "source_loader": s.get("_src_loader"),
            "task_type": s.get("task_type"),
            "direction": s.get("direction"),
            "language": s.get("language"),
            "has_output": bool((s.get("output") or "").strip()),
            "dedup_key": dedup_key(s),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rigorous Qwen3 dataset splits.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--output-prefix", type=str, default="qwen3_rigorous")
    parser.add_argument("--train-target", type=int, default=3600)
    parser.add_argument("--val-target", type=int, default=360)
    parser.add_argument("--test-target", type=int, default=600)
    parser.add_argument("--if-unlabeled-target", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--include-public-in-train", action="store_true")
    parser.add_argument("--no-think-prefix", action="store_true")
    parser.add_argument("--audit-mode", action="store_true", help="Emit lineage/hash audit artifacts.")
    parser.add_argument("--audit-dir", type=str, default="", help="Directory for audit artifacts (default: data/audit).")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base = Path(__file__).resolve().parents[1]
    data_dir = base / args.data_dir
    dataset_dir = base / args.dataset_dir

    train_sources = [
        (data_dir / "train.json", "self_generated"),
        (data_dir / "val.json", "self_generated"),
        (data_dir / "train_v3.json", "v3_dataset"),
        (data_dir / "val_v3.json", "v3_dataset"),
        (data_dir / "train_mixed_3k.json", "mixed_3k"),
        (data_dir / "argilla_ifeval.json", "argilla_ifeval"),
        (data_dir / "ifeval_full_with_meta.json", "chinese_generated"),
    ]

    if args.include_public_in_train:
        train_sources.append((data_dir / "public_val_v2.json", "public_v2"))

    eval_labeled_sources = [
        (data_dir / "public_val_v2.json", "public_v2_eval"),
        (data_dir / "test_v4_enhanced.json", "test_v4_enhanced"),
    ]

    # ----------------------------
    # 1) Build labeled test pools
    # ----------------------------
    test_trans_en2zh: List[Dict] = []
    test_trans_zh2en: List[Dict] = []
    test_summ: List[Dict] = []
    test_if_labeled: List[Dict] = []
    test_if_unlabeled: List[Dict] = []

    for path, source_name in eval_labeled_sources:
        for s in load_json_list(path, source_name):
            task = infer_task_type(s)
            s["task_type"] = task
            has_output = bool((s.get("output") or "").strip())

            if task == "translation":
                if not has_output:
                    continue
                direction = infer_translation_direction(s)
                s["direction"] = direction
                if direction == "en2zh":
                    test_trans_en2zh.append(s)
                else:
                    test_trans_zh2en.append(s)
            elif task == "summarization":
                if has_output:
                    test_summ.append(s)
            elif task == "instruction_following":
                if has_output:
                    test_if_labeled.append(s)
                else:
                    test_if_unlabeled.append(s)

    # Add unlabeled IF from raw benchmark prompt sets
    test_if_unlabeled.extend(load_mifeval_dir(dataset_dir / "m-ifeval"))
    test_if_unlabeled.extend(load_ifeval_jsonl(dataset_dir / "IFEval" / "input_data.jsonl"))

    test_trans_en2zh = dedup_samples(test_trans_en2zh)
    test_trans_zh2en = dedup_samples(test_trans_zh2en)
    test_summ = dedup_samples(test_summ)
    test_if_labeled = dedup_samples(test_if_labeled)
    test_if_unlabeled = dedup_samples(test_if_unlabeled)

    # Balanced labeled test target: translation/summarization/IF ~= 1:1:1
    test_per_task = max(args.test_target // 3, 1)
    test_trans_each = max(test_per_task // 2, 1)

    selected_test = []
    selected_test.extend(sample_take(test_trans_en2zh, test_trans_each, rng))
    selected_test.extend(sample_take(test_trans_zh2en, test_trans_each, rng))
    selected_test.extend(sample_take(test_summ, test_per_task, rng))
    selected_test.extend(sample_take(test_if_labeled, test_per_task, rng))
    selected_test = dedup_samples(selected_test)

    # Sample unlabeled IF benchmark set
    test_if_unlabeled = sample_take(test_if_unlabeled, args.if_unlabeled_target, rng)
    test_if_unlabeled = dedup_samples(test_if_unlabeled)

    test_keys = {dedup_key(x) for x in selected_test} | {dedup_key(x) for x in test_if_unlabeled}

    # ----------------------------
    # 2) Build train/val pools
    # ----------------------------
    train_trans_en2zh: List[Dict] = []
    train_trans_zh2en: List[Dict] = []
    train_summ: List[Dict] = []
    train_if: List[Dict] = []

    skipped_no_output = 0
    skipped_other = 0
    skipped_test_overlap = 0

    for path, source_name in train_sources:
        for s in load_json_list(path, source_name):
            if not (s.get("output") or "").strip():
                skipped_no_output += 1
                continue
            if dedup_key(s) in test_keys:
                skipped_test_overlap += 1
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
                skipped_other += 1

    train_trans_en2zh = dedup_samples(train_trans_en2zh)
    train_trans_zh2en = dedup_samples(train_trans_zh2en)
    train_summ = dedup_samples(train_summ)
    train_if = dedup_samples(train_if)

    # Split targets
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
    # 3) Format and save
    # ----------------------------
    use_think_prefix = not args.no_think_prefix

    def format_train_eval(sample: Dict) -> Dict:
        out = {
            "instruction": sample.get("instruction", ""),
            "input": sample.get("input", ""),
            "output": add_think_prefix(sample.get("output", ""), use_think_prefix),
            "task_type": sample.get("task_type", "unknown"),
            "source": sample.get("source", "unknown"),
        }
        if "direction" in sample:
            out["direction"] = sample["direction"]
        if "language" in sample:
            out["language"] = sample["language"]
        return out

    def format_test(sample: Dict) -> Dict:
        out = {
            "instruction": sample.get("instruction", ""),
            "input": sample.get("input", ""),
            "output": sample.get("output", ""),
            "task_type": sample.get("task_type", "unknown"),
            "source": sample.get("source", "unknown"),
        }
        if "direction" in sample:
            out["direction"] = sample["direction"]
        if "language" in sample:
            out["language"] = sample["language"]
        return out

    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(selected_test)
    rng.shuffle(test_if_unlabeled)

    train_out = [format_train_eval(x) for x in train_set]
    val_out = [format_train_eval(x) for x in val_set]
    test_labeled_out = [format_test(x) for x in selected_test]
    test_if_unlabeled_out = [format_test(x) for x in test_if_unlabeled]

    out_train = data_dir / f"{args.output_prefix}_train.json"
    out_val = data_dir / f"{args.output_prefix}_val.json"
    out_test = data_dir / f"{args.output_prefix}_test_labeled.json"
    out_if_unlabeled = data_dir / f"{args.output_prefix}_test_if_unlabeled.json"
    out_manifest = data_dir / f"{args.output_prefix}_manifest.json"

    out_train.write_text(json.dumps(train_out, ensure_ascii=False, indent=2), encoding="utf-8")
    out_val.write_text(json.dumps(val_out, ensure_ascii=False, indent=2), encoding="utf-8")
    out_test.write_text(json.dumps(test_labeled_out, ensure_ascii=False, indent=2), encoding="utf-8")
    out_if_unlabeled.write_text(json.dumps(test_if_unlabeled_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Leakage checks
    k_train = {dedup_key(x) for x in train_out}
    k_val = {dedup_key(x) for x in val_out}
    k_test = {dedup_key(x) for x in test_labeled_out}
    k_if_unlabeled = {dedup_key(x) for x in test_if_unlabeled_out}

    leakage = {
        "train_val": len(k_train & k_val),
        "train_test_labeled": len(k_train & k_test),
        "val_test_labeled": len(k_val & k_test),
        "train_test_if_unlabeled": len(k_train & k_if_unlabeled),
        "val_test_if_unlabeled": len(k_val & k_if_unlabeled),
        "test_labeled_vs_test_if_unlabeled": len(k_test & k_if_unlabeled),
    }

    manifest = {
        "args": vars(args),
        "counts": {
            "train": len(train_out),
            "val": len(val_out),
            "test_labeled": len(test_labeled_out),
            "test_if_unlabeled": len(test_if_unlabeled_out),
        },
        "task_distribution": {
            "train": count_by_task(train_out),
            "val": count_by_task(val_out),
            "test_labeled": count_by_task(test_labeled_out),
            "test_if_unlabeled": count_by_task(test_if_unlabeled_out),
        },
        "source_distribution": {
            "train": count_by_source(train_out),
            "val": count_by_source(val_out),
            "test_labeled": count_by_source(test_labeled_out),
            "test_if_unlabeled": count_by_source(test_if_unlabeled_out),
        },
        "pool_stats": {
            "skipped_no_output": skipped_no_output,
            "skipped_other": skipped_other,
            "skipped_test_overlap": skipped_test_overlap,
            "train_pool_translation_en2zh": len(train_trans_en2zh),
            "train_pool_translation_zh2en": len(train_trans_zh2en),
            "train_pool_summarization": len(train_summ),
            "train_pool_instruction_following": len(train_if),
            "labeled_test_pool_translation_en2zh": len(test_trans_en2zh),
            "labeled_test_pool_translation_zh2en": len(test_trans_zh2en),
            "labeled_test_pool_summarization": len(test_summ),
            "labeled_test_pool_instruction_following": len(test_if_labeled),
        },
        "leakage": leakage,
    }
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.audit_mode:
        audit_dir = Path(args.audit_dir) if args.audit_dir else (data_dir / "audit")
        audit_dir.mkdir(parents=True, exist_ok=True)

        lineage_path = audit_dir / f"{args.output_prefix}_lineage.jsonl"
        source_snapshot_path = audit_dir / f"{args.output_prefix}_source_snapshot.json"
        hashes_path = audit_dir / f"{args.output_prefix}_hashes.json"

        lineage_records: List[Dict] = []
        lineage_records.extend(build_lineage_records(train_set, "train"))
        lineage_records.extend(build_lineage_records(val_set, "val"))
        lineage_records.extend(build_lineage_records(selected_test, "test_labeled"))
        lineage_records.extend(build_lineage_records(test_if_unlabeled, "test_if_unlabeled"))
        with lineage_path.open("w", encoding="utf-8") as f:
            for rec in lineage_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        raw_if_sources = [
            dataset_dir / "m-ifeval",
            dataset_dir / "IFEval" / "input_data.jsonl",
        ]
        input_files: List[Path] = []
        for p, _ in train_sources:
            if p.exists():
                input_files.append(p)
        for p, _ in eval_labeled_sources:
            if p.exists():
                input_files.append(p)
        for raw_src in raw_if_sources:
            if raw_src.is_dir():
                input_files.extend(sorted(x for x in raw_src.glob("*") if x.is_file()))
            elif raw_src.exists():
                input_files.append(raw_src)
        # De-duplicate while preserving order.
        seen_input: Set[str] = set()
        input_files = [p for p in input_files if not (str(p.resolve()) in seen_input or seen_input.add(str(p.resolve())))]

        source_snapshot = {
            "output_prefix": args.output_prefix,
            "confirmed_user_generated": [
                "data/train.json",
                "data/val.json",
            ],
            "direct_input_files": [
                {"path": str(p), "size_bytes": p.stat().st_size}
                for p in input_files
            ],
            "final_split_source_labels": {
                "train": count_by_source(train_out),
                "val": count_by_source(val_out),
                "test_labeled": count_by_source(test_labeled_out),
                "test_if_unlabeled": count_by_source(test_if_unlabeled_out),
            },
            "final_split_source_files": {
                "train": count_by_source_file(train_set),
                "val": count_by_source_file(val_set),
                "test_labeled": count_by_source_file(selected_test),
                "test_if_unlabeled": count_by_source_file(test_if_unlabeled),
            },
            "lineage_record_count": len(lineage_records),
        }
        source_snapshot_path.write_text(json.dumps(source_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

        hash_entries = {}
        files_to_hash = input_files + [out_train, out_val, out_test, out_if_unlabeled, out_manifest, Path(__file__).resolve()]
        for p in files_to_hash:
            if p.exists() and p.is_file():
                hash_entries[str(p)] = {
                    "sha256": sha256_file(p),
                    "size_bytes": p.stat().st_size,
                }
        hashes_path.write_text(json.dumps({
            "output_prefix": args.output_prefix,
            "files": hash_entries,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Qwen3 rigorous dataset build complete")
    print("=" * 72)
    print(f"train: {out_train} ({len(train_out)})")
    print(f"val:   {out_val} ({len(val_out)})")
    print(f"test:  {out_test} ({len(test_labeled_out)})")
    print(f"test_if_unlabeled: {out_if_unlabeled} ({len(test_if_unlabeled_out)})")
    print(f"manifest: {out_manifest}")
    print("task(train):", manifest["task_distribution"]["train"])
    print("task(val):", manifest["task_distribution"]["val"])
    print("task(test_labeled):", manifest["task_distribution"]["test_labeled"])
    print("leakage:", leakage)
    if args.audit_mode:
        print(f"audit_lineage: {lineage_path}")
        print(f"audit_source_snapshot: {source_snapshot_path}")
        print(f"audit_hashes: {hashes_path}")


if __name__ == "__main__":
    main()
