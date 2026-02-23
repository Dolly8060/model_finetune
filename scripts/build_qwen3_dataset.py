#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3 专用数据集构建脚本

设计原则：
1. 每条样本携带 source（来源）和 task_type（任务类型）溯源字段
2. 为 Qwen3 思考模式在 response 前添加空思考前缀 <think>

</think>


3. 训练/验证/测试集数据隔离（与 test_v4_enhanced.json 交叉去重）
4. 保留 alpaca 格式兼容 LLaMA-Factory（instruction/input/output）
5. 三任务均衡配比：翻译:总结:指令遵循 ≈ 1:1:1

数据源溯源：
- train.json (1800)        → 自建翻译(888)+总结(912)
- val.json (200)           → 自建翻译(112)+总结(88)
- train_v3.json (2707)     → v3版翻译+总结
- val_v3.json (270)        → v3版翻译+总结
- train_mixed_3k.json (3000) → 混合翻译+总结
- public_val_v2.json (900) → 公开翻译数据
- ifeval_full_with_meta.json (579) → 中文指令遵循(模板生成)
- argilla_ifeval.json (1500) → 英文指令遵循(Argilla公开)

输出：
- data/qwen3_train.json (~3200条)
- data/qwen3_val.json (~260条)
"""

import json
import random
import os
import sys
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Set

random.seed(2026)

# Qwen3 思考模式空前缀
THINK_PREFIX = "<think>\n\n</think>\n\n"

# ============================================================
# 任务分类关键词（复用 build_v4 的分类逻辑）
# ============================================================

TRANSLATION_KEYWORDS = [
    '翻译', 'translate', 'translation', '译为', '译成',
    '中文翻译', 'english translation', 'chinese translation',
    'to chinese', 'to english', '译为中文', '译为英文',
    '中译英', '英译中', '翻成中文', '翻成英文',
]

SUMMARIZATION_KEYWORDS = [
    '总结', '概括', '摘要', '提炼', '归纳',
    'summarize', 'summary', 'extract', 'main points', 'key points',
    '核心内容', '主要贡献', '主要观点', '核心观点',
]

EN2ZH_KEYWORDS = ['翻译成中文', '译成中文', 'to chinese', 'into chinese', '中文翻译', '译为中文', '翻成中文', '英译中']
ZH2EN_KEYWORDS = ['翻译成英文', '译成英文', 'to english', 'into english', 'english translation', '译为英文', '翻成英文', '中译英']


def classify_sample(sample: Dict) -> Tuple[str, str]:
    """分类样本：返回 (task_type, sub_type)
    
    task_type: translation / summarization / instruction_following
    sub_type:  en2zh / zh2en / zh / en / None
    """
    # 优先使用已有 task_type 字段
    existing_type = sample.get('task_type', '').lower()
    if existing_type == 'instruction_following':
        lang = sample.get('language', 'en')
        return 'instruction_following', lang
    if existing_type == 'translation':
        return 'translation', _detect_translation_direction(sample)
    if existing_type == 'summarization':
        return 'summarization', None

    instruction = sample.get('instruction', '').lower()

    # 翻译（优先级最高）
    if any(kw in instruction for kw in TRANSLATION_KEYWORDS):
        return 'translation', _detect_translation_direction(sample)

    # 总结
    if any(kw in instruction for kw in SUMMARIZATION_KEYWORDS):
        return 'summarization', None

    # 其他 → 归为 other（不强制归入IF）
    return 'other', None


def _detect_translation_direction(sample: Dict) -> str:
    instruction = sample.get('instruction', '').lower()
    if any(kw in instruction for kw in EN2ZH_KEYWORDS):
        return 'en2zh'
    if any(kw in instruction for kw in ZH2EN_KEYWORDS):
        return 'zh2en'
    # 根据 input 内容推断
    input_text = sample.get('input', '')
    if input_text:
        chinese_ratio = sum(1 for c in input_text if '\u4e00' <= c <= '\u9fff') / max(len(input_text), 1)
        return 'zh2en' if chinese_ratio > 0.3 else 'en2zh'
    return 'en2zh'  # 默认


def get_dedup_key(sample: Dict, key_len: int = 200) -> str:
    """生成去重键"""
    return sample.get('instruction', '')[:key_len] + '|||' + sample.get('input', '')[:100]


def load_source(filepath: str, source_label: str) -> List[Dict]:
    """加载数据源并标注来源"""
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} (不存在)")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 标注来源
    for s in data:
        if 'source' not in s:
            s['source'] = source_label
    print(f"  [LOAD] {filepath}: {len(data)} 条 (source={source_label})")
    return data


def add_think_prefix(output: str) -> str:
    """为 Qwen3 添加空思考前缀"""
    if not output:
        return output
    # 避免重复添加
    if output.startswith('<think>'):
        return output
    return THINK_PREFIX + output


def build_qwen3_dataset(
    train_target: int = 3200,
    val_target: int = 260,
    output_dir: str = 'data',
    test_file: str = 'data/test_v4_enhanced.json',
):
    """构建 Qwen3 专用训练/验证集"""

    print("=" * 60)
    print("Qwen3 数据集构建")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    # ================================================================
    # Step 1: 加载所有原始数据源
    # ================================================================
    print("\n[1/6] 加载原始数据源...")

    # 翻译+总结来源
    src_train = load_source('data/train.json', 'self_generated')
    src_val = load_source('data/val.json', 'self_generated')
    src_v3_train = load_source('data/train_v3.json', 'v3_dataset')
    src_v3_val = load_source('data/val_v3.json', 'v3_dataset')
    src_mixed = load_source('data/train_mixed_3k.json', 'mixed_3k')
    src_public = load_source('data/public_val_v2.json', 'public_v2')

    # 指令遵循来源
    src_ifeval_zh = load_source('data/ifeval_full_with_meta.json', 'chinese_generated')
    src_ifeval_en = load_source('data/argilla_ifeval.json', 'argilla_ifeval')

    # ================================================================
    # Step 2: 加载测试集用于交叉去重
    # ================================================================
    print("\n[2/6] 加载测试集用于交叉去重...")
    test_path = os.path.join(base_dir, test_file)
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"  测试集: {len(test_data)} 条")
    else:
        test_data = []
        print(f"  [WARN] 测试集 {test_file} 不存在，跳过去重")

    test_keys: Set[str] = {get_dedup_key(s) for s in test_data}

    # ================================================================
    # Step 3: 分类所有样本
    # ================================================================
    print("\n[3/6] 分类所有样本...")

    # 收集桶
    trans_en2zh: List[Dict] = []
    trans_zh2en: List[Dict] = []
    summarization: List[Dict] = []
    if_zh: List[Dict] = []
    if_en: List[Dict] = []
    skipped_other = 0
    skipped_no_output = 0
    skipped_test_overlap = 0

    all_sources = (
        src_train + src_val + src_v3_train + src_v3_val +
        src_mixed + src_public + src_ifeval_zh + src_ifeval_en
    )

    for sample in all_sources:
        # 过滤无 output 的样本
        if not sample.get('output', '').strip():
            skipped_no_output += 1
            continue

        # 过滤与测试集重叠的样本
        if get_dedup_key(sample) in test_keys:
            skipped_test_overlap += 1
            continue

        task_type, sub_type = classify_sample(sample)
        sample['task_type'] = task_type

        if task_type == 'translation':
            if sub_type == 'en2zh':
                trans_en2zh.append(sample)
            else:
                trans_zh2en.append(sample)
        elif task_type == 'summarization':
            summarization.append(sample)
        elif task_type == 'instruction_following':
            if sub_type == 'zh':
                if_zh.append(sample)
            else:
                if_en.append(sample)
        else:
            skipped_other += 1

    print(f"  翻译 en2zh: {len(trans_en2zh)}")
    print(f"  翻译 zh2en: {len(trans_zh2en)}")
    print(f"  总结:       {len(summarization)}")
    print(f"  指令遵循 zh: {len(if_zh)}")
    print(f"  指令遵循 en: {len(if_en)}")
    print(f"  跳过(other): {skipped_other}")
    print(f"  跳过(无output): {skipped_no_output}")
    print(f"  跳过(测试集重叠): {skipped_test_overlap}")

    # ================================================================
    # Step 4: 去重
    # ================================================================
    print("\n[4/6] 去重...")

    def dedup(samples: List[Dict]) -> List[Dict]:
        seen = set()
        result = []
        for s in samples:
            key = get_dedup_key(s)
            if key not in seen:
                seen.add(key)
                result.append(s)
        return result

    trans_en2zh = dedup(trans_en2zh)
    trans_zh2en = dedup(trans_zh2en)
    summarization = dedup(summarization)
    if_zh = dedup(if_zh)
    if_en = dedup(if_en)

    print(f"  去重后 - en2zh: {len(trans_en2zh)}, zh2en: {len(trans_zh2en)}, "
          f"总结: {len(summarization)}, IF_zh: {len(if_zh)}, IF_en: {len(if_en)}")

    # ================================================================
    # Step 5: 按配比切分 train/val
    # ================================================================
    print("\n[5/6] 按配比切分 train/val...")

    # 目标: 三任务均衡 1:1:1
    # 翻译内部: en2zh:zh2en = 1:1
    train_per_task = train_target // 3
    val_per_task = val_target // 3

    # 翻译子任务
    train_trans_each = train_per_task // 2
    val_trans_each = val_per_task // 2

    def split_val_train(pool: List[Dict], val_n: int, train_n: int):
        """从池中切分验证集和训练集（验证集优先）"""
        random.shuffle(pool)
        val_actual = min(val_n, len(pool))
        val_set = pool[:val_actual]
        remaining = pool[val_actual:]
        train_actual = min(train_n, len(remaining))
        train_set = remaining[:train_actual]
        return train_set, val_set

    train_en2zh, val_en2zh = split_val_train(trans_en2zh, val_trans_each, train_trans_each)
    train_zh2en, val_zh2en = split_val_train(trans_zh2en, val_trans_each, train_trans_each)
    train_summ, val_summ = split_val_train(summarization, val_per_task, train_per_task)

    # 指令遵循: 中英混合，尽量平衡
    if_all = if_zh + if_en
    random.shuffle(if_all)
    train_if, val_if = split_val_train(if_all, val_per_task, train_per_task)

    # 汇总
    train_data = train_en2zh + train_zh2en + train_summ + train_if
    val_data = val_en2zh + val_zh2en + val_summ + val_if

    print(f"  训练集: {len(train_data)} 条")
    print(f"    - en2zh: {len(train_en2zh)}, zh2en: {len(train_zh2en)}, "
          f"总结: {len(train_summ)}, 指令遵循: {len(train_if)}")
    print(f"  验证集: {len(val_data)} 条")
    print(f"    - en2zh: {len(val_en2zh)}, zh2en: {len(val_zh2en)}, "
          f"总结: {len(val_summ)}, 指令遵循: {len(val_if)}")

    # ================================================================
    # Step 5.5: 训练/验证集交叉去重
    # ================================================================
    val_keys = {get_dedup_key(s) for s in val_data}
    before_train = len(train_data)
    train_data = [s for s in train_data if get_dedup_key(s) not in val_keys]
    if before_train != len(train_data):
        print(f"  移除 train 中与 val 重叠: {before_train - len(train_data)} 条")

    # ================================================================
    # Step 6: 格式化并保存
    # ================================================================
    print("\n[6/6] 格式化并保存...")

    def format_sample(sample: Dict) -> Dict:
        """格式化为 Qwen3 alpaca 格式 + 溯源字段"""
        return {
            'instruction': sample['instruction'],
            'input': sample.get('input', ''),
            'output': add_think_prefix(sample['output']),
            'source': sample.get('source', 'unknown'),
            'task_type': sample.get('task_type', 'unknown'),
        }

    random.shuffle(train_data)
    random.shuffle(val_data)

    train_formatted = [format_sample(s) for s in train_data]
    val_formatted = [format_sample(s) for s in val_data]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'qwen3_train.json')
    val_path = os.path.join(output_dir, 'qwen3_val.json')

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_formatted, f, ensure_ascii=False, indent=2)

    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_formatted, f, ensure_ascii=False, indent=2)

    # ================================================================
    # 验证报告
    # ================================================================
    print("\n" + "=" * 60)
    print("构建完成！")
    print("=" * 60)

    # 溯源统计
    for name, dataset in [('训练集', train_formatted), ('验证集', val_formatted)]:
        print(f"\n{name} ({len(dataset)} 条):")
        src_counts = defaultdict(int)
        task_counts = defaultdict(int)
        for s in dataset:
            src_counts[s['source']] += 1
            task_counts[s['task_type']] += 1
        print("  来源分布:")
        for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
            print(f"    {src}: {cnt}")
        print("  任务分布:")
        for task, cnt in sorted(task_counts.items(), key=lambda x: -x[1]):
            print(f"    {task}: {cnt}")

    # 思考前缀验证
    think_ok = sum(1 for s in train_formatted if s['output'].startswith('<think>'))
    print(f"\n思考前缀验证: {think_ok}/{len(train_formatted)} 条含 <think> 前缀")

    # 数据隔离验证
    print("\n数据隔离验证:")
    train_keys = {get_dedup_key(s) for s in train_formatted}
    val_keys = {get_dedup_key(s) for s in val_formatted}

    tv_overlap = len(train_keys & val_keys)
    tt_overlap = len(train_keys & test_keys)
    vt_overlap = len(val_keys & test_keys)

    print(f"  train-val 重叠: {tv_overlap}")
    print(f"  train-test 重叠: {tt_overlap}")
    print(f"  val-test 重叠: {vt_overlap}")

    if tv_overlap + tt_overlap + vt_overlap == 0:
        print("  [OK] 所有数据集完全隔离！")
    else:
        print("  [WARN] 存在数据泄漏，请检查！")

    print(f"\n输出文件:")
    print(f"  {train_path}")
    print(f"  {val_path}")

    return train_formatted, val_formatted


def main():
    parser = argparse.ArgumentParser(description="构建 Qwen3 专用数据集")
    parser.add_argument("--train-target", type=int, default=3200, help="训练集目标大小")
    parser.add_argument("--val-target", type=int, default=260, help="验证集目标大小")
    parser.add_argument("--output-dir", type=str, default="data", help="输出目录")
    parser.add_argument("--test-file", type=str, default="data/test_v4_enhanced.json",
                        help="测试集文件（用于交叉去重）")
    args = parser.parse_args()

    build_qwen3_dataset(
        train_target=args.train_target,
        val_target=args.val_target,
        output_dir=args.output_dir,
        test_file=args.test_file,
    )


if __name__ == "__main__":
    main()
