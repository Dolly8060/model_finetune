#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v4数据集构建脚本 - 指令遵循能力增强版

设计原则（继承v3经验，扩展指令遵循）：
1. 任务配比：翻译:总结:指令遵循 = 1:1:1（均衡配比）
2. 语言平衡：翻译保持en2zh:zh2en=1:1，指令遵循覆盖中英
3. 数据隔离：训练/验证/测试集完全隔离

目标输出：
- train_v4.json: ~3600条（翻译1200 + 总结1200 + 指令遵循1200）
- val_v4.json: ~360条（翻译120 + 总结120 + 指令遵循120）
- test_v4.json: ~360条（翻译120 + 总结120 + 指令遵循120）

数据来源：
- 翻译/总结：复用 v3 数据集
- 指令遵循：IFEval + M-IFEval + API生成中文数据
"""

import json
import random
import os
import sys
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple

random.seed(2026)

# ============================================================
# 任务分类关键词
# ============================================================

# 翻译关键词
TRANSLATION_KEYWORDS = [
    '翻译', 'translate', 'translation', '译为', '译成',
    '中文翻译', 'english translation', 'chinese translation',
    'to chinese', 'to english', '译为中文', '译为英文'
]

# 总结关键词
SUMMARIZATION_KEYWORDS = [
    '总结', '概括', '摘要', '提炼', '归纳',
    'summarize', 'summary', 'extract', 'main points', 'key points',
    '核心内容', '主要贡献', '主要观点'
]

# 指令遵循关键词（排除翻译和总结后识别）
INSTRUCTION_FOLLOWING_KEYWORDS = [
    # 字数约束
    'at least', 'at most', 'exactly', 'no more than', 'no less than',
    '不少于', '不超过', '至少', '最多', '恰好',
    # 格式约束
    'bullet point', 'numbered list', 'json format', 'markdown',
    '列表形式', 'JSON格式', '问答形式',
    # 关键词约束
    'must include', 'must contain', 'do not use', 'avoid using',
    '必须包含', '不要使用', '每段必须',
    # 结构约束
    'end with', 'start with', 'paragraphs',
    '以问句结尾', '开头必须', '段',
]

# 翻译方向判断
EN2ZH_KEYWORDS = ['翻译成中文', '译成中文', 'to chinese', 'into chinese', '中文翻译', '译为中文']
ZH2EN_KEYWORDS = ['翻译成英文', '译成英文', 'to english', 'into english', 'english translation', '译为英文']


def classify_sample(sample: Dict) -> Tuple[str, str]:
    """分类样本：任务类型 + 翻译方向
    
    Args:
        sample: 数据样本
        
    Returns:
        (task_type, direction) - 任务类型和翻译方向
    """
    instruction = sample.get('instruction', '').lower()
    task_type_field = sample.get('task_type', '').lower()
    
    # 优先使用 task_type 字段
    if task_type_field == 'instruction_following':
        return 'instruction_following', None
    
    # 判断任务类型
    is_translation = any(kw in instruction for kw in TRANSLATION_KEYWORDS)
    is_summarization = any(kw in instruction for kw in SUMMARIZATION_KEYWORDS)
    is_instruction_following = any(kw in instruction for kw in INSTRUCTION_FOLLOWING_KEYWORDS)
    
    # 翻译优先级最高
    if is_translation:
        # 判断翻译方向
        is_en2zh = any(kw in instruction for kw in EN2ZH_KEYWORDS)
        is_zh2en = any(kw in instruction for kw in ZH2EN_KEYWORDS)
        
        if is_en2zh:
            direction = 'en2zh'
        elif is_zh2en:
            direction = 'zh2en'
        else:
            # 根据input判断
            input_text = sample.get('input', '')
            if input_text:
                chinese_ratio = sum(1 for c in input_text if '\u4e00' <= c <= '\u9fff') / max(len(input_text), 1)
                direction = 'zh2en' if chinese_ratio > 0.3 else 'en2zh'
            else:
                direction = 'unknown'
        return 'translation', direction
    
    elif is_summarization:
        return 'summarization', None
    
    elif is_instruction_following:
        return 'instruction_following', None
    
    else:
        return 'other', None


def load_and_classify(filepath: str) -> Dict[str, List[Dict]]:
    """加载并分类数据
    
    Args:
        filepath: 数据文件路径
        
    Returns:
        按类别分组的数据字典
    """
    if not os.path.exists(filepath):
        print(f"  跳过不存在的文件: {filepath}")
        return defaultdict(list)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    classified = defaultdict(list)
    for sample in data:
        task_type, direction = classify_sample(sample)
        if task_type == 'translation':
            classified[f'translation_{direction}'].append(sample)
        elif task_type == 'summarization':
            classified['summarization'].append(sample)
        elif task_type == 'instruction_following':
            # 按语言分类
            lang = sample.get('language', 'unknown')
            if lang == 'unknown':
                # 根据指令内容判断语言
                instruction = sample.get('instruction', '')
                chinese_ratio = sum(1 for c in instruction if '\u4e00' <= c <= '\u9fff') / max(len(instruction), 1)
                lang = 'zh' if chinese_ratio > 0.3 else 'en'
            classified[f'instruction_following_{lang}'].append(sample)
        else:
            classified['other'].append(sample)
    
    return classified


def deduplicate(samples: List[Dict], key_len: int = 200) -> List[Dict]:
    """去重（基于instruction + input的前N字符）
    
    Args:
        samples: 样本列表
        key_len: 用于去重的instruction字符长度（默认200以避免误判）
        
    Returns:
        去重后的样本列表
    """
    seen = set()
    unique = []
    for s in samples:
        # 使用更长的instruction作为key，避免前缀相似导致误判
        key = (s.get('instruction', '')[:key_len] + s.get('input', '')[:100])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def build_v4_dataset(
    translation_ratio: float = 1.0,
    summarization_ratio: float = 1.0,
    instruction_following_ratio: float = 1.0,
    train_size: int = 3600,
    val_size: int = 360,
    test_size: int = 600,  # 测试集增大到600条
    output_dir: str = 'data'
):
    """构建v4数据集
    
    Args:
        translation_ratio: 翻译任务比例权重
        summarization_ratio: 总结任务比例权重  
        instruction_following_ratio: 指令遵循任务比例权重
        train_size: 训练集大小
        val_size: 验证集大小
        test_size: 测试集大小
        output_dir: 输出目录
    """
    
    print("=" * 60)
    print("v4 数据集构建")
    print("=" * 60)
    
    # ================================================================
    # 数据来源配置
    # ================================================================
    
    sources = {
        # v3 数据集（翻译 + 总结）
        'v3_train': 'data/train_v3.json',
        'v3_val': 'data/val_v3.json',
        'v3_test': 'data/test_v3.json',
        
        # 历史混合数据
        'train_mixed': 'data/train_mixed_3k.json',
        
        # 公开数据
        'public_v2': 'data/public_val_v2.json',
        
        # IFEval 数据（指令遵循）- 带响应的主要来源
        'ifeval_combined': 'data/ifeval_combined.json',  # 英文1500 + 中文579
    }
    
    # ================================================================
    # 收集所有数据
    # ================================================================
    
    all_translation_en2zh = []
    all_translation_zh2en = []
    all_summarization = []
    all_instruction_following_en = []
    all_instruction_following_zh = []
    
    print("\n[1/5] 加载数据源...")
    
    for name, path in sources.items():
        full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
        if not os.path.exists(full_path):
            full_path = path
        
        if not os.path.exists(full_path):
            print(f"  跳过: {path} (文件不存在)")
            continue
        
        print(f"\n  处理: {path}")
        classified = load_and_classify(full_path)
        
        for key, samples in classified.items():
            if samples:
                print(f"    {key}: {len(samples)}条")
        
        all_translation_en2zh.extend(classified.get('translation_en2zh', []))
        all_translation_zh2en.extend(classified.get('translation_zh2en', []))
        all_summarization.extend(classified.get('summarization', []))
        all_instruction_following_en.extend(classified.get('instruction_following_en', []))
        all_instruction_following_zh.extend(classified.get('instruction_following_zh', []))
    
    # ================================================================
    # 去重
    # ================================================================
    
    print("\n[2/5] 数据去重...")
    
    all_translation_en2zh = deduplicate(all_translation_en2zh)
    all_translation_zh2en = deduplicate(all_translation_zh2en)
    all_summarization = deduplicate(all_summarization)
    all_instruction_following_en = deduplicate(all_instruction_following_en)
    all_instruction_following_zh = deduplicate(all_instruction_following_zh)
    
    print(f"  en2zh翻译: {len(all_translation_en2zh)}条")
    print(f"  zh2en翻译: {len(all_translation_zh2en)}条")
    print(f"  总结: {len(all_summarization)}条")
    print(f"  指令遵循(英文): {len(all_instruction_following_en)}条")
    print(f"  指令遵循(中文): {len(all_instruction_following_zh)}条")
    
    # 合并指令遵循数据
    all_instruction_following = all_instruction_following_en + all_instruction_following_zh
    
    # ================================================================
    # 计算目标配比
    # ================================================================
    
    print("\n[3/5] 计算数据配比...")
    
    # 归一化比例
    total_ratio = translation_ratio + summarization_ratio + instruction_following_ratio
    trans_ratio = translation_ratio / total_ratio
    summ_ratio = summarization_ratio / total_ratio
    if_ratio = instruction_following_ratio / total_ratio
    
    # 各任务目标数量
    train_trans = int(train_size * trans_ratio)
    train_summ = int(train_size * summ_ratio)
    train_if = train_size - train_trans - train_summ
    
    val_trans = int(val_size * trans_ratio)
    val_summ = int(val_size * summ_ratio)
    val_if = val_size - val_trans - val_summ
    
    test_trans = int(test_size * trans_ratio)
    test_summ = int(test_size * summ_ratio)
    test_if = test_size - test_trans - test_summ
    
    # 翻译按方向平分
    train_en2zh = train_trans // 2
    train_zh2en = train_trans - train_en2zh
    val_en2zh = val_trans // 2
    val_zh2en = val_trans - val_en2zh
    test_en2zh = test_trans // 2
    test_zh2en = test_trans - test_en2zh
    
    print(f"\n  目标配比 (翻译:总结:指令遵循 = {trans_ratio:.2f}:{summ_ratio:.2f}:{if_ratio:.2f})")
    print(f"\n  训练集目标: {train_size}条")
    print(f"    - 翻译: {train_trans} (en2zh:{train_en2zh}, zh2en:{train_zh2en})")
    print(f"    - 总结: {train_summ}")
    print(f"    - 指令遵循: {train_if}")
    
    print(f"\n  验证集目标: {val_size}条")
    print(f"    - 翻译: {val_trans} (en2zh:{val_en2zh}, zh2en:{val_zh2en})")
    print(f"    - 总结: {val_summ}")
    print(f"    - 指令遵循: {val_if}")
    
    print(f"\n  测试集目标: {test_size}条")
    print(f"    - 翻译: {test_trans} (en2zh:{test_en2zh}, zh2en:{test_zh2en})")
    print(f"    - 总结: {test_summ}")
    print(f"    - 指令遵循: {test_if}")
    
    # ================================================================
    # 数据可用性检查与调整
    # ================================================================
    
    print("\n[4/5] 数据可用性检查...")
    
    def adjust_allocation(available, *needs):
        """按比例调整分配（如果数据不足）"""
        total_need = sum(needs)
        if available >= total_need:
            return needs
        ratio = available / total_need
        return tuple(int(n * ratio) for n in needs)
    
    # 检查并调整
    total_en2zh_need = train_en2zh + val_en2zh + test_en2zh
    total_zh2en_need = train_zh2en + val_zh2en + test_zh2en
    total_summ_need = train_summ + val_summ + test_summ
    total_if_need = train_if + val_if + test_if
    
    print(f"  en2zh: 需要 {total_en2zh_need}, 可用 {len(all_translation_en2zh)}")
    print(f"  zh2en: 需要 {total_zh2en_need}, 可用 {len(all_translation_zh2en)}")
    print(f"  总结: 需要 {total_summ_need}, 可用 {len(all_summarization)}")
    print(f"  指令遵循: 需要 {total_if_need}, 可用 {len(all_instruction_following)}")
    
    # 调整分配
    train_en2zh, val_en2zh, test_en2zh = adjust_allocation(
        len(all_translation_en2zh), train_en2zh, val_en2zh, test_en2zh)
    train_zh2en, val_zh2en, test_zh2en = adjust_allocation(
        len(all_translation_zh2en), train_zh2en, val_zh2en, test_zh2en)
    train_summ, val_summ, test_summ = adjust_allocation(
        len(all_summarization), train_summ, val_summ, test_summ)
    train_if, val_if, test_if = adjust_allocation(
        len(all_instruction_following), train_if, val_if, test_if)
    
    # ================================================================
    # 打乱并切分数据
    # ================================================================
    
    random.shuffle(all_translation_en2zh)
    random.shuffle(all_translation_zh2en)
    random.shuffle(all_summarization)
    random.shuffle(all_instruction_following)
    
    # 切分顺序：test -> val -> train（测试集优先）
    
    # en2zh
    test_data_en2zh = all_translation_en2zh[:test_en2zh]
    val_data_en2zh = all_translation_en2zh[test_en2zh:test_en2zh + val_en2zh]
    train_data_en2zh = all_translation_en2zh[test_en2zh + val_en2zh:test_en2zh + val_en2zh + train_en2zh]
    
    # zh2en
    test_data_zh2en = all_translation_zh2en[:test_zh2en]
    val_data_zh2en = all_translation_zh2en[test_zh2en:test_zh2en + val_zh2en]
    train_data_zh2en = all_translation_zh2en[test_zh2en + val_zh2en:test_zh2en + val_zh2en + train_zh2en]
    
    # summarization
    test_data_summ = all_summarization[:test_summ]
    val_data_summ = all_summarization[test_summ:test_summ + val_summ]
    train_data_summ = all_summarization[test_summ + val_summ:test_summ + val_summ + train_summ]
    
    # instruction_following
    test_data_if = all_instruction_following[:test_if]
    val_data_if = all_instruction_following[test_if:test_if + val_if]
    train_data_if = all_instruction_following[test_if + val_if:test_if + val_if + train_if]
    
    # ================================================================
    # 合并并保存
    # ================================================================
    
    print("\n[5/5] 合并并保存数据集...")
    
    def merge_and_shuffle(*datasets):
        """合并数据集并打乱，只保留alpaca格式字段"""
        merged = []
        for ds in datasets:
            merged.extend(ds)
        random.shuffle(merged)
        return [{'instruction': s['instruction'], 'input': s.get('input', ''), 'output': s['output']} 
                for s in merged if s.get('output')]  # 过滤没有output的数据
    
    train_v4 = merge_and_shuffle(train_data_en2zh, train_data_zh2en, train_data_summ, train_data_if)
    val_v4 = merge_and_shuffle(val_data_en2zh, val_data_zh2en, val_data_summ, val_data_if)
    test_v4 = merge_and_shuffle(test_data_en2zh, test_data_zh2en, test_data_summ, test_data_if)
    
    # 强制数据隔离：测试集优先，从val/train中移除重复项
    def get_key(s):
        return s['instruction'][:200] + s.get('input', '')[:100]
    
    test_keys = set(get_key(s) for s in test_v4)
    val_v4 = [s for s in val_v4 if get_key(s) not in test_keys]
    val_keys = set(get_key(s) for s in val_v4)
    train_v4 = [s for s in train_v4 if get_key(s) not in test_keys and get_key(s) not in val_keys]
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_v4.json')
    val_path = os.path.join(output_dir, 'val_v4.json')
    test_path = os.path.join(output_dir, 'test_v4.json')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_v4, f, ensure_ascii=False, indent=2)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_v4, f, ensure_ascii=False, indent=2)
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_v4, f, ensure_ascii=False, indent=2)
    
    # ================================================================
    # 输出统计
    # ================================================================
    
    print("\n" + "=" * 60)
    print("v4数据集构建完成！")
    print("=" * 60)
    
    print(f"\n训练集 (train_v4.json): {len(train_v4)}条")
    print(f"  - en2zh翻译: {len(train_data_en2zh)}条")
    print(f"  - zh2en翻译: {len(train_data_zh2en)}条")
    print(f"  - 总结: {len(train_data_summ)}条")
    print(f"  - 指令遵循: {len(train_data_if)}条")
    actual_train_trans = len(train_data_en2zh) + len(train_data_zh2en)
    print(f"  - 实际配比 翻译:总结:指令 = {actual_train_trans}:{len(train_data_summ)}:{len(train_data_if)}")
    
    print(f"\n验证集 (val_v4.json): {len(val_v4)}条")
    print(f"  - en2zh翻译: {len(val_data_en2zh)}条")
    print(f"  - zh2en翻译: {len(val_data_zh2en)}条")
    print(f"  - 总结: {len(val_data_summ)}条")
    print(f"  - 指令遵循: {len(val_data_if)}条")
    
    print(f"\n测试集 (test_v4.json): {len(test_v4)}条")
    print(f"  - en2zh翻译: {len(test_data_en2zh)}条")
    print(f"  - zh2en翻译: {len(test_data_zh2en)}条")
    print(f"  - 总结: {len(test_data_summ)}条")
    print(f"  - 指令遵循: {len(test_data_if)}条")
    
    # ================================================================
    # 验证数据隔离
    # ================================================================
    
    train_keys = set(s['instruction'][:200] + s.get('input', '')[:100] for s in train_v4)
    val_keys = set(s['instruction'][:200] + s.get('input', '')[:100] for s in val_v4)
    test_keys = set(s['instruction'][:200] + s.get('input', '')[:100] for s in test_v4)
    
    train_val_overlap = len(train_keys & val_keys)
    train_test_overlap = len(train_keys & test_keys)
    val_test_overlap = len(val_keys & test_keys)
    
    print(f"\n数据隔离验证:")
    print(f"  train-val重叠: {train_val_overlap}条")
    print(f"  train-test重叠: {train_test_overlap}条")
    print(f"  val-test重叠: {val_test_overlap}条")
    
    if train_val_overlap + train_test_overlap + val_test_overlap == 0:
        print("  [OK] 所有数据集完全隔离！")
    else:
        print("  [WARN] 存在数据泄漏，请检查！")
    
    print(f"\n已保存:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")
    
    return train_v4, val_v4, test_v4


def main():
    parser = argparse.ArgumentParser(description="构建 v4 数据集（指令遵循增强版）")
    parser.add_argument(
        "--train-size", type=int, default=3600,
        help="训练集目标大小"
    )
    parser.add_argument(
        "--val-size", type=int, default=360,
        help="验证集目标大小"
    )
    parser.add_argument(
        "--test-size", type=int, default=600,
        help="测试集目标大小（默认600条）"
    )
    parser.add_argument(
        "--translation-ratio", type=float, default=1.0,
        help="翻译任务比例权重"
    )
    parser.add_argument(
        "--summarization-ratio", type=float, default=1.0,
        help="总结任务比例权重"
    )
    parser.add_argument(
        "--instruction-following-ratio", type=float, default=1.0,
        help="指令遵循任务比例权重"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="输出目录"
    )
    args = parser.parse_args()
    
    build_v4_dataset(
        translation_ratio=args.translation_ratio,
        summarization_ratio=args.summarization_ratio,
        instruction_following_ratio=args.instruction_following_ratio,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
