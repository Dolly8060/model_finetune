#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v3数据集构建脚本 - 针对翻译能力优化

设计原则（吸取v2/v4教训）：
1. 任务配比：翻译:总结 = 2:1（提高翻译权重）
2. 翻译方向平衡：en2zh : zh2en = 1:1（修复en2zh弱势问题）
3. 数据隔离：训练/验证/测试集完全隔离

目标输出：
- train_v3.json: 3000条（翻译2000 + 总结1000）
- val_v3.json: 300条（翻译200 + 总结100）  
- test_v3.json: 300条（翻译200 + 总结100）
"""

import json
import random
import os
from collections import defaultdict

random.seed(2026)

# 翻译关键词（用于分类）
TRANSLATION_KEYWORDS = [
    '翻译', 'translate', 'translation', '译为', '译成',
    '中文翻译', 'english translation', 'chinese translation'
]

# 总结关键词
SUMMARIZATION_KEYWORDS = [
    '总结', '概括', '摘要', '提炼', '归纳',
    'summarize', 'summary', 'extract', 'main points', 'key points'
]

# 方向判断关键词
EN2ZH_KEYWORDS = ['翻译成中文', '译成中文', 'to chinese', 'into chinese', '中文翻译']
ZH2EN_KEYWORDS = ['翻译成英文', '译成英文', 'to english', 'into english', 'english translation']


def classify_sample(sample):
    """分类样本：任务类型 + 翻译方向"""
    instruction = sample.get('instruction', '').lower()
    
    # 判断任务类型
    is_translation = any(kw in instruction for kw in TRANSLATION_KEYWORDS)
    is_summarization = any(kw in instruction for kw in SUMMARIZATION_KEYWORDS)
    
    if is_translation:
        task_type = 'translation'
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
        return task_type, direction
    elif is_summarization:
        return 'summarization', None
    else:
        return 'other', None


def load_and_classify(filepath):
    """加载并分类数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    classified = defaultdict(list)
    for sample in data:
        task_type, direction = classify_sample(sample)
        if task_type == 'translation':
            classified[f'translation_{direction}'].append(sample)
        elif task_type == 'summarization':
            classified['summarization'].append(sample)
        else:
            classified['other'].append(sample)
    
    return classified


def build_v3_dataset():
    """构建v3数据集"""
    
    # 数据来源
    sources = {
        'train_json': 'data/train.json',           # GPT-4生成数据
        'train_mixed': 'data/train_mixed_3k.json', # 混合数据
        'public_v2': 'data/public_val_v2.json',    # 公开数据（仅用于补充训练，不用于测试）
    }
    
    # 收集所有数据
    all_translation_en2zh = []
    all_translation_zh2en = []
    all_summarization = []
    
    for name, path in sources.items():
        if not os.path.exists(path):
            print(f"跳过不存在的文件: {path}")
            continue
        
        print(f"\n处理: {path}")
        classified = load_and_classify(path)
        
        for key, samples in classified.items():
            print(f"  {key}: {len(samples)}条")
        
        all_translation_en2zh.extend(classified.get('translation_en2zh', []))
        all_translation_zh2en.extend(classified.get('translation_zh2en', []))
        all_summarization.extend(classified.get('summarization', []))
    
    # 去重（基于input前100字符）
    def deduplicate(samples):
        seen = set()
        unique = []
        for s in samples:
            key = s.get('input', '')[:100]
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique
    
    all_translation_en2zh = deduplicate(all_translation_en2zh)
    all_translation_zh2en = deduplicate(all_translation_zh2en)
    all_summarization = deduplicate(all_summarization)
    
    print(f"\n去重后统计:")
    print(f"  en2zh翻译: {len(all_translation_en2zh)}条")
    print(f"  zh2en翻译: {len(all_translation_zh2en)}条")
    print(f"  总结: {len(all_summarization)}条")
    
    # 打乱数据
    random.shuffle(all_translation_en2zh)
    random.shuffle(all_translation_zh2en)
    random.shuffle(all_summarization)
    
    # ================================================================
    # 数据分配策略（v3核心设计）
    # 
    # 目标配比：翻译:总结 = 2:1
    # 翻译方向：en2zh:zh2en = 1:1
    # 
    # train: 3000条 = 翻译2000(en2zh 1000 + zh2en 1000) + 总结1000
    # val:   300条  = 翻译200(en2zh 100 + zh2en 100) + 总结100
    # test:  300条  = 翻译200(en2zh 100 + zh2en 100) + 总结100
    # ================================================================
    
    # 分配数量
    train_en2zh, train_zh2en, train_summ = 1000, 1000, 1000
    val_en2zh, val_zh2en, val_summ = 100, 100, 100
    test_en2zh, test_zh2en, test_summ = 100, 100, 100
    
    # 检查数据是否足够
    total_en2zh_needed = train_en2zh + val_en2zh + test_en2zh
    total_zh2en_needed = train_zh2en + val_zh2en + test_zh2en
    total_summ_needed = train_summ + val_summ + test_summ
    
    print(f"\n数据需求检查:")
    print(f"  en2zh需要: {total_en2zh_needed}, 可用: {len(all_translation_en2zh)}")
    print(f"  zh2en需要: {total_zh2en_needed}, 可用: {len(all_translation_zh2en)}")
    print(f"  总结需要: {total_summ_needed}, 可用: {len(all_summarization)}")
    
    # 按比例调整（如果数据不足）
    def adjust_allocation(available, *needs):
        total_need = sum(needs)
        if available >= total_need:
            return needs
        ratio = available / total_need
        return tuple(int(n * ratio) for n in needs)
    
    train_en2zh, val_en2zh, test_en2zh = adjust_allocation(
        len(all_translation_en2zh), train_en2zh, val_en2zh, test_en2zh)
    train_zh2en, val_zh2en, test_zh2en = adjust_allocation(
        len(all_translation_zh2en), train_zh2en, val_zh2en, test_zh2en)
    train_summ, val_summ, test_summ = adjust_allocation(
        len(all_summarization), train_summ, val_summ, test_summ)
    
    # 切分数据（确保无重叠）
    # 顺序：test -> val -> train（测试集优先，保证测试集质量）
    
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
    
    # 合并数据集
    def merge_and_shuffle(*datasets):
        merged = []
        for ds in datasets:
            merged.extend(ds)
        random.shuffle(merged)
        # 只保留alpaca格式字段
        return [{'instruction': s['instruction'], 'input': s['input'], 'output': s['output']} 
                for s in merged]
    
    train_v3 = merge_and_shuffle(train_data_en2zh, train_data_zh2en, train_data_summ)
    val_v3 = merge_and_shuffle(val_data_en2zh, val_data_zh2en, val_data_summ)
    test_v3 = merge_and_shuffle(test_data_en2zh, test_data_zh2en, test_data_summ)
    
    # 保存数据集
    os.makedirs('data', exist_ok=True)
    
    with open('data/train_v3.json', 'w', encoding='utf-8') as f:
        json.dump(train_v3, f, ensure_ascii=False, indent=2)
    
    with open('data/val_v3.json', 'w', encoding='utf-8') as f:
        json.dump(val_v3, f, ensure_ascii=False, indent=2)
    
    with open('data/test_v3.json', 'w', encoding='utf-8') as f:
        json.dump(test_v3, f, ensure_ascii=False, indent=2)
    
    # 输出统计
    print(f"\n" + "="*50)
    print("v3数据集构建完成！")
    print("="*50)
    
    print(f"\n训练集 (train_v3.json): {len(train_v3)}条")
    print(f"  - en2zh翻译: {len(train_data_en2zh)}条")
    print(f"  - zh2en翻译: {len(train_data_zh2en)}条")
    print(f"  - 总结: {len(train_data_summ)}条")
    print(f"  - 翻译:总结 = {len(train_data_en2zh)+len(train_data_zh2en)}:{len(train_data_summ)}")
    
    print(f"\n验证集 (val_v3.json): {len(val_v3)}条")
    print(f"  - en2zh翻译: {len(val_data_en2zh)}条")
    print(f"  - zh2en翻译: {len(val_data_zh2en)}条")
    print(f"  - 总结: {len(val_data_summ)}条")
    
    print(f"\n测试集 (test_v3.json): {len(test_v3)}条")
    print(f"  - en2zh翻译: {len(test_data_en2zh)}条")
    print(f"  - zh2en翻译: {len(test_data_zh2en)}条")
    print(f"  - 总结: {len(test_data_summ)}条")
    
    # 验证数据隔离
    train_keys = set(s['input'][:100] for s in train_v3)
    val_keys = set(s['input'][:100] for s in val_v3)
    test_keys = set(s['input'][:100] for s in test_v3)
    
    train_val_overlap = len(train_keys & val_keys)
    train_test_overlap = len(train_keys & test_keys)
    val_test_overlap = len(val_keys & test_keys)
    
    print(f"\n数据隔离验证:")
    print(f"  train-val重叠: {train_val_overlap}条")
    print(f"  train-test重叠: {train_test_overlap}条")
    print(f"  val-test重叠: {val_test_overlap}条")
    
    if train_val_overlap + train_test_overlap + val_test_overlap == 0:
        print("  ✓ 所有数据集完全隔离！")
    else:
        print("  ⚠ 存在数据泄漏，请检查！")
    
    return train_v3, val_v3, test_v3


if __name__ == "__main__":
    build_v3_dataset()
