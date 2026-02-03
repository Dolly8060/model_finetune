#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多语言指令遵循测试集构建脚本

构建用于评估指令遵循能力的多语言测试集：
- IFEval (英文): Google官方评估集
- Multi-IF (多语言): 英/法/西/日等
- 中文指令遵循模板

输出:
- test_ifeval_en.json: 英文IFEval测试集
- test_ifeval_multilingual.json: 多语言测试集 (含中文)
- test_ifeval_all.json: 合并的完整测试集
"""

import json
import os
import random
import argparse
from collections import defaultdict

random.seed(2026)


def load_json(filepath: str):
    """加载JSON文件"""
    if not os.path.exists(filepath):
        print(f"  警告: 文件不存在 {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def detect_language(text: str) -> str:
    """检测文本语言"""
    if not text:
        return "unknown"
    
    # 计算中文字符比例
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    chinese_ratio = chinese_chars / len(text)
    
    if chinese_ratio > 0.1:
        return "zh"
    
    # 检测其他语言特征
    if any(c in text for c in 'àâéèêëïîôùûüÿœæç'):
        return "fr"
    if any(c in text for c in 'áéíóúñü¿¡'):
        return "es"
    if any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in text):
        return "ja"
    
    return "en"


def build_ifeval_test_sets(
    data_dir: str = "data",
    output_dir: str = "data",
    english_samples: int = 300,
    multilingual_samples: int = 300
):
    """构建多语言指令遵循测试集
    
    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        english_samples: 英文测试集样本数
        multilingual_samples: 多语言测试集样本数
    """
    
    print("=" * 60)
    print("多语言指令遵循测试集构建")
    print("=" * 60)
    
    # 加载数据源
    sources = {
        'ifeval_eval': os.path.join(data_dir, 'ifeval_eval.json'),      # 英文IFEval
        'multi_if': os.path.join(data_dir, 'multi_if.json'),            # 多语言
        'chinese_templates': os.path.join(data_dir, 'ifeval_chinese_templates.json'),  # 中文
    }
    
    # 按语言分类收集数据
    data_by_lang = defaultdict(list)
    
    print("\n[1/3] 加载数据源...")
    
    for name, path in sources.items():
        data = load_json(path)
        if not data:
            continue
            
        print(f"\n  {name}: {len(data)}条")
        
        for item in data:
            instruction = item.get('instruction', '')
            
            # 获取语言标签
            lang = item.get('language', '')
            if not lang or lang == 'unknown':
                lang = detect_language(instruction)
            
            # 标准化语言代码
            lang = lang.lower()
            if lang in ['english', 'en']:
                lang = 'en'
            elif lang in ['chinese', 'zh']:
                lang = 'zh'
            elif lang in ['french', 'fr']:
                lang = 'fr'
            elif lang in ['spanish', 'es']:
                lang = 'es'
            elif lang in ['japanese', 'ja']:
                lang = 'ja'
            
            # 构建测试样本（无需output，评估时生成）
            test_sample = {
                'instruction': instruction,
                'input': item.get('input', ''),
                'output': '',  # 测试时由模型生成
                'language': lang,
                'source': name,
            }
            
            # 保留元数据用于评估
            if 'instruction_id_list' in item:
                test_sample['instruction_id_list'] = item['instruction_id_list']
            if 'kwargs' in item:
                test_sample['kwargs'] = item['kwargs']
            
            data_by_lang[lang].append(test_sample)
    
    # 统计
    print("\n[2/3] 语言分布统计...")
    for lang, samples in sorted(data_by_lang.items()):
        print(f"  {lang}: {len(samples)}条")
    
    # 构建测试集
    print("\n[3/3] 构建测试集...")
    
    # 英文测试集
    en_data = data_by_lang.get('en', [])
    random.shuffle(en_data)
    test_en = en_data[:english_samples]
    
    # 多语言测试集（非英文）
    multilingual_data = []
    for lang, samples in data_by_lang.items():
        if lang != 'en':
            multilingual_data.extend(samples)
    random.shuffle(multilingual_data)
    test_multilingual = multilingual_data[:multilingual_samples]
    
    # 合并测试集
    test_all = test_en + test_multilingual
    random.shuffle(test_all)
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    
    # 英文测试集
    en_path = os.path.join(output_dir, 'test_ifeval_en.json')
    with open(en_path, 'w', encoding='utf-8') as f:
        json.dump(test_en, f, ensure_ascii=False, indent=2)
    
    # 多语言测试集
    multi_path = os.path.join(output_dir, 'test_ifeval_multilingual.json')
    with open(multi_path, 'w', encoding='utf-8') as f:
        json.dump(test_multilingual, f, ensure_ascii=False, indent=2)
    
    # 合并测试集
    all_path = os.path.join(output_dir, 'test_ifeval_all.json')
    with open(all_path, 'w', encoding='utf-8') as f:
        json.dump(test_all, f, ensure_ascii=False, indent=2)
    
    # 统计输出
    print("\n" + "=" * 60)
    print("测试集构建完成！")
    print("=" * 60)
    
    print(f"\n英文测试集: {en_path}")
    print(f"  - 样本数: {len(test_en)}")
    
    print(f"\n多语言测试集: {multi_path}")
    print(f"  - 样本数: {len(test_multilingual)}")
    multi_lang_stats = defaultdict(int)
    for s in test_multilingual:
        multi_lang_stats[s['language']] += 1
    for lang, count in sorted(multi_lang_stats.items()):
        print(f"    - {lang}: {count}")
    
    print(f"\n合并测试集: {all_path}")
    print(f"  - 样本数: {len(test_all)}")
    
    return test_en, test_multilingual, test_all


def main():
    parser = argparse.ArgumentParser(description="构建多语言指令遵循测试集")
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="数据目录"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="输出目录"
    )
    parser.add_argument(
        "--english-samples", type=int, default=300,
        help="英文测试集样本数"
    )
    parser.add_argument(
        "--multilingual-samples", type=int, default=300,
        help="多语言测试集样本数"
    )
    args = parser.parse_args()
    
    build_ifeval_test_sets(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        english_samples=args.english_samples,
        multilingual_samples=args.multilingual_samples
    )


if __name__ == "__main__":
    main()
