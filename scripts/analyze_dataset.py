#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据集分析脚本"""

import json
import sys
from collections import Counter

def analyze_dataset(file_path):
    """分析数据集构成"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 60)
    print(f"数据集分析：{file_path}")
    print("=" * 60)
    print(f"\n总样本数：{len(data)}\n")
    
    # 分析指令分布
    instructions = [item['instruction'] for item in data]
    instruction_counter = Counter(instructions)
    
    print("指令类型分布（Top 10）：")
    print("-" * 60)
    for i, (inst, count) in enumerate(instruction_counter.most_common(10), 1):
        percentage = count / len(data) * 100
        inst_short = inst[:40] + "..." if len(inst) > 40 else inst
        print(f"{i:2}. [{count:4}条 {percentage:5.1f}%] {inst_short}")
    
    # 分析任务类型（根据指令关键词推断）
    print("\n任务类型推断：")
    print("-" * 60)
    
    translation_keywords = ['翻译', 'translate', 'translation', '译为', '译成']
    summary_keywords = ['总结', 'summarize', '摘要', '概括', 'summary']
    
    translation_count = 0
    summary_count = 0
    other_count = 0
    
    for inst in instructions:
        inst_lower = inst.lower()
        if any(kw in inst_lower for kw in translation_keywords):
            translation_count += 1
        elif any(kw in inst_lower for kw in summary_keywords):
            summary_count += 1
        else:
            other_count += 1
    
    print(f"翻译任务：{translation_count:4}条 ({translation_count/len(data)*100:5.1f}%)")
    print(f"总结任务：{summary_count:4}条 ({summary_count/len(data)*100:5.1f}%)")
    print(f"其他任务：{other_count:4}条 ({other_count/len(data)*100:5.1f}%)")
    
    # 分析数据长度
    print("\n数据长度统计：")
    print("-" * 60)
    
    input_lengths = [len(item['input']) for item in data]
    output_lengths = [len(item['output']) for item in data]
    
    print(f"输入长度：")
    print(f"  平均值：{sum(input_lengths)/len(input_lengths):.0f} 字符")
    print(f"  最小值：{min(input_lengths)} 字符")
    print(f"  最大值：{max(input_lengths)} 字符")
    
    print(f"\n输出长度：")
    print(f"  平均值：{sum(output_lengths)/len(output_lengths):.0f} 字符")
    print(f"  最小值：{min(output_lengths)} 字符")
    print(f"  最大值：{max(output_lengths)} 字符")
    
    # 示例展示
    print("\n数据样例（前3条）：")
    print("=" * 60)
    for i, item in enumerate(data[:3], 1):
        print(f"\n样例 {i}:")
        print(f"指令：{item['instruction']}")
        print(f"输入：{item['input'][:100]}..." if len(item['input']) > 100 else f"输入：{item['input']}")
        print(f"输出：{item['output'][:100]}..." if len(item['output']) > 100 else f"输出：{item['output']}")
        print("-" * 60)

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/train_mixed_3k.json"
    analyze_dataset(file_path)
