#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练数据增强脚本
混合公开数据集 + AI生成数据，提升泛化能力
"""

import json
import random
import argparse
from typing import List, Dict


def load_json(file_path: str) -> List[Dict]:
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: List[Dict], file_path: str):
    """保存JSON数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def mix_datasets(
    ai_generated: List[Dict],
    public_data: List[Dict],
    mix_ratio: float = 0.3
) -> List[Dict]:
    """
    混合数据集
    
    Args:
        ai_generated: AI生成的数据
        public_data: 公开数据集
        mix_ratio: 公开数据占比 (0-1)
    
    Returns:
        混合后的数据集
    """
    ai_count = len(ai_generated)
    public_count = int(ai_count * mix_ratio / (1 - mix_ratio))
    
    # 如果公开数据不足，全部使用
    if public_count > len(public_data):
        public_count = len(public_data)
        print(f"⚠️ 公开数据不足，实际混合比例: {public_count / (ai_count + public_count):.2%}")
    
    # 随机采样公开数据
    random.seed(42)
    sampled_public = random.sample(public_data, public_count)
    
    # 合并并打乱
    mixed_data = ai_generated + sampled_public
    random.shuffle(mixed_data)
    
    return mixed_data


def main():
    parser = argparse.ArgumentParser(description="混合训练数据以提升泛化能力")
    parser.add_argument(
        "--ai-data", type=str, default="data/train.json",
        help="AI生成的训练数据"
    )
    parser.add_argument(
        "--public-data", type=str, default="data/public_eval.json",
        help="公开数据集"
    )
    parser.add_argument(
        "--output", type=str, default="data/train_mixed.json",
        help="输出文件"
    )
    parser.add_argument(
        "--mix-ratio", type=float, default=0.3,
        help="公开数据占比 (0.3 表示 AI:公开=7:3)"
    )
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载 AI 生成数据: {args.ai_data}")
    ai_data = load_json(args.ai_data)
    print(f"  样本数: {len(ai_data)}")
    
    print(f"加载公开数据集: {args.public_data}")
    public_data = load_json(args.public_data)
    print(f"  样本数: {len(public_data)}")
    
    # 混合数据
    print(f"\n混合数据 (目标比例: AI={1-args.mix_ratio:.0%}, 公开={args.mix_ratio:.0%})...")
    mixed_data = mix_datasets(ai_data, public_data, args.mix_ratio)
    
    # 保存
    save_json(mixed_data, args.output)
    
    print("\n" + "=" * 50)
    print("数据增强完成")
    print("=" * 50)
    print(f"原始 AI 数据: {len(ai_data)}")
    print(f"公开数据采样: {len(mixed_data) - len(ai_data)}")
    print(f"混合后总数: {len(mixed_data)}")
    print(f"实际混合比例: {(len(mixed_data) - len(ai_data)) / len(mixed_data):.1%}")
    print(f"\n已保存到: {args.output}")
    print("\n下一步:")
    print(f"  修改配置文件使用新数据集训练")
    print(f"  或运行: python run.py train --config configs/finetune_lora.yaml")


if __name__ == "__main__":
    main()
