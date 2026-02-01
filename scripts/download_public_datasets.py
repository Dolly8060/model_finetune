#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公开数据集下载与转换脚本
下载arXiv（总结）和WMT（翻译）数据集，转换为评估格式
"""

import os
import json
import argparse
import random
from typing import List, Dict
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("请先安装datasets库: pip install datasets")
    exit(1)


# 总结任务的多样化指令
SUMMARIZATION_INSTRUCTIONS = [
    "请总结以下计算机科学论文的核心内容。",
    "Summarize the key contributions of this paper.",
    "阅读下文，提炼其主要观点和方法。",
    "用简洁的语言概括这篇论文的主要贡献。",
    "Extract the main ideas from this research paper.",
    "请对以下学术论文进行摘要。",
]

# 翻译任务的多样化指令（英译中）
EN2ZH_INSTRUCTIONS = [
    "将以下英文内容翻译成中文。",
    "Translate the following text into Chinese.",
    "请把下面这段话翻译成中文。",
    "中文翻译：",
    "请将以下内容译为中文，保持原意。",
]

# 翻译任务的多样化指令（中译英）
ZH2EN_INSTRUCTIONS = [
    "将以下中文内容翻译成英文。",
    "Translate the following text into English.",
    "请把下面这段话翻译成英文。",
    "English translation:",
    "请将以下内容译为英文。",
]


def download_arxiv_summarization(max_samples: int = 200) -> List[Dict]:
    """下载arXiv论文总结数据集
    
    Args:
        max_samples: 最大样本数，0表示下载全部
    """
    print("下载arXiv总结数据集...")
    
    # 构建split字符串
    if max_samples > 0:
        split_str = f"test[:{max_samples}]"
    else:
        split_str = "test"
        print("  (下载全部测试集数据)")
    
    try:
        # 加载测试集
        dataset = load_dataset(
            "ccdv/arxiv-summarization", 
            split=split_str
        )
    except Exception as e:
        print(f"下载arXiv数据集失败: {e}")
        print("尝试使用备用方式...")
        # 备用：使用科学论文数据集
        try:
            dataset = load_dataset(
                "scientific_papers",
                "arxiv",
                split=split_str
            )
        except Exception as e2:
            print(f"备用方式也失败: {e2}")
            return []
    
    eval_data = []
    for item in tqdm(dataset, desc="处理arXiv数据"):
        # 获取文章和摘要
        article = item.get("article", item.get("document", ""))
        abstract = item.get("abstract", item.get("summary", ""))
        
        if not article or not abstract:
            continue
        
        # 截断过长的文章（保留前2000字符）
        if len(article) > 2000:
            article = article[:2000] + "..."
        
        eval_data.append({
            "instruction": random.choice(SUMMARIZATION_INSTRUCTIONS),
            "input": article,
            "output": abstract,
            "task_type": "summarization",
            "source": "arxiv"
        })
    
    print(f"成功处理 {len(eval_data)} 条arXiv总结数据")
    return eval_data


def download_wmt_translation(max_samples: int = 200) -> List[Dict]:
    """下载WMT翻译数据集（中英）
    
    Args:
        max_samples: 最大样本数，0表示下载全部
    """
    print("下载WMT翻译数据集...")
    
    eval_data = []
    dataset_sources = [
        # 方案1: Helsinki-NLP的OPUS数据集（最可靠）
        {
            "name": "Helsinki-NLP/opus-100",
            "config": "en-zh",
            "split": "test",
            "en_key": lambda x: x.get("translation", {}).get("en", ""),
            "zh_key": lambda x: x.get("translation", {}).get("zh", ""),
        },
        # 方案2: tatoeba 多语言翻译数据集
        {
            "name": "Helsinki-NLP/tatoeba_mt",
            "config": "eng-cmn_Hans",
            "split": "test",
            "en_key": lambda x: x.get("sourceString", ""),
            "zh_key": lambda x: x.get("targetString", ""),
        },
    ]
    
    for source in dataset_sources:
        if eval_data:  # 已有数据则跳过
            break
            
        try:
            print(f"  尝试加载 {source['name']} ({source['config']})...")
            
            # 构建split字符串
            base_split = source["split"]
            if max_samples > 0:
                split_str = f"{base_split}[:{max_samples}]"
            else:
                split_str = base_split
            
            dataset = load_dataset(
                source["name"],
                source["config"],
                split=split_str
            )
            
            for item in tqdm(dataset, desc=f"处理{source['name']}数据"):
                en = source["en_key"](item)
                zh = source["zh_key"](item)
                
                if not zh or not en:
                    continue
                
                # 随机选择翻译方向
                if random.random() < 0.5:
                    eval_data.append({
                        "instruction": random.choice(EN2ZH_INSTRUCTIONS),
                        "input": en,
                        "output": zh,
                        "task_type": "translation",
                        "direction": "en2zh",
                        "source": source["name"].split("/")[-1]
                    })
                else:
                    eval_data.append({
                        "instruction": random.choice(ZH2EN_INSTRUCTIONS),
                        "input": zh,
                        "output": en,
                        "task_type": "translation",
                        "direction": "zh2en",
                        "source": source["name"].split("/")[-1]
                    })
                    
            print(f"  成功从 {source['name']} 加载数据")
            
        except Exception as e:
            print(f"  {source['name']} 加载失败: {e}")
            continue
    
    if not eval_data:
        print("所有WMT/翻译数据源都失败，请检查网络连接")
    else:
        print(f"成功处理 {len(eval_data)} 条翻译数据")
    
    return eval_data


def download_flores_translation(max_samples: int = 100) -> List[Dict]:
    """下载翻译基准数据集（替代FLORES）
    
    由于facebook/flores和openlanguagedata/flores_plus不再直接可用，
    使用其他高质量翻译数据集作为替代。
    
    Args:
        max_samples: 最大样本数，0表示下载全部
    """
    print("下载翻译基准数据集...")
    
    eval_data = []
    
    # 备选数据集列表
    flores_alternatives = [
        # 方案1: WMT newstest数据集
        {
            "name": "wmt/wmt19",
            "config": "zh-en",
            "split": "validation",
            "en_key": lambda x: x.get("translation", {}).get("en", ""),
            "zh_key": lambda x: x.get("translation", {}).get("zh", ""),
        },
        # 方案2: ted_multi 多语言TED演讲
        {
            "name": "ted_multi",
            "config": None,
            "split": "test",
            "en_key": lambda x: get_ted_lang(x, "en"),
            "zh_key": lambda x: get_ted_lang(x, "zh"),
            "filter_fn": lambda x: "en" in x.get("translations", {}).get("language", []) and 
                                   "zh" in x.get("translations", {}).get("language", []),
        },
        # 方案3: Helsinki-NLP的opus_books
        {
            "name": "Helsinki-NLP/opus_books",
            "config": "en-zh",
            "split": "train",
            "en_key": lambda x: x.get("translation", {}).get("en", ""),
            "zh_key": lambda x: x.get("translation", {}).get("zh", ""),
        },
    ]
    
    for source in flores_alternatives:
        if eval_data:  # 已有数据则跳过
            break
        
        try:
            print(f"  尝试加载 {source['name']}...")
            
            # 构建split字符串
            base_split = source["split"]
            if max_samples > 0:
                split_str = f"{base_split}[:{max_samples}]"
            else:
                split_str = base_split
            
            if source["config"]:
                dataset = load_dataset(
                    source["name"],
                    source["config"],
                    split=split_str
                )
            else:
                dataset = load_dataset(
                    source["name"],
                    split=split_str
                )
            
            # 如果有过滤函数，先过滤
            if source.get("filter_fn"):
                dataset = [item for item in dataset if source["filter_fn"](item)]
                if max_samples > 0:
                    dataset = dataset[:max_samples]
            
            for item in tqdm(dataset, desc=f"处理{source['name']}数据"):
                en = source["en_key"](item)
                zh = source["zh_key"](item)
                
                if not en or not zh:
                    continue
                
                if random.random() < 0.5:
                    eval_data.append({
                        "instruction": random.choice(EN2ZH_INSTRUCTIONS),
                        "input": en,
                        "output": zh,
                        "task_type": "translation",
                        "direction": "en2zh",
                        "source": source["name"].split("/")[-1]
                    })
                else:
                    eval_data.append({
                        "instruction": random.choice(ZH2EN_INSTRUCTIONS),
                        "input": zh,
                        "output": en,
                        "task_type": "translation",
                        "direction": "zh2en",
                        "source": source["name"].split("/")[-1]
                    })
            
            print(f"  成功从 {source['name']} 加载数据")
            
        except Exception as e:
            print(f"  {source['name']} 加载失败: {e}")
            continue
    
    if not eval_data:
        print("所有翻译基准数据源都失败")
    else:
        print(f"成功处理 {len(eval_data)} 条翻译基准数据")
    
    return eval_data


def get_ted_lang(item, lang_code):
    """从ted_multi数据集中提取指定语言的文本"""
    translations = item.get("translations", {})
    languages = translations.get("language", [])
    texts = translations.get("translation", [])
    
    if lang_code in languages:
        idx = languages.index(lang_code)
        if idx < len(texts):
            return texts[idx]
    return ""


def main():
    parser = argparse.ArgumentParser(description="下载公开数据集用于评估")
    parser.add_argument(
        "--output", type=str, default="data/public_eval.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--arxiv-samples", type=int, default=500,
        help="arXiv总结样本数（0表示下载全部）"
    )
    parser.add_argument(
        "--translation-samples", type=int, default=500,
        help="翻译样本数（0表示下载全部）"
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "arxiv", "wmt", "flores"],
        help="要下载的数据集"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="下载所有数据（不限制样本数）"
    )
    args = parser.parse_args()
    
    # 如果指定--all，则不限制样本数
    if args.all:
        args.arxiv_samples = 0
        args.translation_samples = 0
    
    all_data = []
    
    # 下载数据集
    if args.dataset in ["all", "arxiv"]:
        arxiv_data = download_arxiv_summarization(args.arxiv_samples)
        all_data.extend(arxiv_data)
    
    if args.dataset in ["all", "wmt"]:
        wmt_data = download_wmt_translation(args.translation_samples)
        all_data.extend(wmt_data)
    
    if args.dataset in ["all", "flores"]:
        flores_data = download_flores_translation(args.translation_samples // 2)
        all_data.extend(flores_data)
    
    if not all_data:
        print("未能下载任何数据，请检查网络连接")
        return
    
    # 打乱数据
    random.shuffle(all_data)
    
    # 转换为alpaca格式（移除元信息）
    alpaca_data = []
    for item in all_data:
        alpaca_data.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"],
        })
    
    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    # 同时保存带元信息的版本
    meta_file = args.output.replace(".json", "_with_meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    # 统计
    task_stats = {}
    source_stats = {}
    for item in all_data:
        task = item.get("task_type", "unknown")
        source = item.get("source", "unknown")
        task_stats[task] = task_stats.get(task, 0) + 1
        source_stats[source] = source_stats.get(source, 0) + 1
    
    print("\n" + "=" * 50)
    print("数据集统计")
    print("=" * 50)
    print(f"总样本数: {len(all_data)}")
    print(f"\n任务分布:")
    for task, count in task_stats.items():
        print(f"  - {task}: {count}")
    print(f"\n数据来源:")
    for source, count in source_stats.items():
        print(f"  - {source}: {count}")
    print(f"\n已保存:")
    print(f"  - 评估数据: {args.output}")
    print(f"  - 带元信息: {meta_file}")


if __name__ == "__main__":
    main()
