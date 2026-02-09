#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建增强版测试集 test_v4_enhanced.json
将PMMEval-mifeval中英双语数据合并到现有test_v4.json中
"""

import json
import os
from typing import List, Dict


def convert_mifeval_to_standard(mifeval_path: str, lang: str) -> List[Dict]:
    """将PMMEval-mifeval格式转换为标准格式
    
    PMMEval格式:
    {
        "0": {"origin_prompt": [{"role": "HUMAN", "prompt": "..."}]},
        ...
    }
    
    标准格式:
    {
        "instruction": "...",
        "input": "",
        "output": "",  # 指令遵循评估不需要参考答案
        "task_type": "instruction_following",
        "language": "en" or "zh"
    }
    """
    with open(mifeval_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted = []
    for key, item in data.items():
        prompt_list = item.get('origin_prompt', [])
        if prompt_list:
            prompt_text = prompt_list[0].get('prompt', '')
            converted.append({
                "instruction": prompt_text,
                "input": "",
                "output": "",  # 指令遵循通过约束检查评估，不需要参考答案
                "task_type": "instruction_following",
                "language": lang,
                "source": f"PMMEval-mifeval-{lang}"
            })
    
    return converted


def add_task_type_to_existing(samples: List[Dict]) -> List[Dict]:
    """为现有样本添加task_type字段（如果没有）
    
    分类策略（优先级从高到低）：
    1. 翻译：关键词明确且误分类风险低
    2. 总结：关键词+模式匹配
    3. 指令遵循：必须有明确的格式/约束要求
    4. other：无法归类的样本，不强制归入IF
    
    核心改进：不再把 "other" 全部归入 instruction_following，
    而是通过约束检测来决定是否属于IF类。
    """
    import re
    
    # 翻译关键词（优先级最高，误分类风险低）
    translation_keywords = [
        "翻译", "translate", "translation", "译为", "译成",
        "english translation", "中文翻译", "中译英", "英译中",
        "help me translate", "translate this",
        "翻成中文", "翻成英文", "翻译成",
    ]
    
    # 总结关键词
    summary_keywords = [
        "总结", "摘要", "概括", "summary", "summarize",
        "提炼", "提取主要观点", "总结以下", "概括下文",
        "核心思想", "主要观点",
        "extract the main ideas", "main points of",
        "key points", "extract.*from.*text",
        "extract.*from.*paper", "extract.*from.*content",
    ]
    
    # 明确的指令遵循约束模式（必须是格式/结构要求，不是内容描述）
    # 这些正则比关键词更精确，减少误匹配
    instruction_following_patterns = [
        # 英文约束
        r"at least \d+\s+(?:words?|sentences?|paragraphs?|bullet|sections?)",
        r"at most \d+\s+(?:words?|sentences?|paragraphs?)",
        r"no more than \d+\s+(?:words?|sentences?|paragraphs?)",
        r"no less than \d+\s+(?:words?|sentences?|paragraphs?)",
        r"exactly \d+\s+(?:words?|sentences?|paragraphs?|sections?)",
        r"must (?:include|contain)\s+['\"]",
        r"do not (?:use|include)\s+['\"]",
        r"avoid (?:using)?\s+['\"]",
        r"(?:your )?(?:entire )?response (?:should|must)",
        r"your answer (?:should|must)",
        r"(?:in |use )?(?:json|markdown) format",
        r"(?:start|begin|end) with\s+['\"]",
        r"bullet point",
        r"numbered list",
        r"(?:wrapped|enclosed) in",
        r"separated (?:with|by)",
        r"highlight.*\d+.*section",
        r"(?:all|entire).*(?:capital|uppercase|lowercase)",
        r"postscript|P\.S\.",
        r"(?:double )?quotation marks",
        r"\d+\s+asterisk",
        # 中文约束
        r"不少于\d+",
        r"不超过\d+",
        r"至少\d+(?:个)?(?:字|词|句|段)",
        r"最多\d+",
        r"恰好\d+(?:个)?(?:字|词)",
        r"字数要求",
        r"必须包含(?:以下)?关键词",
        r"不要使用['\"\u2018\u2019\u201c\u201d]",
        r"禁止使用",
        r"(?:请)?用列表形式",
        r"以?编号",
        r"三段式",
        r"以问句结尾",
        r"开头(?:必须)?[为是]",
        r"每段(?:都)?(?:必须)?包含",
        r"以问答形式",
        r"不要使用.*(?:形容词|副词|程度)",
        r"用(?:Markdown|markdown)格式",
    ]
    
    for sample in samples:
        if sample.get('task_type') and sample['task_type'] != 'unknown':
            continue
            
        instr = sample.get('instruction', '')
        instr_lower = instr.lower()
        
        # Step 1: 检查是否有明确的格式约束 → instruction_following
        has_constraint = any(
            re.search(pat, instr, re.IGNORECASE)
            for pat in instruction_following_patterns
        )
        
        if has_constraint:
            sample['task_type'] = 'instruction_following'
            continue
        
        # Step 2: 检查翻译
        if any(kw in instr_lower for kw in translation_keywords):
            sample['task_type'] = 'translation'
            continue
        
        # Step 3: 检查总结（包括正则模式）
        is_summary = any(kw in instr_lower for kw in summary_keywords)
        if not is_summary:
            is_summary = bool(re.search(
                r"extract.*(?:main|key).*(?:ideas?|points?)|"
                r"(?:main|key)\s+(?:ideas?|points?)\s+(?:of|from)",
                instr_lower
            ))
        if is_summary:
            sample['task_type'] = 'summarization'
            continue
        
        # Step 4: 无法归类 → other（不再强制归入IF）
        sample['task_type'] = 'other'
    
    return samples


# 支持的语言列表（PMMEval-mifeval全部11种语言）
SUPPORTED_LANGUAGES = [
    ('en', '英文'),
    ('zh', '中文'),
    ('de', '德文'),
    ('fr', '法文'),
    ('es', '西班牙文'),
    ('it', '意大利文'),
    ('ja', '日文'),
    ('pt', '葡萄牙文'),
    ('pl', '波兰文'),
    ('ro', '罗马尼亚文'),
    ('ar', '阿拉伯文'),
]


def main():
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp_dir = os.path.join(data_dir, 'tmp')
    output_dir = os.path.join(data_dir, 'data')
    
    # 1. 加载现有test_v4.json
    test_v4_path = os.path.join(output_dir, 'test_v4.json')
    print(f"加载现有测试集: {test_v4_path}")
    with open(test_v4_path, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    print(f"  现有样本数: {len(existing_data)}")
    
    # 2. 为现有样本添加task_type
    existing_data = add_task_type_to_existing(existing_data)
    
    # 统计现有任务分布
    task_counts = {}
    for s in existing_data:
        t = s.get('task_type', 'unknown')
        task_counts[t] = task_counts.get(t, 0) + 1
    print(f"  现有任务分布: {task_counts}")
    
    # 3. 转换全部11种语言的PMMEval-mifeval数据
    print(f"\n转换全部M-IFEval数据（11种语言）:")
    all_mifeval = []
    lang_stats = {}
    
    for lang_code, lang_name in SUPPORTED_LANGUAGES:
        mifeval_path = os.path.join(tmp_dir, f'PMMEval-mifeval-{lang_code}.json')
        if os.path.exists(mifeval_path):
            converted = convert_mifeval_to_standard(mifeval_path, lang_code)
            all_mifeval.extend(converted)
            lang_stats[lang_name] = len(converted)
            print(f"  {lang_name} ({lang_code}): {len(converted)}条")
        else:
            print(f"  {lang_name} ({lang_code}): 文件不存在，跳过")
    
    print(f"  M-IFEval总计: {len(all_mifeval)}条")
    
    # 4. 合并数据
    # "other"类别保持不变，不再强制归入指令遵循
    # 这些样本在评估时会被单独统计
    enhanced_data = existing_data + all_mifeval
    print(f"\n合并后总样本数: {len(enhanced_data)}")
    
    # 统计最终任务分布
    final_task_counts = {}
    for s in enhanced_data:
        t = s.get('task_type', 'unknown')
        final_task_counts[t] = final_task_counts.get(t, 0) + 1
    print(f"最终任务分布: {final_task_counts}")
    
    # 5. 保存增强版测试集
    enhanced_path = os.path.join(output_dir, 'test_v4_enhanced.json')
    with open(enhanced_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    print(f"\n增强版测试集已保存: {enhanced_path}")
    
    # 6. 同时生成纯M-IFEval测试集（用于单独评估）
    mifeval_only_path = os.path.join(output_dir, 'test_mifeval.json')
    with open(mifeval_only_path, 'w', encoding='utf-8') as f:
        json.dump(all_mifeval, f, ensure_ascii=False, indent=2)
    print(f"纯M-IFEval测试集已保存: {mifeval_only_path} ({len(all_mifeval)}条)")
    
    # 7. 按语言分组统计
    print("\n" + "="*60)
    print("数据集构建完成！")
    print("="*60)
    print(f"\ntest_v4_enhanced.json: {len(enhanced_data)}条")
    print(f"  - 翻译: {final_task_counts.get('translation', 0)}条")
    print(f"  - 总结: {final_task_counts.get('summarization', 0)}条")
    print(f"  - 指令遵循: {final_task_counts.get('instruction_following', 0)}条")
    if final_task_counts.get('other', 0) > 0:
        print(f"  - 其他: {final_task_counts.get('other', 0)}条")
    print(f"\nM-IFEval语言分布:")
    for lang_name, count in lang_stats.items():
        print(f"  - {lang_name}: {count}条")


if __name__ == "__main__":
    main()
