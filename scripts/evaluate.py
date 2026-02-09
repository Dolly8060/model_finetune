#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估脚本 - 对比微调前后以及不同规模模型在翻译、总结和指令遵循任务上的表现
评估指标：BLEU、ROUGE、BERTScore（业界标准）+ 指令遵循率（IFR）

支持功能：
1. 基础模型 vs 微调模型对比
2. 多模型横向对比（如1B vs 7B vs 8B）
3. 指令遵循能力评估（v6新增）
"""

import os
import json
import argparse
import random
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from dotenv import load_dotenv

# 评估指标
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score
import jieba

load_dotenv()


# ============================================================
# 指令遵循评估器（v6新增）
# ============================================================

def _count_words(text: str) -> int:
    """智能词数计算：中文使用jieba分词，其他语言使用空格分词
    
    Args:
        text: 输入文本
        
    Returns:
        词数
    """
    # 检测是否包含中文/日文字符
    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff')
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    
    if total_chars > 0 and cjk_chars / total_chars > 0.3:
        # 中文/日文：使用jieba分词
        words = list(jieba.cut(text))
        # 过滤掉空白和标点
        words = [w for w in words if w.strip() and not all(c in '，。！？、；：""''（）【】《》' for c in w)]
        return len(words)
    else:
        # 其他语言：使用空格分词
        return len(text.split())


def _count_sentences(text: str) -> int:
    """智能句子数计算：支持多语言句子结尾符号
    
    Args:
        text: 输入文本
        
    Returns:
        句子数
    """
    # 多语言句子结尾符号
    sentence_endings = [
        r'[.!?](?:\s|$)',           # 英文等
        r'[。！？]',                  # 中文/日文
        r'[।॥]',                     # 印地语
        r'[؟!.]',                    # 阿拉伯文
        r'[.!?。！？]\s*[\'"）\)」』】]?\s*$',  # 带引号的结尾
    ]
    
    count = 0
    for pattern in sentence_endings:
        count += len(re.findall(pattern, text))
    
    # 去重：同一位置可能被多个模式匹配
    # 简化处理：返回最大匹配数
    return max(count // 2, 1) if count > 0 else 1


def _check_language(text: str, expected_lang: str) -> bool:
    """检查文本是否主要使用指定语言
    
    Args:
        text: 输入文本
        expected_lang: 期望的语言代码（zh, en, fr, de, es, it, ja, pt, pl, ro, ar等）
        
    Returns:
        是否符合语言要求
    """
    if not text or not expected_lang:
        return True
    
    expected_lang = expected_lang.lower().strip()
    
    # 语言特征检测
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    if total_chars == 0:
        return True
    
    # 中文字符
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    # 日文假名
    japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u30ff' or '\u31f0' <= c <= '\u31ff')
    # 韩文
    korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
    # 阿拉伯文
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06ff')
    # 西里尔字母（俄语等）
    cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
    # 拉丁字符（含重音）
    latin_chars = sum(1 for c in text if ('a' <= c.lower() <= 'z') or ('\u00c0' <= c <= '\u024f'))
    
    # 语言判断阈值（宽松：30%）
    threshold = 0.25
    
    if expected_lang in ('zh', 'chinese', '中文'):
        return chinese_chars / total_chars >= threshold
    elif expected_lang in ('ja', 'japanese', '日文', '日语'):
        return (japanese_chars + chinese_chars) / total_chars >= threshold and japanese_chars > 0
    elif expected_lang in ('ko', 'korean', '韩文', '韩语'):
        return korean_chars / total_chars >= threshold
    elif expected_lang in ('ar', 'arabic', '阿拉伯文'):
        return arabic_chars / total_chars >= threshold
    elif expected_lang in ('ru', 'russian', '俄文', '俄语'):
        return cyrillic_chars / total_chars >= threshold
    elif expected_lang in ('en', 'english', '英文', '英语'):
        # 英语：拉丁字符为主，且中文日文韩文阿拉伯文很少
        asian_chars = chinese_chars + japanese_chars + korean_chars + arabic_chars
        return latin_chars / total_chars >= threshold and asian_chars / total_chars < 0.1
    elif expected_lang in ('fr', 'french', '法文', '法语',
                           'de', 'german', '德文', '德语',
                           'es', 'spanish', '西班牙文', '西班牙语',
                           'it', 'italian', '意大利文', '意大利语',
                           'pt', 'portuguese', '葡萄牙文', '葡萄牙语',
                           'pl', 'polish', '波兰文', '波兰语',
                           'ro', 'romanian', '罗马尼亚文', '罗马尼亚语'):
        # 欧洲语言：拉丁字符为主
        asian_chars = chinese_chars + japanese_chars + korean_chars + arabic_chars
        return latin_chars / total_chars >= threshold and asian_chars / total_chars < 0.1
    else:
        # 未知语言，默认通过
        return True


class InstructionFollowingEvaluator:
    """指令遵循评估器 - 评估模型对可验证指令的遵循能力"""
    
    # 可验证指令类型及对应的检测函数
    VERIFIABLE_INSTRUCTIONS = {
        # ============================================================
        # 语言约束（v6新增 - 非常重要！）
        # ============================================================
        "response_language": {
            "patterns": [
                # 英文指令
                r"(?:your )?(?:ENTIRE )?response (?:should|must) be in (\w+)(?: language)?",
                r"answer (?:should|must) be (?:entirely )?in (\w+)",
                r"write (?:your )?(?:entire )?(?:response|answer) in (\w+)",
                r"respond (?:entirely )?in (\w+)",
                r"use only (\w+)(?: language)?",
                r"no other language.*(?:only|use) (\w+)",
                # 中文指令
                r"用(\w+)回答",
                r"使用(\w+)语言",
                r"回答必须是(\w+)",
            ],
            "check": lambda text, lang: _check_language(text, lang),
        },
        
        # ============================================================
        # 句子数约束（v6新增）
        # ============================================================
        "min_sentences": {
            "patterns": [
                # 英文
                r"at least (\d+) sentences?",
                r"no (?:less|fewer) than (\d+) sentences?",
                r"minimum (?:of )?(\d+) sentences?",
                r"(\d+) sentences? or more",
                # 中文
                r"至少(\d+)(?:个)?句",
                r"不少于(\d+)(?:个)?句",
                # 其他语言
                r"au moins (\d+) phrases?",  # 法文
                r"al menos (\d+) oraciones?",  # 西班牙文
                r"mindestens (\d+) [Ss]ätze?",  # 德文
            ],
            "check": lambda text, param: _count_sentences(text) >= int(param),
        },
        "max_sentences": {
            "patterns": [
                # 英文
                r"at most (\d+) sentences?",
                r"no more than (\d+) sentences?",
                r"maximum (?:of )?(\d+) sentences?",
                r"(\d+) sentences? or (?:less|fewer)",
                # 中文
                r"最多(\d+)(?:个)?句",
                r"不超过(\d+)(?:个)?句",
            ],
            "check": lambda text, param: _count_sentences(text) <= int(param),
        },
        
        # ============================================================
        # 标题格式约束（v6新增）
        # ============================================================
        "title_double_brackets": {
            "patterns": [
                # 英文
                r"title.*(?:wrapped|enclosed) in double angular brackets",
                r"contain(?:s)? a title.*double angular brackets",
                r"answer must contain a title.*<<.*>>",
                r"title.*such as <<.*>>",
                # 中文
                r"标题.*用双尖括号",
                r"标题.*<<.*>>",
            ],
            "check": lambda text, _: bool(re.search(r'<<[^<>]+>>', text)),
        },
        "title_square_brackets": {
            "patterns": [
                r"title.*\[.*\]",
                r"title in square brackets",
            ],
            "check": lambda text, _: bool(re.search(r'\[[^\[\]]+\]', text.split('\n')[0] if text else '')),
        },
        
        # ============================================================
        # 段落分隔符约束（v6新增）
        # ============================================================
        "paragraph_divider": {
            "patterns": [
                # 英文
                r"paragraphs? (?:are |should be )?separated (?:with|by) (?:the )?(?:markdown )?divider[:\s]*\*\*\*",
                r"use \*\*\* (?:as |to |for )?(?:the )?(?:paragraph )?divider",
                r"separate paragraphs? (?:with|using) \*\*\*",
                # 中文
                r"段落.*用.*\*\*\*.*分隔",
                r"使用\*\*\*分隔段落",
            ],
            "check": lambda text, _: '***' in text,
        },
        
        # ============================================================
        # 词频约束（v6增强）
        # ============================================================
        "word_frequency": {
            "patterns": [
                # 英文 - 多种表达方式
                r"(?:the )?word ['\"]?(\w+)['\"]? should appear (?:at least )?(\d+) times?",
                r"['\"]?(\w+)['\"]? (?:should |must )?appear(?:s)? (?:at least )?(\d+) times?",
                r"use (?:the )?word ['\"]?(\w+)['\"]? (?:at least )?(\d+) times?",
                r"include ['\"]?(\w+)['\"]? (?:at least )?(\d+) times?",
                r"['\"]?(\w+)['\"]?.*at least (\d+) times?",
                # 中文
                r"['\"]?(\w+)['\"]?.*(?:出现|使用).*(?:至少)?(\d+)次",
                r"(?:词|字)['\"]?(\w+)['\"]?.*(\d+)次",
            ],
            "check": lambda text, params: text.lower().count(params[0].lower()) >= int(params[1]) if isinstance(params, tuple) and len(params) >= 2 else True,
        },
        
        # ============================================================
        # 字数约束（原有）
        # ============================================================
        "min_words": {
            "patterns": [
                # 英文
                r"at least (\d+) words",
                r"no less than (\d+) words", 
                r"minimum (?:of )?(\d+) words",
                r"(\d+) words or more",
                # 中文
                r"不少于(\d+)个?字",
                r"至少(\d+)个?字",
                r"最少(\d+)个?字",
                # 德文
                r"mindestens (\d+) [Ww]örter",
                r"wenigstens (\d+) [Ww]örter",
                # 法文
                r"au moins (\d+) mots",
                r"minimum (\d+) mots",
                # 西班牙文
                r"al menos (\d+) palabras",
                r"mínimo (?:de )?(\d+) palabras",
                # 意大利文
                r"almeno (\d+) parole",
                r"minimo (\d+) parole",
                # 日文
                r"少なくとも(\d+)語",
                r"(\d+)語以上",
                # 葡萄牙文
                r"pelo menos (\d+) palavras",
                r"mínimo (?:de )?(\d+) palavras",
                # 波兰文
                r"co najmniej (\d+) słów",
                r"minimum (\d+) słów",
                # 罗马尼亚文
                r"cel puțin (\d+) cuvinte",
                r"minim (\d+) cuvinte",
                # 阿拉伯文
                r"على الأقل (\d+) كلم[ةات]",
                r"لا يقل عن (\d+) كلم[ةات]",
            ],
            "check": lambda text, param: _count_words(text) >= int(param),
        },
        "max_words": {
            "patterns": [
                # 英文
                r"at most (\d+) words",
                r"no more than (\d+) words",
                r"maximum (?:of )?(\d+) words",
                r"(\d+) words or (?:less|fewer)",
                r"not exceed (\d+) words",
                # 中文
                r"不超过(\d+)个?字",
                r"最多(\d+)个?字",
                r"不多于(\d+)个?字",
                # 德文
                r"höchstens (\d+) [Ww]örter",
                r"maximal (\d+) [Ww]örter",
                r"nicht mehr als (\d+) [Ww]örter",
                # 法文
                r"au plus (\d+) mots",
                r"maximum (\d+) mots",
                r"pas plus de (\d+) mots",
                # 西班牙文
                r"como máximo (\d+) palabras",
                r"máximo (?:de )?(\d+) palabras",
                r"no más de (\d+) palabras",
                # 意大利文
                r"al massimo (\d+) parole",
                r"massimo (\d+) parole",
                r"non più di (\d+) parole",
                # 日文
                r"最大(\d+)語",
                r"(\d+)語以下",
                r"(\d+)語を超えない",
                # 葡萄牙文
                r"no máximo (\d+) palavras",
                r"máximo (?:de )?(\d+) palavras",
                r"não mais (?:de |que )(\d+) palavras",
                # 波兰文
                r"nie więcej niż (\d+) słów",
                r"maksymalnie (\d+) słów",
                r"co najwyżej (\d+) słów",
                # 罗马尼亚文
                r"cel mult (\d+) cuvinte",
                r"maxim (\d+) cuvinte",
                # 阿拉伯文
                r"على الأكثر (\d+) كلم[ةات]",
                r"لا يزيد عن (\d+) كلم[ةات]",
            ],
            "check": lambda text, param: _count_words(text) <= int(param),
        },
        "exact_words": {
            "patterns": [
                # 英文
                r"exactly (\d+) words",
                r"precisely (\d+) words",
                # 中文
                r"恰好(\d+)个?字",
                r"正好(\d+)个?字",
                # 德文
                r"genau (\d+) [Ww]örter",
                r"exakt (\d+) [Ww]örter",
                # 法文
                r"exactement (\d+) mots",
                # 西班牙文
                r"exactamente (\d+) palabras",
                # 意大利文
                r"esattamente (\d+) parole",
                # 日文
                r"ちょうど(\d+)語",
                r"正確に(\d+)語",
                # 葡萄牙文
                r"exatamente (\d+) palavras",
                # 波兰文
                r"dokładnie (\d+) słów",
                # 罗马尼亚文
                r"exact (\d+) cuvinte",
                # 阿拉伯文
                r"بالضبط (\d+) كلم[ةات]",
            ],
            "check": lambda text, param: abs(_count_words(text) - int(param)) <= 5,
        },
        
        # 段落约束
        "min_paragraphs": {
            "patterns": [
                # 英文
                r"at least (\d+) paragraphs?",
                r"minimum (?:of )?(\d+) paragraphs?",
                # 中文
                r"至少(\d+)段",
                r"不少于(\d+)段",
                # 德文
                r"mindestens (\d+) [Aa]bsätze?",
                # 法文
                r"au moins (\d+) paragraphes?",
                # 西班牙文
                r"al menos (\d+) párrafos?",
                # 意大利文
                r"almeno (\d+) paragrafi?",
                # 日文
                r"少なくとも(\d+)段落",
                # 葡萄牙文
                r"pelo menos (\d+) parágrafos?",
                # 波兰文
                r"co najmniej (\d+) akapitów",
                # 罗马尼亚文
                r"cel puțin (\d+) paragrafe?",
                # 阿拉伯文
                r"على الأقل (\d+) فقر[ةات]",
            ],
            "check": lambda text, param: text.count('\n\n') + 1 >= int(param) or text.count('\n') + 1 >= int(param),
        },
        "exact_paragraphs": {
            "patterns": [
                # 英文
                r"exactly (\d+) paragraphs?",
                r"write (\d+) paragraphs?",
                r"(\d+) paragraphs? exactly",
                r"(?:must |should )?have (\d+) paragraphs?",
                r"response (?:must |should )?have (\d+) paragraphs?",
                r"(?:must |should )?contain (\d+) paragraphs?",
                # 中文
                r"写(\d+)段",
                r"分(\d+)段",
                r"恰好(\d+)段",
                r"必须(?:有|包含)(\d+)段",
                # 德文
                r"genau (\d+) [Aa]bsätze?",
                # 法文
                r"exactement (\d+) paragraphes?",
                # 西班牙文
                r"exactamente (\d+) párrafos?",
                # 意大利文
                r"esattamente (\d+) paragrafi?",
                # 日文
                r"ちょうど(\d+)段落",
                # 葡萄牙文
                r"exatamente (\d+) parágrafos?",
                # 波兰文
                r"dokładnie (\d+) akapitów",
                # 罗马尼亚文
                r"exact (\d+) paragrafe?",
                # 阿拉伯文
                r"بالضبط (\d+) فقر[ةات]",
            ],
            "check": lambda text, param: abs((text.count('\n\n') + 1) - int(param)) <= 1,
        },
        
        # 关键词包含
        "keyword_include": {
            "patterns": [
                # 英文
                r"must include ['\"]([^'\"]+)['\"]",
                r"include the (?:word|keyword|phrase) ['\"]([^'\"]+)['\"]",
                r"must contain ['\"]([^'\"]+)['\"]",
                r"make sure to include ['\"]([^'\"]+)['\"]",
                # 中文
                r"必须包含['\"]([^'\"]+)['\"]",
                r"包含关键词['\"]([^'\"]+)['\"]",
                r"要包含['\"]([^'\"]+)['\"]",
                # 德文
                r"muss ['\"]([^'\"]+)['\"] enthalten",
                r"enthält ['\"]([^'\"]+)['\"]",
                # 法文
                r"doit inclure ['\"]([^'\"]+)['\"]",
                r"doit contenir ['\"]([^'\"]+)['\"]",
                # 西班牙文
                r"debe incluir ['\"]([^'\"]+)['\"]",
                r"debe contener ['\"]([^'\"]+)['\"]",
                # 意大利文
                r"deve includere ['\"]([^'\"]+)['\"]",
                r"deve contenere ['\"]([^'\"]+)['\"]",
                # 日文
                r"['\"]([^'\"]+)['\"]を含める",
                r"['\"]([^'\"]+)['\"]を入れる",
                # 葡萄牙文
                r"deve incluir ['\"]([^'\"]+)['\"]",
                r"deve conter ['\"]([^'\"]+)['\"]",
                # 波兰文
                r"musi zawierać ['\"]([^'\"]+)['\"]",
                # 罗马尼亚文
                r"trebuie să includă ['\"]([^'\"]+)['\"]",
                # 阿拉伯文
                r"يجب أن يتضمن ['\"]([^'\"]+)['\"]",
            ],
            "check": lambda text, param: param.lower() in text.lower(),
        },
        "keyword_count": {
            "patterns": [
                # 英文
                r"['\"]([^'\"]+)['\"].*at least (\d+) times",
                r"['\"]([^'\"]+)['\"].*(?:appear|occur).*?(\d+) times",
                r"use ['\"]([^'\"]+)['\"].*?(\d+) times",
                # 中文
                r"['\"]([^'\"]+)['\"].*出现.*(\d+)次",
                r"包含['\"]([^'\"]+)['\"](\d+)次",
                # 德文
                r"['\"]([^'\"]+)['\"].*mindestens (\d+) [Mm]al",
                # 法文
                r"['\"]([^'\"]+)['\"].*au moins (\d+) fois",
                # 西班牙文
                r"['\"]([^'\"]+)['\"].*al menos (\d+) veces",
                # 意大利文
                r"['\"]([^'\"]+)['\"].*almeno (\d+) volte",
                # 日文
                r"['\"]([^'\"]+)['\"].*少なくとも(\d+)回",
                # 葡萄牙文
                r"['\"]([^'\"]+)['\"].*pelo menos (\d+) vezes",
                # 波兰文
                r"['\"]([^'\"]+)['\"].*co najmniej (\d+) razy",
                # 罗马尼亚文
                r"['\"]([^'\"]+)['\"].*cel puțin (\d+) ori",
            ],
            "check": lambda text, params: text.lower().count(params[0].lower()) >= int(params[1]) if isinstance(params, tuple) else True,
        },
        
        # 排除约束
        "keyword_exclude": {
            "patterns": [
                # 英文
                r"do not use ['\"]([^'\"]+)['\"]",
                r"avoid (?:using )?['\"]([^'\"]+)['\"]",
                r"without ['\"]([^'\"]+)['\"]",
                r"don't use ['\"]([^'\"]+)['\"]",
                r"must not (?:use|include|contain) ['\"]([^'\"]+)['\"]",
                # 中文
                r"不要使用['\"]([^'\"]+)['\"]",
                r"禁止使用['\"]([^'\"]+)['\"]",
                r"不能包含['\"]([^'\"]+)['\"]",
                r"避免['\"]([^'\"]+)['\"]",
                # 德文
                r"verwende nicht ['\"]([^'\"]+)['\"]",
                r"vermeide ['\"]([^'\"]+)['\"]",
                r"ohne ['\"]([^'\"]+)['\"]",
                # 法文
                r"n'utilisez pas ['\"]([^'\"]+)['\"]",
                r"évitez ['\"]([^'\"]+)['\"]",
                r"sans ['\"]([^'\"]+)['\"]",
                # 西班牙文
                r"no uses ['\"]([^'\"]+)['\"]",
                r"evita ['\"]([^'\"]+)['\"]",
                r"sin ['\"]([^'\"]+)['\"]",
                # 意大利文
                r"non usare ['\"]([^'\"]+)['\"]",
                r"senza ['\"]([^'\"]+)['\"]",
                # 日文
                r"['\"]([^'\"]+)['\"]を使わない",
                r"['\"]([^'\"]+)['\"]を避ける",
                # 葡萄牙文
                r"não use ['\"]([^'\"]+)['\"]",
                r"evite ['\"]([^'\"]+)['\"]",
                r"sem ['\"]([^'\"]+)['\"]",
                # 波兰文
                r"nie używaj ['\"]([^'\"]+)['\"]",
                r"unikaj ['\"]([^'\"]+)['\"]",
                # 罗马尼亚文
                r"nu folosi ['\"]([^'\"]+)['\"]",
                r"evită ['\"]([^'\"]+)['\"]",
                # 阿拉伯文
                r"لا تستخدم ['\"]([^'\"]+)['\"]",
                r"تجنب ['\"]([^'\"]+)['\"]",
            ],
            "check": lambda text, param: param.lower() not in text.lower(),
        },
        
        # 格式约束
        "bullet_points": {
            "patterns": [
                # 英文
                r"use bullet points",
                r"in bullet point format",
                r"as a bulleted list",
                r"bullet list",
                # 中文
                r"使用列表",
                r"列表形式",
                r"项目符号",
                # 德文
                r"[Aa]ufzählungszeichen",
                r"[Ss]tichpunkte",
                r"als Liste",
                # 法文
                r"puces",
                r"liste à puces",
                # 西班牙文
                r"viñetas",
                r"lista con viñetas",
                # 意大利文
                r"elenco puntato",
                r"punti elenco",
                # 日文
                r"箇条書き",
                r"リスト形式",
                # 葡萄牙文
                r"marcadores",
                r"lista com marcadores",
                # 波兰文
                r"punktory",
                r"lista punktowana",
                # 罗马尼亚文
                r"marcatori",
                r"listă cu marcatori",
                # 阿拉伯文
                r"نقاط نقطية",
                r"قائمة نقطية",
            ],
            "check": lambda text, _: bool(re.search(r'[-•*●◦‣]\s', text)) or bool(re.search(r'^\s*[\-\*•]\s', text, re.MULTILINE)),
        },
        "numbered_list": {
            "patterns": [
                # 英文
                r"use numbered list",
                r"numbered format",
                r"as a numbered list",
                # 中文
                r"使用编号",
                r"编号列表",
                r"数字列表",
                # 德文
                r"nummerierte Liste",
                r"mit Nummern",
                # 法文
                r"liste numérotée",
                r"format numéroté",
                # 西班牙文
                r"lista numerada",
                r"formato numerado",
                # 意大利文
                r"elenco numerato",
                r"lista numerata",
                # 日文
                r"番号付きリスト",
                r"数字付き",
                # 葡萄牙文
                r"lista numerada",
                # 波兰文
                r"lista numerowana",
                # 罗马尼亚文
                r"listă numerotată",
                # 阿拉伯文
                r"قائمة مرقمة",
            ],
            "check": lambda text, _: bool(re.search(r'^\s*\d+[.、)\]]\s', text, re.MULTILINE)),
        },
        "json_format": {
            "patterns": [
                # 英文
                r"in json format",
                r"as json",
                r"json output",
                r"output.*json",
                # 中文
                r"json格式",
                r"JSON格式",
                # 德文
                r"im JSON-Format",
                r"als JSON",
                # 法文
                r"en format JSON",
                r"au format JSON",
                # 西班牙文
                r"en formato JSON",
                r"como JSON",
                # 意大利文
                r"in formato JSON",
                r"come JSON",
                # 日文
                r"JSON形式",
                r"JSONフォーマット",
                # 葡萄牙文
                r"em formato JSON",
                # 波兰文
                r"w formacie JSON",
                # 罗马尼亚文
                r"în format JSON",
                # 阿拉伯文
                r"بتنسيق JSON",
            ],
            "check": lambda text, _: ('{' in text and '}' in text) or ('[' in text and ']' in text),
        },
        
        # 结构约束
        "end_with_question": {
            "patterns": [
                # 英文
                r"end with a question",
                r"ends with a question",
                r"finish with a question",
                # 中文
                r"以问句结尾",
                r"以问题结尾",
                # 德文
                r"mit einer Frage enden",
                r"endet mit einer Frage",
                # 法文
                r"terminer par une question",
                r"finir par une question",
                # 西班牙文
                r"terminar con una pregunta",
                r"acabar con una pregunta",
                # 意大利文
                r"terminare con una domanda",
                r"finire con una domanda",
                # 日文
                r"質問で終わる",
                r"問いで終わる",
                # 葡萄牙文
                r"terminar com uma pergunta",
                # 波兰文
                r"zakończyć pytaniem",
                # 罗马尼亚文
                r"termina cu o întrebare",
                # 阿拉伯文
                r"ينتهي بسؤال",
            ],
            "check": lambda text, _: text.rstrip().endswith('?') or text.rstrip().endswith('？') or text.rstrip().endswith('؟'),
        },
        "start_with": {
            "patterns": [
                # 英文
                r"start with ['\"]([^'\"]+)['\"]",
                r"begin with ['\"]([^'\"]+)['\"]",
                r"starts with ['\"]([^'\"]+)['\"]",
                # 中文
                r"开头[是为]['\"]([^'\"]+)['\"]",
                r"以['\"]([^'\"]+)['\"]开头",
                # 德文
                r"beginne mit ['\"]([^'\"]+)['\"]",
                r"anfangen mit ['\"]([^'\"]+)['\"]",
                # 法文
                r"commencer par ['\"]([^'\"]+)['\"]",
                r"débuter par ['\"]([^'\"]+)['\"]",
                # 西班牙文
                r"comenzar con ['\"]([^'\"]+)['\"]",
                r"empezar con ['\"]([^'\"]+)['\"]",
                # 意大利文
                r"iniziare con ['\"]([^'\"]+)['\"]",
                r"cominciare con ['\"]([^'\"]+)['\"]",
                # 日文
                r"['\"]([^'\"]+)['\"]で始める",
                r"['\"]([^'\"]+)['\"]から始める",
                # 葡萄牙文
                r"começar com ['\"]([^'\"]+)['\"]",
                r"iniciar com ['\"]([^'\"]+)['\"]",
                # 波兰文
                r"zaczynać od ['\"]([^'\"]+)['\"]",
                # 罗马尼亚文
                r"începe cu ['\"]([^'\"]+)['\"]",
                # 阿拉伯文
                r"ابدأ بـ ['\"]([^'\"]+)['\"]",
            ],
            "check": lambda text, param: text.strip().lower().startswith(param.lower()),
        },
    }
    
    @classmethod
    def extract_constraints(cls, instruction: str) -> List[Tuple[str, any]]:
        """从指令中提取可验证的约束条件
        
        Args:
            instruction: 指令文本
            
        Returns:
            约束条件列表 [(constraint_type, parameter), ...]
        """
        constraints = []
        instruction_lower = instruction.lower()
        
        for constraint_type, config in cls.VERIFIABLE_INSTRUCTIONS.items():
            for pattern in config["patterns"]:
                match = re.search(pattern, instruction, re.IGNORECASE)
                if match:
                    # 提取参数
                    if match.groups():
                        if len(match.groups()) == 2:
                            param = (match.group(1), match.group(2))
                        else:
                            param = match.group(1)
                    else:
                        param = None
                    constraints.append((constraint_type, param))
                    break  # 每种类型只匹配一次
        
        return constraints
    
    @classmethod
    def check_constraint(cls, text: str, constraint_type: str, param: any) -> bool:
        """检查输出是否满足约束条件
        
        Args:
            text: 模型输出文本
            constraint_type: 约束类型
            param: 约束参数
            
        Returns:
            是否满足约束
        """
        if constraint_type not in cls.VERIFIABLE_INSTRUCTIONS:
            return True  # 未知约束类型默认通过
        
        check_fn = cls.VERIFIABLE_INSTRUCTIONS[constraint_type]["check"]
        try:
            return check_fn(text, param)
        except Exception:
            return False
    
    @classmethod
    def evaluate_instruction_following(
        cls, 
        instruction: str, 
        output: str
    ) -> Dict:
        """评估单个样本的指令遵循情况
        
        Args:
            instruction: 指令文本
            output: 模型输出
            
        Returns:
            评估结果 {
                "constraints": [(type, param, passed), ...],
                "total": int,
                "passed": int,
                "rate": float
            }
        """
        constraints = cls.extract_constraints(instruction)
        
        if not constraints:
            return {
                "constraints": [],
                "total": 0,
                "passed": 0,
                "rate": 1.0,  # 无约束默认通过
            }
        
        results = []
        passed_count = 0
        
        for constraint_type, param in constraints:
            passed = cls.check_constraint(output, constraint_type, param)
            results.append((constraint_type, param, passed))
            if passed:
                passed_count += 1
        
        return {
            "constraints": results,
            "total": len(constraints),
            "passed": passed_count,
            "rate": passed_count / len(constraints) if constraints else 1.0,
        }
    
    @classmethod
    def compute_corpus_metrics(cls, eval_results: List[Dict]) -> Dict:
        """计算语料级指令遵循指标
        
        Args:
            eval_results: 各样本的评估结果列表
            
        Returns:
            语料级指标（专业的指令遵循评估体系）
        """
        total_samples = len(eval_results)
        # 过滤有约束的样本
        samples_with_constraints = [r for r in eval_results if r["total"] > 0]
        
        if not samples_with_constraints:
            return {
                # 核心指标
                "instruction_following_rate": 100.0,
                "strict_accuracy": 100.0,
                "loose_accuracy": 100.0,
                # 统计信息
                "samples_evaluated": 0,
                "total_constraints": 0,
                "avg_constraints_per_sample": 0.0,
                "constraint_coverage": 0.0,
                # 按约束类型分解
                "by_constraint_type": {},
            }
        
        # 计算各项指标
        total_constraints = sum(r["total"] for r in samples_with_constraints)
        total_passed = sum(r["passed"] for r in samples_with_constraints)
        
        # ========== 核心指标 ==========
        # 1. IFR (Instruction Following Rate): 约束级通过率
        ifr = (total_passed / total_constraints * 100) if total_constraints > 0 else 100.0
        
        # 2. Strict Accuracy: 所有约束都通过的样本比例
        strict_pass = sum(1 for r in samples_with_constraints if r["passed"] == r["total"])
        strict_acc = (strict_pass / len(samples_with_constraints) * 100)
        
        # 3. Loose Accuracy: 至少通过一半约束的样本比例
        loose_pass = sum(1 for r in samples_with_constraints if r["passed"] >= r["total"] / 2)
        loose_acc = (loose_pass / len(samples_with_constraints) * 100)
        
        # ========== 统计信息 ==========
        # 平均每样本约束数
        avg_constraints = total_constraints / len(samples_with_constraints)
        
        # 约束覆盖率：有约束的样本占总样本比例
        constraint_coverage = (len(samples_with_constraints) / total_samples * 100) if total_samples > 0 else 0.0
        
        # ========== 按约束类型分解 ==========
        constraint_type_stats = {}
        for result in samples_with_constraints:
            for constraint_type, param, passed in result.get("constraints", []):
                if constraint_type not in constraint_type_stats:
                    constraint_type_stats[constraint_type] = {"total": 0, "passed": 0}
                constraint_type_stats[constraint_type]["total"] += 1
                if passed:
                    constraint_type_stats[constraint_type]["passed"] += 1
        
        # 计算每种约束类型的通过率
        by_constraint_type = {}
        for ctype, stats in constraint_type_stats.items():
            rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
            by_constraint_type[ctype] = {
                "total": stats["total"],
                "passed": stats["passed"],
                "rate": rate,
            }
        
        # 按约束大类聚合（字数/格式/关键词/结构/语言）
        category_mapping = {
            # 字数约束
            "min_words": "word_count",
            "max_words": "word_count",
            "exact_words": "word_count",
            # 句子数约束
            "min_sentences": "sentence_count",
            "max_sentences": "sentence_count",
            # 段落约束
            "min_paragraphs": "structure",
            "exact_paragraphs": "structure",
            "paragraph_divider": "structure",
            # 关键词约束
            "keyword_include": "keyword",
            "keyword_count": "keyword",
            "keyword_exclude": "keyword",
            "word_frequency": "keyword",
            # 格式约束
            "bullet_points": "format",
            "numbered_list": "format",
            "json_format": "format",
            "title_double_brackets": "format",
            "title_square_brackets": "format",
            # 结构约束
            "end_with_question": "structure",
            "start_with": "structure",
            # 语言约束
            "response_language": "language",
        }
        
        category_stats = {}
        for ctype, stats in constraint_type_stats.items():
            category = category_mapping.get(ctype, "other")
            if category not in category_stats:
                category_stats[category] = {"total": 0, "passed": 0}
            category_stats[category]["total"] += stats["total"]
            category_stats[category]["passed"] += stats["passed"]
        
        by_category = {}
        category_labels = {
            "word_count": "字数约束",
            "sentence_count": "句子数约束",
            "format": "格式约束",
            "keyword": "关键词约束",
            "structure": "结构约束",
            "language": "语言约束",
            "other": "其他约束",
        }
        for cat, stats in category_stats.items():
            rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
            by_category[cat] = {
                "label": category_labels.get(cat, cat),
                "total": stats["total"],
                "passed": stats["passed"],
                "rate": rate,
            }
        
        return {
            # 核心指标
            "instruction_following_rate": ifr,
            "strict_accuracy": strict_acc,
            "loose_accuracy": loose_acc,
            # 统计信息
            "samples_evaluated": len(samples_with_constraints),
            "total_constraints": total_constraints,
            "avg_constraints_per_sample": avg_constraints,
            "constraint_coverage": constraint_coverage,
            # 按约束类型分解
            "by_constraint_type": by_constraint_type,
            "by_category": by_category,
        }


# ============================================================
# 指令遵循任务识别关键词
# ============================================================

INSTRUCTION_FOLLOWING_KEYWORDS = [
    # 英文
    "at least", "at most", "exactly", "no more than", "no less than",
    "must include", "must contain", "do not use", "avoid using",
    "bullet point", "numbered list", "json format", "markdown format",
    "end with", "start with", "begin with",
    "paragraphs", "sentences", "words",
    # 中文
    "不少于", "不超过", "至少", "最多", "恰好",
    "必须包含", "不要使用", "禁止使用",
    "列表形式", "编号", "JSON格式",
    "以问句结尾", "以问题结尾", "开头必须",
]


# 预定义模型的prompt模板
MODEL_TEMPLATES = {
    # Granite系列
    "granite": {
        "format": "<|start_of_role|>user<|end_of_role|>{instruction}\n\n{input}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
    },
    # Qwen系列
    "qwen": {
        "format": "<|im_start|>user\n{instruction}\n\n{input}<|im_end|>\n<|im_start|>assistant\n",
    },
    # LLaMA-3系列
    "llama": {
        "format": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    # 通用ChatML格式
    "chatml": {
        "format": "<|im_start|>user\n{instruction}\n\n{input}<|im_end|>\n<|im_start|>assistant\n",
    },
    # 默认格式（简单拼接）
    "default": {
        "format": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    },
}


@dataclass
class EvalResult:
    """评估结果"""
    bleu: float
    rouge1: float
    rouge2: float
    rougeL: float
    bert_precision: float
    bert_recall: float
    bert_f1: float


def detect_model_type(model_name: str) -> str:
    """根据模型名称检测模型类型，返回对应的模板key"""
    model_name_lower = model_name.lower()
    
    if "granite" in model_name_lower:
        return "granite"
    elif "qwen" in model_name_lower:
        return "qwen"
    elif "llama" in model_name_lower:
        return "llama"
    elif "gemma" in model_name_lower:
        return "chatml"  # Gemma使用类似ChatML的格式
    else:
        return "default"


def format_prompt(instruction: str, input_text: str, model_type: str) -> str:
    """根据模型类型格式化prompt"""
    template = MODEL_TEMPLATES.get(model_type, MODEL_TEMPLATES["default"])
    return template["format"].format(instruction=instruction, input=input_text)


def load_model_and_tokenizer(
    model_path: str,
    adapter_path: str = None,
    device_map: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """
    加载模型和tokenizer
    
    Returns:
        model, tokenizer, model_type
    """
    print(f"加载模型: {model_path}")
    
    # 检测模型类型
    model_type = detect_model_type(model_path)
    print(f"  检测到模型类型: {model_type}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",  # 禁用Flash Attention，提升兼容性
    )
    
    # 如果有adapter，加载LoRA
    if adapter_path:
        print(f"  加载LoRA适配器: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    return model, tokenizer, model_type


def release_model(model, tokenizer):
    """释放模型显存"""
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("  已释放显存")


class ModelEvaluator:
    def __init__(
        self,
        base_model_path: str,
        finetuned_model_path: str = None,
        adapter_path: str = None,
        device: str = "cuda",
        max_new_tokens: int = 512,
    ):
        """
        初始化评估器
        
        Args:
            base_model_path: 基础模型路径
            finetuned_model_path: 全量微调后的模型路径（与adapter_path二选一）
            adapter_path: LoRA adapter路径
            device: 设备
            max_new_tokens: 生成最大token数
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        print("加载Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        print("加载基础模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",  # 禁用Flash Attention，提升兼容性
        )
        self.base_model.eval()
        
        # 保存adapter路径，延迟加载（避免影响base_model评估）
        self.finetuned_model = None
        self.adapter_path = adapter_path
        self.finetuned_model_path = finetuned_model_path
        self.base_model_path = base_model_path
        
        # 初始化评估器
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        self.bleu = BLEU(effective_order=True, lowercase=False, tokenize='zh')

    def _generate(self, model, prompt: str) -> str:
        """使用模型生成回复"""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1536
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # 使用贪婪解码确保可复现
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 只取新生成的部分
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()

    def _format_prompt(self, instruction: str, input_text: str) -> str:
        """格式化prompt（Granite模板）"""
        # Granite使用的对话模板
        prompt = f"""<|start_of_role|>user<|end_of_role|>{instruction}

{input_text}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>"""
        return prompt

    def _tokenize_chinese(self, text: str) -> List[str]:
        """中文分词（用于BLEU计算）"""
        return list(jieba.cut(text))

    def _detect_lang(self, text: str) -> str:
        """检测文本语言（中文/英文/混合）"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return "en"
        ratio = chinese_chars / total_chars
        if ratio > 0.3:
            return "zh"
        return "en"

    def _classify_sample(self, sample: Dict) -> Optional[str]:
        """根据instruction或task_type判断任务类型：翻译/总结/指令遵循"""
        task_type = sample.get("task_type", "").lower()
        if task_type in ("translation", "summarization", "instruction_following"):
            return task_type
        instr = sample.get("instruction", "")
        instr_lower = instr.lower()
        translation_keywords = [
            "翻译", "translate", "translation", "译为", "译成",
            "english translation", "中文翻译", "中译英", "英译中",
        ]
        summary_keywords = [
            "总结", "摘要", "概括", "summary", "summarize",
            "提炼", "提取主要观点", "总结以下", "概括下文",
        ]
        # 优先检查翻译和总结
        if any(kw in instr_lower for kw in translation_keywords):
            return "translation"
        if any(kw in instr_lower for kw in summary_keywords):
            return "summarization"
        # 检查指令遵循（v6新增）
        if any(kw in instr_lower for kw in INSTRUCTION_FOLLOWING_KEYWORDS):
            return "instruction_following"
        return None

    def _compute_metrics(
        self, predictions: List[str], references: List[str], lang: str = "mixed"
    ) -> EvalResult:
        """计算评估指标"""
        # 处理空预测
        predictions = [p.strip() if p else " " for p in predictions]
        references = [r.strip() if r else " " for r in references]
        
        # 清理预测中的常见前缀（模型可能生成的提示性文本）
        cleaned_predictions = []
        for pred in predictions:
            # 移除常见前缀
            for prefix in [
                "Here is the English translation:",
                "Here is the Chinese translation:",
                "Translation:",
                "翻译：",
                "翻译结果：",
                "译文：",
            ]:
                if pred.startswith(prefix):
                    pred = pred[len(prefix):].strip()
                    break
            cleaned_predictions.append(pred)
        
        predictions = cleaned_predictions
        
        # 自动检测语言（基于参考答案）
        sample_text = ''.join(references[:10])
        detected_lang = self._detect_lang(sample_text)
        
        # BLEU（使用SacreBLEU的自动分词）
        try:
            if detected_lang == "zh":
                # 中文：使用字符级分词
                bleu_score = self.bleu.corpus_score(predictions, [[r] for r in references]).score
            else:
                # 英文：SacreBLEU自动处理
                bleu_score = self.bleu.corpus_score(predictions, [[r] for r in references]).score
        except Exception as e:
            print(f"BLEU计算失败: {e}, 返回0")
            bleu_score = 0.0
        
        # ROUGE
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        # BERTScore（自动检测语言）
        P, R, F1 = bert_score(
            predictions, references, 
            lang=detected_lang,
            verbose=False
        )
        
        return EvalResult(
            bleu=bleu_score,
            rouge1=np.mean(rouge1_scores) * 100,
            rouge2=np.mean(rouge2_scores) * 100,
            rougeL=np.mean(rougeL_scores) * 100,
            bert_precision=P.mean().item() * 100,
            bert_recall=R.mean().item() * 100,
            bert_f1=F1.mean().item() * 100,
        )

    def evaluate_model(
        self, model, samples: List[Dict], desc: str = "评估"
    ) -> Tuple[EvalResult, List[Dict]]:
        """评估单个模型"""
        predictions = []
        references = []
        detailed_results = []
        
        for sample in tqdm(samples, desc=desc):
            prompt = self._format_prompt(sample["instruction"], sample["input"])
            prediction = self._generate(model, prompt)
            predictions.append(prediction)
            references.append(sample["output"])
            
            detailed_results.append({
                "instruction": sample["instruction"],
                "input": sample["input"][:200] + "..." if len(sample["input"]) > 200 else sample["input"],
                "reference": sample["output"],
                "prediction": prediction,
            })
        
        metrics = self._compute_metrics(predictions, references)
        return metrics, detailed_results

    def compare_models(
        self, eval_file: str, max_samples: int = 0, output_dir: str = "evaluation"
    ) -> Dict:
        """对比微调前后的模型表现（按任务类型拆分：翻译/总结/指令遵循）"""
        # 加载评估数据
        print(f"加载评估数据: {eval_file}")
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
        # 随机采样（保证可复现）；max_samples=0 表示全量
        if max_samples and len(eval_data) > max_samples:
            random.seed(42)
            eval_data = random.sample(eval_data, max_samples)
        print(f"评估样本数: {len(eval_data)}")
        
        # 按任务类型拆分
        translation_data: List[Dict] = []
        summarization_data: List[Dict] = []
        instruction_following_data: List[Dict] = []
        # 指令遵循按来源/语言分组
        if_by_source: Dict[str, List[Dict]] = {}
        
        for sample in eval_data:
            task = self._classify_sample(sample)
            if task == "translation":
                translation_data.append(sample)
            elif task == "summarization":
                summarization_data.append(sample)
            elif task == "instruction_following":
                instruction_following_data.append(sample)
                # 按来源分组
                source = sample.get("source", "")
                lang = sample.get("language", "")
                if "mifeval" in source.lower():
                    group_key = f"M-IFEval-{lang}" if lang else "M-IFEval"
                elif source:
                    group_key = source
                else:
                    group_key = "IFEval" if lang == "en" else f"IFEval-{lang}" if lang else "Other"
                if group_key not in if_by_source:
                    if_by_source[group_key] = []
                if_by_source[group_key].append(sample)
        
        print(f"翻译样本数: {len(translation_data)}")
        print(f"总结样本数: {len(summarization_data)}")
        print(f"指令遵循样本数: {len(instruction_following_data)}")
        if if_by_source:
            print(f"指令遵循来源分布:")
            for src, samples in sorted(if_by_source.items()):
                print(f"  - {src}: {len(samples)}条")
        
        results: Dict[str, Dict] = {}
        
        # 评估基础模型
        print("\n" + "=" * 50)
        print("评估基础模型")
        print("=" * 50)
        
        if translation_data:
            base_metrics_tr, base_details_tr = self.evaluate_model(
                self.base_model, translation_data, "基础模型-翻译子集"
            )
            results.setdefault("translation", {})["base_model"] = {
                "metrics": base_metrics_tr.__dict__,
                "details": base_details_tr,
            }
        
        if summarization_data:
            base_metrics_sum, base_details_sum = self.evaluate_model(
                self.base_model, summarization_data, "基础模型-总结子集"
            )
            results.setdefault("summarization", {})["base_model"] = {
                "metrics": base_metrics_sum.__dict__,
                "details": base_details_sum,
            }
        
        if instruction_following_data:
            base_metrics_if, base_details_if = self.evaluate_model(
                self.base_model, instruction_following_data, "基础模型-指令遵循子集"
            )
            # 计算指令遵循特有指标
            if_eval_results = []
            for detail in base_details_if:
                if_result = InstructionFollowingEvaluator.evaluate_instruction_following(
                    detail["instruction"], detail["prediction"]
                )
                if_eval_results.append(if_result)
            if_corpus_metrics = InstructionFollowingEvaluator.compute_corpus_metrics(if_eval_results)
            
            results.setdefault("instruction_following", {})["base_model"] = {
                "metrics": {**base_metrics_if.__dict__, **if_corpus_metrics},
                "details": base_details_if,
            }
            
            # 按来源分组评估（仅计算指令遵循指标）
            if_by_source_results = {}
            for src, src_samples in if_by_source.items():
                src_details = [d for d, s in zip(base_details_if, instruction_following_data) 
                              if s.get("source", "") == src_samples[0].get("source", "") and 
                                 s.get("language", "") == src_samples[0].get("language", "")]
                if not src_details:
                    # 通过索引匹配
                    src_indices = set()
                    for i, s in enumerate(instruction_following_data):
                        s_src = s.get("source", "")
                        s_lang = s.get("language", "")
                        if "mifeval" in s_src.lower():
                            s_key = f"M-IFEval-{s_lang}" if s_lang else "M-IFEval"
                        elif s_src:
                            s_key = s_src
                        else:
                            s_key = "IFEval" if s_lang == "en" else f"IFEval-{s_lang}" if s_lang else "Other"
                        if s_key == src:
                            src_indices.add(i)
                    src_details = [base_details_if[i] for i in src_indices]
                
                src_if_results = []
                for detail in src_details:
                    if_result = InstructionFollowingEvaluator.evaluate_instruction_following(
                        detail["instruction"], detail["prediction"]
                    )
                    src_if_results.append(if_result)
                src_metrics = InstructionFollowingEvaluator.compute_corpus_metrics(src_if_results)
                if_by_source_results[src] = {
                    "samples": len(src_details),
                    **src_metrics
                }
            results["instruction_following"]["base_model"]["by_source"] = if_by_source_results
        
        # 释放基础模型显存（如果需要加载微调模型）
        if self.finetuned_model_path or self.adapter_path:
            print("\n释放基础模型显存...")
            del self.base_model
            gc.collect()
            torch.cuda.empty_cache()
        
        # 加载微调模型
        if self.finetuned_model_path:
            print("\n加载全量微调模型...")
            self.finetuned_model = AutoModelForCausalLM.from_pretrained(
                self.finetuned_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",  # 禁用Flash Attention，提升兼容性
            )
            self.finetuned_model.eval()
        elif self.adapter_path:
            print("\n加载LoRA适配器...")
            finetuned_base = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",  # 禁用Flash Attention，提升兼容性
            )
            self.finetuned_model = PeftModel.from_pretrained(
                finetuned_base, self.adapter_path
            )
            self.finetuned_model.eval()
        
        # 评估微调模型
        if self.finetuned_model:
            print("\n" + "=" * 50)
            print("评估微调模型")
            print("=" * 50)
            if translation_data:
                ft_metrics_tr, ft_details_tr = self.evaluate_model(
                    self.finetuned_model, translation_data, "微调模型-翻译子集"
                )
                results.setdefault("translation", {})["finetuned_model"] = {
                    "metrics": ft_metrics_tr.__dict__,
                    "details": ft_details_tr,
                }
            if summarization_data:
                ft_metrics_sum, ft_details_sum = self.evaluate_model(
                    self.finetuned_model, summarization_data, "微调模型-总结子集"
                )
                results.setdefault("summarization", {})["finetuned_model"] = {
                    "metrics": ft_metrics_sum.__dict__,
                    "details": ft_details_sum,
                }
            if instruction_following_data:
                ft_metrics_if, ft_details_if = self.evaluate_model(
                    self.finetuned_model, instruction_following_data, "微调模型-指令遵循子集"
                )
                # 计算指令遵循特有指标
                if_eval_results = []
                for detail in ft_details_if:
                    if_result = InstructionFollowingEvaluator.evaluate_instruction_following(
                        detail["instruction"], detail["prediction"]
                    )
                    if_eval_results.append(if_result)
                if_corpus_metrics = InstructionFollowingEvaluator.compute_corpus_metrics(if_eval_results)
                
                results.setdefault("instruction_following", {})["finetuned_model"] = {
                    "metrics": {**ft_metrics_if.__dict__, **if_corpus_metrics},
                    "details": ft_details_if,
                }
                
                # 按来源分组评估
                ft_by_source_results = {}
                for src, src_samples in if_by_source.items():
                    src_indices = set()
                    for i, s in enumerate(instruction_following_data):
                        s_src = s.get("source", "")
                        s_lang = s.get("language", "")
                        if "mifeval" in s_src.lower():
                            s_key = f"M-IFEval-{s_lang}" if s_lang else "M-IFEval"
                        elif s_src:
                            s_key = s_src
                        else:
                            s_key = "IFEval" if s_lang == "en" else f"IFEval-{s_lang}" if s_lang else "Other"
                        if s_key == src:
                            src_indices.add(i)
                    src_details = [ft_details_if[i] for i in src_indices]
                    
                    src_if_results = []
                    for detail in src_details:
                        if_result = InstructionFollowingEvaluator.evaluate_instruction_following(
                            detail["instruction"], detail["prediction"]
                        )
                        src_if_results.append(if_result)
                    src_metrics = InstructionFollowingEvaluator.compute_corpus_metrics(src_if_results)
                    ft_by_source_results[src] = {
                        "samples": len(src_details),
                        **src_metrics
                    }
                results["instruction_following"]["finetuned_model"]["by_source"] = ft_by_source_results
        
        # 生成对比报告
        self._generate_report(results, output_dir)
        
        return results

    def _generate_report(self, results: Dict, output_dir: str):
        """生成评估报告（按任务类型拆分）"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存指标摘要（按子集/任务类型）
        summary = {}
        for subset_name, subset_res in results.items():
            summary[subset_name] = {}
            if "base_model" in subset_res:
                summary[subset_name]["base_model"] = subset_res["base_model"]["metrics"]
            if "finetuned_model" in subset_res:
                summary[subset_name]["finetuned_model"] = subset_res["finetuned_model"]["metrics"]
        with open(os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 保存详细结果（用于人工复核）
        details_out = {}
        for subset_name, subset_res in results.items():
            details_out[subset_name] = {}
            if "base_model" in subset_res:
                details_out[subset_name]["base_model"] = subset_res["base_model"]["details"][:20]
            if "finetuned_model" in subset_res:
                details_out[subset_name]["finetuned_model"] = subset_res["finetuned_model"]["details"][:20]
        with open(os.path.join(output_dir, "eval_details.json"), "w", encoding="utf-8") as f:
            json.dump(details_out, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存: {os.path.join(output_dir, 'eval_details.json')} (各子集前20条样本)")
        
        # 生成Markdown报告
        report = ["# 模型评估报告\n"]
        report.append("## 评估指标说明\n")
        report.append("### 翻译/总结任务指标")
        report.append("- **BLEU**: 机器翻译标准评估指标，衡量n-gram重合度")
        report.append("- **ROUGE-1/2/L**: 文本摘要评估指标，衡量词/短语覆盖率")
        report.append("- **BERTScore**: 基于BERT的语义相似度评估\n")
        report.append("### 指令遵循任务指标（核心）")
        report.append("- **IFR (Instruction Following Rate)**: 约束级通过率，所有约束中被满足的比例")
        report.append("- **Strict Accuracy**: 样本级完全通过率，所有约束都满足的样本比例")
        report.append("- **Loose Accuracy**: 宽松通过率，至少满足一半约束的样本比例")
        report.append("- **约束类型分解**: 按字数/格式/关键词/结构等约束类型分别统计通过率\n")
        
        metrics_names = [
            ("BLEU", "bleu"),
            ("ROUGE-1", "rouge1"),
            ("ROUGE-2", "rouge2"),
            ("ROUGE-L", "rougeL"),
            ("BERTScore-P", "bert_precision"),
            ("BERTScore-R", "bert_recall"),
            ("BERTScore-F1", "bert_f1"),
        ]
        
        # 指令遵循核心指标
        if_metrics_names = [
            ("IFR (约束通过率)", "instruction_following_rate"),
            ("Strict Acc (完全通过率)", "strict_accuracy"),
            ("Loose Acc (宽松通过率)", "loose_accuracy"),
        ]
        
        # 指令遵循统计信息
        if_stats_names = [
            ("评估样本数", "samples_evaluated"),
            ("总约束数", "total_constraints"),
            ("平均约束数/样本", "avg_constraints_per_sample"),
            ("约束覆盖率", "constraint_coverage"),
        ]
        
        subset_labels = {
            "translation": "翻译子集 (Translation)",
            "summarization": "总结子集 (Summarization)",
            "instruction_following": "指令遵循子集 (Instruction Following)",
        }
        
        for subset_name, label in subset_labels.items():
            if subset_name not in results:
                continue
            subset_res = results[subset_name]
            base_m = subset_res["base_model"]["metrics"]
            ft_m = subset_res.get("finetuned_model", {}).get("metrics", {})
            
            report.append(f"\n## {label}\n")
            
            # 指令遵循子集使用专门的指标体系
            if subset_name == "instruction_following":
                # 核心指标表
                report.append("### 核心指令遵循指标\n")
                report.append("> 说明：IFR衡量约束级通过率，Strict Acc衡量样本级完全通过率，Loose Acc衡量至少通过一半约束的样本比例\n")
                report.append("| 指标 | 基础模型 | 微调模型 | 提升 |")
                report.append("|------|----------|----------|------|")
                for display_name, key in if_metrics_names:
                    base_val = base_m.get(key, 0)
                    ft_val = ft_m.get(key, 0) if ft_m else 0
                    if ft_m and isinstance(ft_val, (int, float)):
                        diff = ft_val - base_val
                        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
                    else:
                        diff_str = "-"
                    base_str = f"{base_val:.2f}%" if isinstance(base_val, float) else str(base_val)
                    ft_str = f"{ft_val:.2f}%" if isinstance(ft_val, float) else str(ft_val) if ft_m else "-"
                    report.append(f"| {display_name} | {base_str} | {ft_str} | {diff_str} |")
                
                # 统计信息表
                report.append("\n### 评估统计信息\n")
                report.append("| 统计项 | 基础模型 | 微调模型 |")
                report.append("|--------|----------|----------|")
                for display_name, key in if_stats_names:
                    base_val = base_m.get(key, 0)
                    ft_val = ft_m.get(key, 0) if ft_m else "-"
                    if key in ("avg_constraints_per_sample", "constraint_coverage"):
                        base_str = f"{base_val:.2f}" if isinstance(base_val, (int, float)) else str(base_val)
                        ft_str = f"{ft_val:.2f}" if isinstance(ft_val, (int, float)) else str(ft_val)
                        if key == "constraint_coverage":
                            base_str += "%"
                            ft_str = ft_str + "%" if ft_str != "-" else ft_str
                    else:
                        base_str = str(int(base_val)) if isinstance(base_val, (int, float)) else str(base_val)
                        ft_str = str(int(ft_val)) if isinstance(ft_val, (int, float)) else str(ft_val)
                    report.append(f"| {display_name} | {base_str} | {ft_str} |")
                
                # 按约束大类分解
                base_by_cat = base_m.get("by_category", {})
                ft_by_cat = ft_m.get("by_category", {}) if ft_m else {}
                if base_by_cat or ft_by_cat:
                    report.append("\n### 按约束类型分解\n")
                    report.append("> 各类约束的遵循率对比\n")
                    report.append("| 约束类型 | 基础模型 | 微调模型 | 提升 |")
                    report.append("|----------|----------|----------|------|")
                    
                    all_cats = set(base_by_cat.keys()) | set(ft_by_cat.keys())
                    cat_order = ["word_count", "format", "keyword", "structure", "other"]
                    sorted_cats = sorted(all_cats, key=lambda x: cat_order.index(x) if x in cat_order else 99)
                    
                    for cat in sorted_cats:
                        base_info = base_by_cat.get(cat, {})
                        ft_info = ft_by_cat.get(cat, {})
                        label_cat = base_info.get("label", ft_info.get("label", cat))
                        base_rate = base_info.get("rate", 0)
                        ft_rate = ft_info.get("rate", 0)
                        base_count = base_info.get("total", 0)
                        ft_count = ft_info.get("total", 0)
                        
                        diff = ft_rate - base_rate if ft_info else 0
                        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
                        
                        base_cat_str = f"{base_rate:.1f}% ({base_info.get('passed', 0)}/{base_count})"
                        ft_cat_str = f"{ft_rate:.1f}% ({ft_info.get('passed', 0)}/{ft_count})" if ft_info else "-"
                        
                        report.append(f"| {label_cat} | {base_cat_str} | {ft_cat_str} | {diff_str} |")
                
                # 按来源/语言分组的得分（IFEval和M-IFEval各语言）
                base_by_source = subset_res.get("base_model", {}).get("by_source", {})
                ft_by_source = subset_res.get("finetuned_model", {}).get("by_source", {})
                if base_by_source or ft_by_source:
                    report.append("\n### 按测试集/语言分组得分\n")
                    report.append("> IFEval和M-IFEval各语言版本的单独得分\n")
                    report.append("| 测试集 | 样本数 | 基础IFR | 微调IFR | 提升 | 基础Strict | 微调Strict | 提升 |")
                    report.append("|----------|--------|---------|---------|------|------------|------------|------|")
                    
                    # 排序：先IFEval，再M-IFEval按语言排序
                    all_sources = list(set(base_by_source.keys()) | set(ft_by_source.keys()))
                    def source_sort_key(s):
                        if s == "IFEval":
                            return (0, "")
                        elif s.startswith("M-IFEval-"):
                            return (1, s.split("-")[-1])
                        else:
                            return (2, s)
                    all_sources.sort(key=source_sort_key)
                    
                    for src in all_sources:
                        base_info = base_by_source.get(src, {})
                        ft_info = ft_by_source.get(src, {})
                        samples = base_info.get("samples", ft_info.get("samples", 0))
                        
                        base_ifr = base_info.get("instruction_following_rate", 0)
                        ft_ifr = ft_info.get("instruction_following_rate", 0)
                        ifr_diff = ft_ifr - base_ifr if ft_info else 0
                        ifr_diff_str = f"+{ifr_diff:.1f}" if ifr_diff > 0 else f"{ifr_diff:.1f}"
                        
                        base_strict = base_info.get("strict_accuracy", 0)
                        ft_strict = ft_info.get("strict_accuracy", 0)
                        strict_diff = ft_strict - base_strict if ft_info else 0
                        strict_diff_str = f"+{strict_diff:.1f}" if strict_diff > 0 else f"{strict_diff:.1f}"
                        
                        ft_ifr_str = f"{ft_ifr:.1f}%" if ft_info else "-"
                        ft_strict_str = f"{ft_strict:.1f}%" if ft_info else "-"
                        
                        report.append(f"| {src} | {samples} | {base_ifr:.1f}% | {ft_ifr_str} | {ifr_diff_str} | {base_strict:.1f}% | {ft_strict_str} | {strict_diff_str} |")
                    
                    # 汇总行
                    total_samples = sum(base_by_source.get(s, {}).get("samples", 0) for s in all_sources)
                    base_total_ifr = base_m.get('instruction_following_rate', 0)
                    ft_total_ifr = ft_m.get('instruction_following_rate', 0) if ft_m else 0
                    base_total_strict = base_m.get('strict_accuracy', 0)
                    ft_total_strict = ft_m.get('strict_accuracy', 0) if ft_m else 0
                    report.append(f"| **总计** | {total_samples} | {base_total_ifr:.1f}% | {ft_total_ifr:.1f}% | - | {base_total_strict:.1f}% | {ft_total_strict:.1f}% | - |")
                
                # 内容质量参考（辅助指标）
                report.append("\n### 内容质量参考（辅助指标）\n")
                report.append("> 注：以下指标仅供参考，指令遵循任务的核心评判依据是上述约束遵循率\n")
                report.append("| 指标 | 基础模型 | 微调模型 | 提升 |")
                report.append("|------|----------|----------|------|")
                content_metrics = [("BERTScore-F1", "bert_f1")]
                for display_name, key in content_metrics:
                    base_val = base_m.get(key, 0)
                    ft_val = ft_m.get(key, 0) if ft_m else 0
                    diff = ft_val - base_val if ft_m else 0
                    diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
                    ft_str = f"{ft_val:.2f}" if ft_m else "-"
                    report.append(f"| {display_name} | {base_val:.2f} | {ft_str} | {diff_str} |")
                
                # 结论
                report.append("\n### 子集结论\n")
                if ft_m:
                    ifr_diff = ft_m.get("instruction_following_rate", 0) - base_m.get("instruction_following_rate", 0)
                    strict_diff = ft_m.get("strict_accuracy", 0) - base_m.get("strict_accuracy", 0)
                    
                    if ifr_diff > 0 and strict_diff > 0:
                        report.append(f"微调后模型在指令遵循能力上有提升：IFR +{ifr_diff:.2f}%, Strict Acc +{strict_diff:.2f}%")
                    elif ifr_diff > 0:
                        report.append(f"微调后模型的约束遵循率有提升（IFR +{ifr_diff:.2f}%），但完全通过率略有下降（Strict Acc {strict_diff:.2f}%）")
                    elif strict_diff > 0:
                        report.append(f"微调后模型的完全通过率有提升（Strict Acc +{strict_diff:.2f}%），约束遵循率变化：IFR {ifr_diff:.2f}%")
                    else:
                        report.append(f"微调后模型在指令遵循能力上未见明显提升（IFR {ifr_diff:.2f}%, Strict Acc {strict_diff:.2f}%），建议增加指令遵循训练数据或调整训练策略。")
                else:
                    report.append("仅评估了基础模型，请提供微调后的模型进行对比。")
                
                continue  # 跳过通用指标输出
            
            # 非指令遵循子集（翻译/总结）使用通用指标
            report.append("| 指标 | 基础模型 | 微调模型 | 提升 |")
            report.append("|------|----------|----------|------|")
            
            for display_name, key in metrics_names:
                base_val = base_m.get(key, 0)
                ft_val = ft_m.get(key, 0) if ft_m else "-"
                if ft_m:
                    diff = ft_val - base_val
                    diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
                else:
                    diff_str = "-"
                ft_str = f"{ft_val:.2f}" if isinstance(ft_val, (int, float)) else ft_val
                report.append(f"| {display_name} | {base_val:.2f} | {ft_str} | {diff_str} |")
            
            # 非指令遵循子集的结论
            report.append("\n### 子集结论\n")
            if ft_m:
                improvements = []
                for display_name, key in metrics_names:
                    diff = ft_m.get(key, 0) - base_m.get(key, 0)
                    if diff > 0:
                        improvements.append(f"{display_name} (+{diff:.2f})")
                if improvements:
                    report.append(f"微调后模型在该子集的以下指标上有提升: {', '.join(improvements)}")
                else:
                    report.append("微调后模型在该子集的主要指标上未见明显提升，建议调整训练参数或增加数据量。")
            else:
                report.append("仅评估了基础模型，请提供微调后的模型进行对比。")
        
        
        report_path = os.path.join(output_dir, "eval_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(f"\n评估报告已保存: {report_path}")
        
        # 打印摘要（终端）
        print("\n" + "=" * 60)
        print("评估结果摘要（按子集）")
        print("=" * 60)
        for subset_name, label in subset_labels.items():
            if subset_name not in results:
                continue
            subset_res = results[subset_name]
            base_m = subset_res["base_model"]["metrics"]
            ft_m = subset_res.get("finetuned_model", {}).get("metrics", {})
            print(f"\n[{label}]")
            print(f"{'指标':<20} {'基础模型':>12} {'微调模型':>12} {'提升':>10}")
            print("-" * 60)
            
            # 指令遵循子集使用专门的指标
            if subset_name == "instruction_following":
                for display_name, key in if_metrics_names:
                    base_val = base_m.get(key, 0)
                    ft_val = ft_m.get(key, 0) if ft_m else 0
                    diff = ft_val - base_val if ft_m else 0
                    diff_str = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"
                    ft_str = f"{ft_val:.2f}%" if ft_m else "-"
                    print(f"{display_name:<20} {base_val:>11.2f}% {ft_str:>12} {diff_str:>10}")
            else:
                # 翻译/总结子集使用BLEU/ROUGE指标
                for display_name, key in metrics_names:
                    base_val = base_m.get(key, 0)
                    ft_val = ft_m.get(key, 0) if ft_m else 0
                    diff = ft_val - base_val if ft_m else 0
                    diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
                    ft_str = f"{ft_val:.2f}" if ft_m else "-"
                    print(f"{display_name:<20} {base_val:>12.2f} {ft_str:>12} {diff_str:>10}")
        print("=" * 60)


class MultiModelEvaluator:
    """多模型对比评估器"""
    
    def __init__(self, max_new_tokens: int = 512):
        self.max_new_tokens = max_new_tokens
        self.rouge_scorer_obj = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        self.bleu = BLEU(effective_order=True, lowercase=False, tokenize='zh')
    
    def _generate(self, model, tokenizer, prompt: str) -> str:
        """使用模型生成回复"""
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1536
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()
    
    def _tokenize_chinese(self, text: str) -> List[str]:
        return list(jieba.cut(text))
    
    def _detect_lang(self, text: str) -> str:
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return "en"
        return "zh" if chinese_chars / total_chars > 0.3 else "en"
    
    def _compute_metrics(self, predictions: List[str], references: List[str]) -> EvalResult:
        """计算评估指标"""
        # 处理空预测
        predictions = [p.strip() if p else " " for p in predictions]
        references = [r.strip() if r else " " for r in references]
        
        # 清理预测中的常见前缀
        cleaned_predictions = []
        for pred in predictions:
            for prefix in [
                "Here is the English translation:",
                "Here is the Chinese translation:",
                "Translation:",
                "翻译：",
                "翻译结果：",
                "译文：",
            ]:
                if pred.startswith(prefix):
                    pred = pred[len(prefix):].strip()
                    break
            cleaned_predictions.append(pred)
        predictions = cleaned_predictions
        
        # 检测语言
        sample_text = ''.join(references[:10])
        detected_lang = self._detect_lang(sample_text)
        
        # BLEU（修复：使用SacreBLEU的中文分词）
        try:
            if detected_lang == "zh":
                # 中文：使用字符级分词（SacreBLEU的zh tokenizer）
                bleu_score = self.bleu.corpus_score(predictions, [[r] for r in references]).score
            else:
                # 英文：SacreBLEU自动处理
                bleu_score = self.bleu.corpus_score(predictions, [[r] for r in references]).score
        except Exception as e:
            print(f"BLEU计算失败: {e}, 返回0")
            bleu_score = 0.0
        
        # ROUGE
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer_obj.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        # BERTScore
        P, R, F1 = bert_score(predictions, references, lang=detected_lang, verbose=False)
        
        return EvalResult(
            bleu=bleu_score,
            rouge1=np.mean(rouge1_scores) * 100,
            rouge2=np.mean(rouge2_scores) * 100,
            rougeL=np.mean(rougeL_scores) * 100,
            bert_precision=P.mean().item() * 100,
            bert_recall=R.mean().item() * 100,
            bert_f1=F1.mean().item() * 100,
        )
    
    def _classify_sample(self, sample: Dict) -> Optional[str]:
        """根据instruction或task_type判断任务类型：翻译/总结/指令遵循"""
        task_type = sample.get("task_type", "").lower()
        if task_type in ("translation", "summarization", "instruction_following"):
            return task_type
        instr = sample.get("instruction", "")
        instr_lower = instr.lower()
        translation_keywords = [
            "翻译", "translate", "translation", "译为", "译成",
            "english translation", "中文翻译", "中译英", "英译中",
        ]
        summary_keywords = [
            "总结", "摘要", "概括", "summary", "summarize",
            "提炼", "提取主要观点", "总结以下", "概括下文",
        ]
        # 优先检查翻译和总结
        if any(kw in instr_lower for kw in translation_keywords):
            return "translation"
        if any(kw in instr_lower for kw in summary_keywords):
            return "summarization"
        # 检查指令遵循（v6新增）
        if any(kw in instr_lower for kw in INSTRUCTION_FOLLOWING_KEYWORDS):
            return "instruction_following"
        return None
    
    def evaluate_single_model(
        self,
        model,
        tokenizer,
        model_type: str,
        samples: List[Dict],
        model_name: str,
        desc_suffix: str = "",
    ) -> Tuple[EvalResult, List[Dict]]:
        """评估单个模型"""
        predictions = []
        references = []
        detailed_results = []
        
        desc = f"评估 {model_name}{desc_suffix}"
        for sample in tqdm(samples, desc=desc):
            prompt = format_prompt(sample["instruction"], sample["input"], model_type)
            prediction = self._generate(model, tokenizer, prompt)
            predictions.append(prediction)
            references.append(sample["output"])
            
            detailed_results.append({
                "instruction": sample["instruction"],
                "input": sample["input"][:200] + "..." if len(sample["input"]) > 200 else sample["input"],
                "reference": sample["output"],
                "prediction": prediction,
            })
        
        metrics = self._compute_metrics(predictions, references)
        return metrics, detailed_results
    
    def compare_multiple_models(
        self,
        model_configs: List[Dict],
        eval_file: str,
        max_samples: int = 0,
        output_dir: str = "evaluation",
    ) -> Dict:
        """
        对比多个模型（按任务类型拆分：翻译/总结）
        
        Args:
            model_configs: 模型配置列表，每个元素为 {"name": "显示名称", "path": "模型路径", "adapter": "可选的adapter路径"}
            eval_file: 评估数据文件
            max_samples: 最大评估样本数
            output_dir: 输出目录
        """
        # 加载评估数据
        print(f"\n加载评估数据: {eval_file}")
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
        if max_samples and len(eval_data) > max_samples:
            random.seed(42)
            eval_data = random.sample(eval_data, max_samples)
        print(f"评估样本数: {len(eval_data)}")
        
        # 按任务类型拆分数据
        translation_data: List[Dict] = []
        summarization_data: List[Dict] = []
        instruction_following_data: List[Dict] = []
        for sample in eval_data:
            task = self._classify_sample(sample)
            if task == "translation":
                translation_data.append(sample)
            elif task == "summarization":
                summarization_data.append(sample)
            elif task == "instruction_following":
                instruction_following_data.append(sample)
        
        print(f"翻译样本数: {len(translation_data)}")
        print(f"总结样本数: {len(summarization_data)}")
        print(f"指令遵循样本数: {len(instruction_following_data)}")
        
        results: Dict[str, Dict] = {}
        
        # 逐个评估模型
        for i, config in enumerate(model_configs):
            model_name = config["name"]
            model_path = config["path"]
            adapter_path = config.get("adapter")
            
            print(f"\n{'=' * 60}")
            print(f"[{i+1}/{len(model_configs)}] 评估模型: {model_name}")
            print(f"{'=' * 60}")
            
            try:
                # 加载模型
                model, tokenizer, model_type = load_model_and_tokenizer(
                    model_path, adapter_path
                )
                
                # 评估翻译子集
                if translation_data:
                    tr_metrics, tr_details = self.evaluate_single_model(
                        model, tokenizer, model_type, translation_data, model_name, "-翻译子集"
                    )
                    results.setdefault("translation", {})[model_name] = {
                        "path": model_path,
                        "metrics": tr_metrics.__dict__,
                        "details": tr_details,
                    }
                
                # 评估总结子集
                if summarization_data:
                    sum_metrics, sum_details = self.evaluate_single_model(
                        model, tokenizer, model_type, summarization_data, model_name, "-总结子集"
                    )
                    results.setdefault("summarization", {})[model_name] = {
                        "path": model_path,
                        "metrics": sum_metrics.__dict__,
                        "details": sum_details,
                    }
                
                # 评估指令遵循子集（v6新增）
                if instruction_following_data:
                    if_metrics, if_details = self.evaluate_single_model(
                        model, tokenizer, model_type, instruction_following_data, model_name, "-指令遵循子集"
                    )
                    # 计算指令遵循特有指标
                    if_eval_results = []
                    for detail in if_details:
                        if_result = InstructionFollowingEvaluator.evaluate_instruction_following(
                            detail["instruction"], detail["prediction"]
                        )
                        if_eval_results.append(if_result)
                    if_corpus_metrics = InstructionFollowingEvaluator.compute_corpus_metrics(if_eval_results)
                    
                    results.setdefault("instruction_following", {})[model_name] = {
                        "path": model_path,
                        "metrics": {**if_metrics.__dict__, **if_corpus_metrics},
                        "details": if_details,
                    }
                
                # 释放显存
                print(f"释放 {model_name} 显存...")
                release_model(model, tokenizer)
                
            except Exception as e:
                import traceback
                error_msg = str(e)
                error_trace = traceback.format_exc()
                print(f"\n{'!'*60}")
                print(f"!!! 评估 {model_name} 失败 !!!")
                print(f"!!! 错误类型: {type(e).__name__}")
                print(f"!!! 错误信息: {error_msg}")
                print(f"{'!'*60}")
                traceback.print_exc()
                print(f"{'!'*60}\n")
                
                # 记录详细错误信息，包括堆栈跟踪
                error_info = {
                    "path": model_path,
                    "adapter": adapter_path,
                    "error": error_msg,
                    "error_type": type(e).__name__,
                    "error_trace": error_trace[:500],  # 保存部分堆栈
                }
                if translation_data:
                    results.setdefault("translation", {})[model_name] = error_info.copy()
                if summarization_data:
                    results.setdefault("summarization", {})[model_name] = error_info.copy()
                if instruction_following_data:
                    results.setdefault("instruction_following", {})[model_name] = error_info.copy()
                
                # 尝试释放可能残留的显存
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"  已清理显存（错误恢复）")
                except:
                    pass
        
        # 生成对比报告
        self._generate_comparison_report(results, output_dir)
        
        return results
    
    def _generate_comparison_report(self, results: Dict, output_dir: str):
        """生成多模型对比报告（按任务类型拆分）"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存指标摘要（按子集，包括成功和失败的模型）
        summary = {}
        for subset_name, subset_res in results.items():
            summary[subset_name] = {}
            for model_name, model_data in subset_res.items():
                if "metrics" in model_data:
                    summary[subset_name][model_name] = model_data["metrics"]
                elif "error" in model_data:
                    # 保存失败模型的错误信息
                    summary[subset_name][model_name] = {
                        "error": model_data["error"],
                        "error_type": model_data.get("error_type", "Unknown"),
                        "path": model_data.get("path", "N/A"),
                    }
        
        summary_path = os.path.join(output_dir, "comparison_results.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n指标摘要已保存: {summary_path}")
        
        # 保存详细结果（包括失败模型的信息）
        details_out = {}
        for subset_name, subset_res in results.items():
            details_out[subset_name] = {}
            for model_name, model_data in subset_res.items():
                if "details" in model_data:
                    details_out[subset_name][model_name] = model_data["details"][:10]
                elif "error" in model_data:
                    # 保存失败模型的错误详情
                    details_out[subset_name][model_name] = {
                        "error": model_data["error"],
                        "error_type": model_data.get("error_type", "Unknown"),
                        "error_trace": model_data.get("error_trace", "N/A"),
                    }
        
        details_path = os.path.join(output_dir, "comparison_details.json")
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(details_out, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存: {details_path} (每模型每子集前10条)")
        
        # 生成Markdown报告
        report = ["# 多模型对比评估报告\n"]
        report.append("## 评估指标说明\n")
        report.append("- **BLEU**: 机器翻译标准评估指标，主要用于翻译子集")
        report.append("- **ROUGE-1/2/L**: 文本摘要评估指标，主要用于总结子集")
        report.append("- **BERTScore**: 基于BERT的语义相似度评估\n")
        
        metrics_names = [
            ("BLEU", "bleu"),
            ("ROUGE-1", "rouge1"),
            ("ROUGE-2", "rouge2"),
            ("ROUGE-L", "rougeL"),
            ("BERTScore-P", "bert_precision"),
            ("BERTScore-R", "bert_recall"),
            ("BERTScore-F1", "bert_f1"),
        ]
        
        subset_labels = {
            "translation": "翻译子集 (Translation)",
            "summarization": "总结子集 (Summarization)",
            "instruction_following": "指令遵循子集 (Instruction Following)",
        }
        
        # 获取所有模型名称（包括成功和失败的）
        all_model_names = set()
        for subset_res in results.values():
            all_model_names.update(subset_res.keys())
        
        # 分离成功和失败的模型
        successful_models = []
        failed_models = []
        for name in all_model_names:
            has_error = False
            for subset_res in results.values():
                if name in subset_res and subset_res[name].get("error"):
                    has_error = True
                    break
            if has_error:
                failed_models.append(name)
            else:
                successful_models.append(name)
        
        # 如果有失败的模型，在报告开头显示警告
        if failed_models:
            report.append("## ⚠️ 评估警告\n")
            report.append("以下模型评估失败：\n")
            for name in failed_models:
                error_info = None
                for subset_res in results.values():
                    if name in subset_res and subset_res[name].get("error"):
                        error_info = subset_res[name]
                        break
                if error_info:
                    report.append(f"- **{name}**")
                    report.append(f"  - 模型路径: `{error_info.get('path', 'N/A')}`")
                    report.append(f"  - 错误类型: `{error_info.get('error_type', 'Unknown')}`")
                    report.append(f"  - 错误信息: `{error_info.get('error', 'N/A')}`\n")
            report.append("")
        
        all_model_names = successful_models  # 后续只处理成功的模型
        
        # 模型信息表
        if all_model_names:
            report.append("## 模型信息\n")
            report.append("| 模型名称 | 模型路径 |")
            report.append("|----------|----------|")
            for name in all_model_names:
                path = ""
                for subset_res in results.values():
                    if name in subset_res and "path" in subset_res[name]:
                        path = subset_res[name]["path"]
                        break
                report.append(f"| {name} | {path} |")
        
        # 按子集输出对比结果
        for subset_name, label in subset_labels.items():
            if subset_name not in results:
                continue
            
            subset_res = results[subset_name]
            valid_models = [n for n in all_model_names if n in subset_res and "metrics" in subset_res[n]]
            
            if not valid_models:
                continue
            
            report.append(f"\n## {label}\n")
            
            # 指令遵循子集使用专门的指标
            if subset_name == "instruction_following":
                # 指令遵循核心指标
                if_metrics_names = [
                    ("IFR (约束通过率)", "instruction_following_rate"),
                    ("Strict Acc (完全通过率)", "strict_accuracy"),
                    ("Loose Acc (宽松通过率)", "loose_accuracy"),
                ]
                
                # 构建表头
                header = "| 指标 |"
                separator = "|------|"
                for name in valid_models:
                    header += f" {name} |"
                    separator += "-------:|"
                report.append(header)
                report.append(separator)
                
                # 输出指令遵循指标
                for display_name, key in if_metrics_names:
                    row = f"| {display_name} |"
                    for name in valid_models:
                        val = subset_res[name]["metrics"].get(key, 0)
                        row += f" {val:.2f}% |"
                    report.append(row)
                
                # 指令遵循最佳模型
                report.append(f"\n### {label} - 最佳模型\n")
                report.append("| 指标 | 最佳模型 | 分数 |")
                report.append("|------|----------|------|")
                
                for display_name, key in if_metrics_names:
                    best_model = max(
                        valid_models,
                        key=lambda n: subset_res[n]["metrics"].get(key, 0)
                    )
                    best_score = subset_res[best_model]["metrics"].get(key, 0)
                    report.append(f"| {display_name} | {best_model} | {best_score:.2f}% |")
            else:
                # 非指令遵循子集使用通用指标
                # 构建表头
                header = "| 指标 |"
                separator = "|------|"
                for name in valid_models:
                    header += f" {name} |"
                    separator += "-------:|"
                report.append(header)
                report.append(separator)
                
                # 输出各指标
                for display_name, key in metrics_names:
                    row = f"| {display_name} |"
                    for name in valid_models:
                        val = subset_res[name]["metrics"].get(key, 0)
                        row += f" {val:.2f} |"
                    report.append(row)
                
                # 找出各指标的最佳模型
                report.append(f"\n### {label} - 最佳模型\n")
                report.append("| 指标 | 最佳模型 | 分数 |")
                report.append("|------|----------|------|")
                
                for display_name, key in metrics_names:
                    best_model = max(
                        valid_models,
                        key=lambda n: subset_res[n]["metrics"].get(key, 0)
                    )
                    best_score = subset_res[best_model]["metrics"].get(key, 0)
                    report.append(f"| {display_name} | {best_model} | {best_score:.2f} |")
        
        # 综合结论
        report.append("\n## 综合结论\n")
        
        for subset_name, label in subset_labels.items():
            if subset_name not in results:
                continue
            
            subset_res = results[subset_name]
            valid_models = [n for n in all_model_names if n in subset_res and "metrics" in subset_res[n]]
            
            if not valid_models:
                continue
            
            # 计算子集综合得分
            avg_scores = {}
            for name in valid_models:
                metrics = subset_res[name]["metrics"]
                if subset_name == "instruction_following":
                    # 指令遵循使用核心指标平均
                    if_metrics_keys = ["instruction_following_rate", "strict_accuracy", "loose_accuracy"]
                    avg = np.mean([metrics.get(k, 0) for k in if_metrics_keys])
                else:
                    # 其他子集使用通用指标平均
                    avg = np.mean([metrics.get(k, 0) for _, k in metrics_names])
                avg_scores[name] = avg
            
            sorted_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            report.append(f"\n**{label}综合排名**：\n")
            for i, (name, score) in enumerate(sorted_models, 1):
                report.append(f"{i}. **{name}**: 平均得分 {score:.2f}")
        
        report_path = os.path.join(output_dir, "comparison_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(f"对比报告已保存: {report_path}")
        
        # 打印终端摘要
        print("\n" + "=" * 80)
        print("多模型评估结果摘要（按子集）")
        print("=" * 80)
        
        for subset_name, label in subset_labels.items():
            if subset_name not in results:
                continue
            
            subset_res = results[subset_name]
            valid_models = [n for n in all_model_names if n in subset_res and "metrics" in subset_res[n]]
            
            if not valid_models:
                continue
            
            print(f"\n[{label}]")
            
            # 打印表头
            header_line = f"{'指标':<15}"
            for name in valid_models:
                # 截断过长的名称
                display_name = name[:12] if len(name) > 12 else name
                header_line += f"{display_name:>15}"
            print(header_line)
            print("-" * 80)
            
            for display_name, key in metrics_names:
                row = f"{display_name:<15}"
                for name in valid_models:
                    val = subset_res[name]["metrics"].get(key, 0)
                    row += f"{val:>15.2f}"
                print(row)
        
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="评估模型表现（支持微调对比和多模型对比）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  1. 微调前后对比:
     python evaluate.py --base-model path/to/base --adapter path/to/adapter

  2. 多模型对比:
     python evaluate.py --compare-models \\
       "微调1B:path/to/finetuned" \\
       "Qwen2.5-7B:Qwen/Qwen2.5-7B-Instruct" \\
       "LLaMA-3.1-8B:meta-llama/Llama-3.1-8B-Instruct"

  3. 带adapter的多模型对比:
     python evaluate.py --compare-models \\
       "基础1B:path/to/base" \\
       "微调1B:path/to/base:path/to/adapter"
        """
    )
    
    # 原有参数（微调对比模式）
    parser.add_argument(
        "--base-model", type=str, default=None, help="基础模型路径"
    )
    parser.add_argument(
        "--finetuned-model", type=str, default=None, help="全量微调后的模型路径"
    )
    parser.add_argument(
        "--adapter", type=str, default=None, help="LoRA adapter路径"
    )
    
    # 新增参数（多模型对比模式）
    parser.add_argument(
        "--compare-models", type=str, nargs="+", default=None,
        help="多模型对比，格式: '名称:模型路径' 或 '名称:模型路径:adapter路径'"
    )
    
    # 通用参数
    parser.add_argument(
        "--eval-file", type=str, default="data/val.json", help="评估数据文件"
    )
    parser.add_argument(
        "--max-samples", type=int, default=0, help="最大评估样本数（0表示使用全部样本）"
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation", help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 多模型对比模式
    if args.compare_models:
        print("\n" + "=" * 60)
        print("多模型对比评估模式")
        print("=" * 60)
        
        # 解析模型配置（支持Windows路径，如 D:/path）
        def parse_model_spec(spec: str):
            """
            解析模型配置字符串，支持Windows路径
            格式: 名称:模型路径 或 名称:模型路径:adapter路径
            Windows路径示例: 微调模型:D:/models/base:D:/adapters/lora
            """
            # 使用正则匹配，处理Windows驱动器字母
            import re
            # 匹配模式: name : [drive:]/path [: [drive:]/adapter]
            # Windows驱动器是单字母后跟冒号，如 D:
            pattern = r'^([^:]+):([A-Za-z]:)?([^:]+)(?::([A-Za-z]:)?([^:]+))?$'
            match = re.match(pattern, spec)
            
            if match:
                name = match.group(1)
                drive1 = match.group(2) or ""
                path1 = match.group(3)
                drive2 = match.group(4) or ""
                path2 = match.group(5)
                
                model_path = drive1 + path1
                adapter_path = (drive2 + path2) if path2 else None
                
                return name, model_path, adapter_path
            return None
        
        model_configs = []
        for model_spec in args.compare_models:
            result = parse_model_spec(model_spec)
            if result is None:
                print(f"错误: 无效的模型配置格式: {model_spec}")
                print("正确格式: '名称:模型路径' 或 '名称:模型路径:adapter路径'")
                print("示例: '微调模型:D:/models/base:outputs/lora'")
                return
            
            name, path, adapter = result
            model_configs.append({
                "name": name,
                "path": path,
                "adapter": adapter,
            })
        
        print(f"\n将评估 {len(model_configs)} 个模型:")
        for config in model_configs:
            adapter_info = f" (adapter: {config['adapter']})" if config['adapter'] else ""
            print(f"  - {config['name']}: {config['path']}{adapter_info}")
        
        evaluator = MultiModelEvaluator()
        evaluator.compare_multiple_models(
            model_configs=model_configs,
            eval_file=args.eval_file,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
        )
    
    # 原有的微调对比模式
    else:
        base_model = args.base_model or os.getenv("MODEL_PATH")
        if not base_model:
            print("错误: 请指定--base-model参数或设置MODEL_PATH环境变量")
            print("或使用--compare-models进行多模型对比")
            return
        
        evaluator = ModelEvaluator(
            base_model_path=base_model,
            finetuned_model_path=args.finetuned_model,
            adapter_path=args.adapter,
        )
        
        evaluator.compare_models(
            eval_file=args.eval_file,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
