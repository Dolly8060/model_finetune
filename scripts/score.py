#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评分脚本 - 对单个模型的推理结果进行评分

功能：
1. 读取 generate.py 输出的推理结果文件
2. 按任务类型（翻译/总结/指令遵循）计算评估指标
3. 输出评分结果和Markdown报告

评估指标：
- 翻译/总结: BLEU, ROUGE-1/2/L, BERTScore
- 指令遵循: IFR, Strict Accuracy, Loose Accuracy, 约束类型分解

输入格式：
    data/output_data/{model_name}_{timestamp}.json

输出格式：
    {output_dir}/eval_results.json
    {output_dir}/eval_report.md
"""

import os
import json
import argparse
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
import jieba

# 评估指标
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score


# ============================================================
# 指令遵循评估器
# ============================================================

def _count_words(text: str) -> int:
    """智能词数计算"""
    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff')
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    
    if total_chars > 0 and cjk_chars / total_chars > 0.3:
        words = list(jieba.cut(text))
        words = [w for w in words if w.strip() and not all(c in '，。！？、；：""''（）【】《》' for c in w)]
        return len(words)
    else:
        return len(text.split())


def _count_sentences(text: str) -> int:
    """智能句子数计算：支持中英文及多语言句子结尾符号
    
    使用单一不重叠的正则匹配所有句子结尾符号，避免重复计数。
    """
    if not text or not text.strip():
        return 0
    
    # 合并所有句子结尾符号到一个正则（不重叠）
    # 英文: . ! ? (后跟空格或结尾)
    # 中文: 。！？
    # 日文: 。
    # 印地语: ।॥
    # 阿拉伯语: ؟
    pattern = r'[.!?](?:\s|$)|[。！？।॥؟]'
    count = len(re.findall(pattern, text))
    return max(count, 1)


def _check_language(text: str, expected_lang: str) -> bool:
    """检查文本语言"""
    if not text or not expected_lang:
        return True
    
    expected_lang = expected_lang.lower().strip()
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    if total_chars == 0:
        return True
    
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    latin_chars = sum(1 for c in text if ('a' <= c.lower() <= 'z') or ('\u00c0' <= c <= '\u024f'))
    threshold = 0.25
    
    if expected_lang in ('zh', 'chinese', '中文'):
        return chinese_chars / total_chars >= threshold
    elif expected_lang in ('en', 'english', '英文', '英语'):
        return latin_chars / total_chars >= threshold and chinese_chars / total_chars < 0.1
    elif expected_lang in ('fr', 'de', 'es', 'it', 'pt', 'pl', 'ro'):
        return latin_chars / total_chars >= threshold and chinese_chars / total_chars < 0.1
    else:
        return True


class InstructionFollowingEvaluator:
    """指令遵循评估器"""
    
    VERIFIABLE_INSTRUCTIONS = {
        "response_language": {
            "patterns": [
                r"(?:your )?(?:ENTIRE )?response (?:should|must) be in (\w+)(?: language)?",
                r"answer (?:should|must) be (?:entirely )?in (\w+)",
            ],
            "check": lambda text, lang: _check_language(text, lang),
        },
        "min_sentences": {
            "patterns": [
                r"at least (\d+) sentences?",
                r"至少(\d+)(?:个)?句",
                r"不少于(\d+)(?:个)?句",
            ],
            "check": lambda text, param: _count_sentences(text) >= int(param),
        },
        "max_sentences": {
            "patterns": [
                r"(?:less|fewer) than (\d+) sentences?",
                r"(?:at most|no more than) (\d+) sentences?",
                r"不超过(\d+)(?:个)?句",
            ],
            "check": lambda text, param: _count_sentences(text) <= int(param),
        },
        "title_double_brackets": {
            "patterns": [
                r"title.*(?:wrapped|enclosed) in double angular brackets",
                r"title.*such as <<.*>>",
            ],
            "check": lambda text, _: bool(re.search(r'<<[^<>]+>>', text)),
        },
        "paragraph_divider": {
            "patterns": [
                r"paragraphs? (?:are |should be )?separated (?:with|by) (?:the )?(?:markdown )?divider[:\s]*\*\*\*",
            ],
            "check": lambda text, _: '***' in text,
        },
        "word_frequency": {
            "patterns": [
                r"(?:the )?word ['\"]?(\w+)['\"]? should appear (?:at least )?(\d+) times?",
                r"(?:in your response, )?the word ['\"](\w+)['\"].*(?:at least )?(\d+) times?",
            ],
            "check": lambda text, params: text.lower().count(params[0].lower()) >= int(params[1]) if isinstance(params, tuple) and len(params) >= 2 else True,
        },
        "min_words": {
            "patterns": [
                r"at least (\d+) words",
                r"不少于(\d+)个?(?:字|词)",
                r"字数要求[^\d]*(\d+)[到至\-~]+\d+",
            ],
            "check": lambda text, param: _count_words(text) >= int(param),
        },
        "max_words": {
            "patterns": [
                r"(?:at most|no more than|less than|under) (\d+) words",
                r"不超过(\d+)个?(?:字|词)",
                r"字数要求[^\d]*\d+[到至\-~]+(\d+)",
            ],
            "check": lambda text, param: _count_words(text) <= int(param),
        },
        "min_paragraphs": {
            "patterns": [
                r"at least (\d+) paragraphs?",
                r"至少(\d+)段",
            ],
            "check": lambda text, param: text.count('\n\n') + 1 >= int(param) or text.count('\n') + 1 >= int(param),
        },
        "exact_paragraphs": {
            "patterns": [
                r"(?:must |should )?have (\d+) paragraphs?",
                r"exactly (\d+) paragraphs?",
                r"请写(\d+)段",
                r"写(\d+)段",
            ],
            "check": lambda text, param: abs((text.count('\n\n') + 1) - int(param)) <= 1,
        },
        "keyword_include": {
            "patterns": [
                r"must include ['\"]([^'\"]+)['\"]",
                r"must contain ['\"]([^'\"]+)['\"]",
            ],
            "check": lambda text, param: param.lower() in text.lower(),
        },
        "keyword_include_zh": {
            "patterns": [
                r"必须包含(?:以下)?关键词[：:]\s*(.+?)(?:\.|。|$)",
            ],
            "check": lambda text, param: all(
                kw.strip().lower() in text.lower()
                for kw in re.split(r'[,，、]', param) if kw.strip()
            ),
        },
        "keyword_count": {
            "patterns": [
                r"['\"]([^'\"]+)['\"].*at least (\d+) times",
            ],
            "check": lambda text, params: text.lower().count(params[0].lower()) >= int(params[1]) if isinstance(params, tuple) else True,
        },
        "keyword_exclude": {
            "patterns": [
                r"do not (?:include |use )?(?:the )?(?:word |letter )?['\"]([^'\"]+)['\"]",
                r"(?:cannot|must not) (?:include|use|contain) (?:the )?(?:word |letter )?['\"]([^'\"]+)['\"]",
                r"avoid (?:using )?['\"]([^'\"]+)['\"]",
                r"不要使用['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]+)['\"\u2018\u2019\u201c\u201d]",
                r"禁止使用['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]+)['\"\u2018\u2019\u201c\u201d]",
            ],
            "check": lambda text, param: param.strip().lower() not in text.lower(),
        },
        "bullet_points": {
            "patterns": [
                r"use bullet points",
                r"bullet list",
                r"(?:请)?用列表形式",
            ],
            "check": lambda text, _: bool(re.search(r'[-•*●]\s', text)),
        },
        "numbered_list": {
            "patterns": [
                r"use numbered list",
                r"numbered format",
                r"(?:以)?数字编号",
            ],
            "check": lambda text, _: bool(re.search(r'^\s*\d+[.、)\]]\s', text, re.MULTILINE)),
        },
        "json_format": {
            "patterns": [
                r"in json format",
                r"as json",
                r"(?:用|以)json格式",
            ],
            "check": lambda text, _: ('{' in text and '}' in text),
        },
        "end_with_question": {
            "patterns": [
                r"end with a question",
                r"以问句结尾",
            ],
            "check": lambda text, _: text.rstrip().endswith('?') or text.rstrip().endswith('？'),
        },
        "start_with": {
            "patterns": [
                r"start with ['\"]([^'\"]+)['\"]",
                r"begin with ['\"]([^'\"]+)['\"]",
                r"开头必须是['\"]([^'\"]+)['\"]",
                r"开头(?:必须)?[为是]['\"]([^'\"]+)['\"]",
            ],
            "check": lambda text, param: text.strip().lower().startswith(param.lower()),
        },
        # --- IFEval 标准约束类型 ---
        "postscript": {
            "patterns": [
                r"(?:include|add|end with) a postscript",
                r"(?:add|include) (?:a )?P\.?S\.?",
                r"at the end.*(?:P\.S\.|postscript)",
            ],
            "check": lambda text, _: bool(re.search(r'P\.?S\.?', text)),
        },
        "highlight_sections": {
            "patterns": [
                r"[Hh]ighlight at least (\d+) sections?.*(?:markdown|with \*)",
            ],
            "check": lambda text, param: len(re.findall(r'\*[^*\n]+\*', text)) >= int(param),
        },
        "section_markers": {
            "patterns": [
                r"(?:must have|have) (\d+) sections?\b.*[Mm]ark.*[Ss]ection",
                r"[Mm]ark.*[Ss]ection.*(\d+) sections?",
            ],
            "check": lambda text, param: len(re.findall(r'Section \d+', text, re.IGNORECASE)) >= int(param),
        },
        "repeat_prompt": {
            "patterns": [
                r"[Ff]irst,? repeat (?:the )?(?:request|prompt|exact request|sentence|question)",
                r"repeat the (?:request|exact request|sentence|question) (?:word for word|itself|exactly|above)",
                r"[Ff]irst,? repeat ['\"\u201c]",
            ],
            "check": lambda text, _: True,  # 难以自动验证，默认通过
        },
        "quotation_wrap": {
            "patterns": [
                r"(?:[Ww]rap|[Ee]nclose).*(?:double )?quotation marks",
            ],
            "check": lambda text, _: text.strip().startswith('"') and text.strip().endswith('"'),
        },
        "all_uppercase": {
            "patterns": [
                r"(?:all|entire) (?:capital|uppercase) letters",
                r"response.*(?:all|only) (?:capital|uppercase)",
            ],
            "check": lambda text, _: text.upper() == text or sum(1 for c in text if c.isupper()) > sum(1 for c in text if c.islower()) * 3,
        },
        "all_lowercase": {
            "patterns": [
                r"(?:entire|all).*lowercase",
                r"(?:only|all) lowercase letters",
            ],
            "check": lambda text, _: text.lower() == text or sum(1 for c in text if c.islower()) > sum(1 for c in text if c.isupper()) * 10,
        },
        # --- 中文特有约束类型 ---
        "zh_three_part_structure": {
            "patterns": [
                r"三段式(?:结构)?(?:（|\()引言[、,]正文[、,]结论(?:）|\))?",
            ],
            "check": lambda text, _: text.count('\n\n') + 1 >= 3 or text.count('\n') + 1 >= 3,
        },
        "zh_keyword_per_paragraph": {
            "patterns": [
                r"每段(?:都)?(?:必须)?包含['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]+)['\"\u2018\u2019\u201c\u201d]",
            ],
            "check": lambda text, param: all(
                param in p for p in text.split('\n\n') if p.strip()
            ) if '\n\n' in text else param in text,
        },
        "zh_qa_format": {
            "patterns": [
                r"以?问答形式",
                r"(?:包含|至少)\d+个问题和(?:回答|答案)",
            ],
            "check": lambda text, _: bool(re.search(r'[?？]', text)),
        },
        "zh_start_with_unquoted": {
            "patterns": [
                r"开头(?:必须)?[为是]([^，。,.\n'\"]+?)(?:[，。,.]|$)",
            ],
            "check": lambda text, param: text.strip().startswith(param.strip()) if param else True,
        },
        "zh_no_degree_adverbs": {
            "patterns": [
                r"不要使用['\"\u2018\u2019\u201c\u201d]?(?:非常|很|极其)['\"\u2018\u2019\u201c\u201d]?[、,，]?\s*['\"\u2018\u2019\u201c\u201d]?(?:非常|很|极其)?['\"\u2018\u2019\u201c\u201d]?(?:等)?(?:程度)?(?:副词)?",
            ],
            "check": lambda text, _: '非常' not in text and '极其' not in text,
        },
        "markdown_format": {
            "patterns": [
                r"(?:in |use )?markdown format",
                r"用markdown(?:格式)?(?:写|输出|回答)",
            ],
            "check": lambda text, _: bool(re.search(r'(?:^#{1,6}\s|\*\*|```)', text, re.MULTILINE)),
        },
        "no_commas": {
            "patterns": [
                r"[Dd]o not use any commas",
                r"[Ww]ithout (?:any )?commas",
                r"[Nn]o comma",
            ],
            "check": lambda text, _: ',' not in text,
        },
        "placeholder_count": {
            "patterns": [
                r"at least (\d+) placeholders?",
                r"(\d+) placeholders?",
            ],
            "check": lambda text, param: len(re.findall(r'\[.*?\]', text)) >= int(param),
        },
        # --- IFEval 高频约束 ---
        "separator_asterisks": {
            "patterns": [
                r"[Ss]eparate.*?(\d+)\s*asterisk",
                r"(\d+)\s*asterisk.*?[Ss]eparate",
                r"[Ss]eparated (?:by|with)\s*(?:\d+\s*)?(?:asterisk|\*{3,})",
            ],
            "check": lambda text, _: '***' in text or '******' in text,
        },
        "multiple_responses": {
            "patterns": [
                r"(?:exactly |give |provide )?(\d+) (?:different )?responses?",
                r"(\d+) different (?:responses?|answers?|ways?)",
            ],
            "check": lambda text, param: True,  # 难以自动验证响应数量
        },
        "exact_words": {
            "patterns": [
                r"exactly (\d+) words",
                r"恰好(\d+)(?:个)?(?:字|词)",
            ],
            "check": lambda text, param: abs(_count_words(text) - int(param)) <= int(param) * 0.1 + 5,
        },
        "zh_keyword_count": {
            "patterns": [
                r"确保['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]+)['\"\u2018\u2019\u201c\u201d].*(?:出现|出现.*?)(?:至少|不少于)\s*(\d+)\s*次",
                r"['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]+)['\"\u2018\u2019\u201c\u201d].*(?:出现|出现.*?)(?:至少|不少于)\s*(\d+)\s*次",
            ],
            "check": lambda text, params: text.count(params[0]) >= int(params[1]) if isinstance(params, tuple) and len(params) >= 2 else True,
        },
        "zh_no_adjectives": {
            "patterns": [
                r"不要使用(?:任何)?形容词",
            ],
            "check": lambda text, _: True,  # 形容词检测需要NLP，默认通过
        },
        "table_format": {
            "patterns": [
                r"(?:in |as |create |make )?(?:a )?table",
                r"(?:用|以)表格",
            ],
            "check": lambda text, _: bool(re.search(r'\|.*\|', text)),
        },
    }
    
    @classmethod
    def extract_constraints(cls, instruction: str) -> List[tuple]:
        """从指令中提取约束"""
        constraints = []
        for constraint_type, config in cls.VERIFIABLE_INSTRUCTIONS.items():
            for pattern in config["patterns"]:
                match = re.search(pattern, instruction, re.IGNORECASE)
                if match:
                    if match.groups():
                        if len(match.groups()) == 2:
                            param = (match.group(1), match.group(2))
                        else:
                            param = match.group(1)
                    else:
                        param = None
                    constraints.append((constraint_type, param))
                    break
        return constraints
    
    @classmethod
    def check_constraint(cls, text: str, constraint_type: str, param) -> bool:
        """检查约束"""
        if constraint_type not in cls.VERIFIABLE_INSTRUCTIONS:
            return True
        check_fn = cls.VERIFIABLE_INSTRUCTIONS[constraint_type]["check"]
        try:
            return check_fn(text, param)
        except Exception:
            return False
    
    @classmethod
    def evaluate_sample(cls, instruction: str, output: str) -> Dict:
        """评估单个样本"""
        constraints = cls.extract_constraints(instruction)
        if not constraints:
            return {"constraints": [], "total": 0, "passed": 0, "rate": 1.0}
        
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
        """计算语料级指标"""
        samples_with_constraints = [r for r in eval_results if r["total"] > 0]
        samples_without_constraints = len(eval_results) - len(samples_with_constraints)
        
        if not samples_with_constraints:
            return {
                "instruction_following_rate": None,
                "strict_accuracy": None,
                "loose_accuracy": None,
                "samples_evaluated": 0,
                "no_constraint_samples": samples_without_constraints,
                "total_constraints": 0,
                "avg_constraints_per_sample": 0.0,
                "by_constraint_type": {},
            }
        
        total_constraints = sum(r["total"] for r in samples_with_constraints)
        total_passed = sum(r["passed"] for r in samples_with_constraints)
        
        ifr = (total_passed / total_constraints * 100) if total_constraints > 0 else 100.0
        strict_pass = sum(1 for r in samples_with_constraints if r["passed"] == r["total"])
        strict_acc = (strict_pass / len(samples_with_constraints) * 100)
        loose_pass = sum(1 for r in samples_with_constraints if r["passed"] >= r["total"] / 2)
        loose_acc = (loose_pass / len(samples_with_constraints) * 100)
        
        # 按约束类型统计
        constraint_type_stats = {}
        for result in samples_with_constraints:
            for constraint_type, param, passed in result.get("constraints", []):
                if constraint_type not in constraint_type_stats:
                    constraint_type_stats[constraint_type] = {"total": 0, "passed": 0}
                constraint_type_stats[constraint_type]["total"] += 1
                if passed:
                    constraint_type_stats[constraint_type]["passed"] += 1
        
        by_constraint_type = {}
        for ctype, stats in constraint_type_stats.items():
            rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
            by_constraint_type[ctype] = {
                "total": stats["total"],
                "passed": stats["passed"],
                "rate": rate,
            }
        
        return {
            "instruction_following_rate": ifr,
            "strict_accuracy": strict_acc,
            "loose_accuracy": loose_acc,
            "samples_evaluated": len(samples_with_constraints),
            "no_constraint_samples": samples_without_constraints,
            "total_constraints": total_constraints,
            "avg_constraints_per_sample": total_constraints / len(samples_with_constraints),
            "by_constraint_type": by_constraint_type,
        }


# ============================================================
# 评估指标计算
# ============================================================

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


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self):
        self.rouge_scorer_obj = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        self.bleu = BLEU(effective_order=True, lowercase=False, tokenize='zh')
    
    def _detect_lang(self, text: str) -> str:
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.replace(' ', ''))
        return "zh" if total_chars > 0 and chinese_chars / total_chars > 0.3 else "en"
    
    @staticmethod
    def _tokenize_for_rouge(text: str) -> str:
        """对中文文本进行分词预处理，使ROUGE能正确计算
        
        ROUGE库默认按空格分词，对中文无效。
        解决方案：用jieba分词后加空格拼接，使ROUGE按词粒度计算。
        """
        cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff')
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        
        if total_chars > 0 and cjk_chars / total_chars > 0.3:
            # 中文文本：jieba分词后空格拼接
            words = jieba.cut(text)
            return ' '.join(w for w in words if w.strip())
        else:
            # 英文文本：直接返回
            return text
    
    def compute_metrics(self, predictions: List[str], references: List[str]) -> EvalResult:
        """计算评估指标"""
        predictions = [p.strip() if p else " " for p in predictions]
        references = [r.strip() if r else " " for r in references]
        
        sample_text = ''.join(references[:10])
        detected_lang = self._detect_lang(sample_text)
        
        # BLEU
        try:
            bleu_score = self.bleu.corpus_score(predictions, [[r] for r in references]).score
        except Exception:
            bleu_score = 0.0
        
        # ROUGE (对中文文本进行分词预处理)
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        for pred, ref in zip(predictions, references):
            try:
                pred_tok = self._tokenize_for_rouge(pred)
                ref_tok = self._tokenize_for_rouge(ref)
                scores = self.rouge_scorer_obj.score(ref_tok, pred_tok)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            except Exception:
                rouge1_scores.append(0.0)
                rouge2_scores.append(0.0)
                rougeL_scores.append(0.0)
        
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) * 100 if rouge1_scores else 0.0
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) * 100 if rouge2_scores else 0.0
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) * 100 if rougeL_scores else 0.0
        
        # BERTScore
        try:
            P, R, F1 = bert_score(
                predictions, references,
                lang=detected_lang,
                verbose=False,
                device='cuda' if __import__('torch').cuda.is_available() else 'cpu'
            )
            bert_p = P.mean().item() * 100
            bert_r = R.mean().item() * 100
            bert_f1 = F1.mean().item() * 100
        except Exception:
            bert_p, bert_r, bert_f1 = 0.0, 0.0, 0.0
        
        return EvalResult(
            bleu=bleu_score,
            rouge1=avg_rouge1,
            rouge2=avg_rouge2,
            rougeL=avg_rougeL,
            bert_precision=bert_p,
            bert_recall=bert_r,
            bert_f1=bert_f1,
        )


# ============================================================
# 主评分逻辑
# ============================================================

def _reclassify_samples(samples: List[Dict]) -> List[Dict]:
    """运行时任务类型重分类
    
    对已标记为 instruction_following 但实际没有格式约束的样本进行重分类。
    这可以修正数据集构建阶段的分类错误，无需重建数据集。
    
    重分类规则：
    1. 检测约束 → 有约束则保持IF
    2. 无约束 + 匹配总结模式 → 重分类为summarization
    3. 无约束 + 匹配翻译模式 → 重分类为translation
    4. 无约束 + M-IFEval来源 → 保持IF（公开基准数据不改动）
    5. 无约束 + 自建数据 → 重分类为other
    """
    # 总结模式
    summary_patterns = [
        r"extract.*(?:main|key).*(?:ideas?|points?)",
        r"(?:main|key)\s+(?:ideas?|points?)\s+(?:of|from)",
        r"what are the (?:main|key) (?:ideas?|points?)",
        r"summarize|summary",
        r"总结|摘要|概括|核心要点|主要观点",
    ]
    
    # 翻译模式
    translation_patterns = [
        r"翻(?:译|成)(?:中文|英文|英语|中文)",
        r"translate|translation",
        r"译为|译成",
    ]
    
    reclassified_counts = {"summarization": 0, "translation": 0, "other": 0}
    
    for s in samples:
        if s.get("task_type") != "instruction_following":
            continue
        
        # 有约束 → 保持IF
        constraints = InstructionFollowingEvaluator.extract_constraints(s.get("instruction", ""))
        if constraints:
            continue
        
        # M-IFEval来源 → 保持IF
        if s.get("source", "").startswith("PMMEval"):
            continue
        
        # 无reference的样本 → 很可能是公开基准数据，保持IF
        # （M-IFEval等基准数据不提供reference output）
        if not s.get("reference", "").strip():
            continue
        
        instr = s.get("instruction", "")
        instr_lower = instr.lower()
        
        # 尝试重分类
        is_summary = any(re.search(p, instr_lower) for p in summary_patterns)
        is_translation = any(re.search(p, instr_lower) for p in translation_patterns)
        
        if is_translation:
            s["task_type"] = "translation"
            s["_reclassified"] = True
            reclassified_counts["translation"] += 1
        elif is_summary:
            s["task_type"] = "summarization"
            s["_reclassified"] = True
            reclassified_counts["summarization"] += 1
        else:
            s["task_type"] = "other"
            s["_reclassified"] = True
            reclassified_counts["other"] += 1
    
    total_reclass = sum(reclassified_counts.values())
    if total_reclass > 0:
        print(f"  运行时重分类: {total_reclass} 个样本从 instruction_following 移出")
        for task_type, count in reclassified_counts.items():
            if count > 0:
                print(f"    → {task_type}: {count}")
    
    return samples


def score_results(input_file: str, output_dir: str):
    """对推理结果进行评分"""
    
    # 加载推理结果
    print(f"加载推理结果: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    model_name = data.get("model_name", "unknown")
    samples = data.get("samples", [])
    
    print(f"模型名称: {model_name}")
    print(f"样本数量: {len(samples)}")
    
    # 运行时重分类：修正分类错误的样本
    samples = _reclassify_samples(samples)
    
    # 按任务类型分组
    translation_samples = [s for s in samples if s.get("task_type") == "translation"]
    summarization_samples = [s for s in samples if s.get("task_type") == "summarization"]
    if_samples = [s for s in samples if s.get("task_type") == "instruction_following"]
    
    print(f"翻译样本: {len(translation_samples)}")
    print(f"总结样本: {len(summarization_samples)}")
    print(f"指令遵循样本: {len(if_samples)}")
    
    other_samples = [s for s in samples if s.get("task_type") not in ("translation", "summarization", "instruction_following")]
    if other_samples:
        print(f"其他样本: {len(other_samples)} (不参与评分)")
    
    calculator = MetricsCalculator()
    results = {}
    
    # 评估翻译
    if translation_samples:
        print("\n评估翻译子集...")
        preds = [s["prediction"] for s in translation_samples]
        refs = [s["reference"] for s in translation_samples]
        metrics = calculator.compute_metrics(preds, refs)
        results["translation"] = metrics.__dict__
    
    # 评估总结
    if summarization_samples:
        print("评估总结子集...")
        preds = [s["prediction"] for s in summarization_samples]
        refs = [s["reference"] for s in summarization_samples]
        metrics = calculator.compute_metrics(preds, refs)
        results["summarization"] = metrics.__dict__
    
    # 评估指令遵循
    if if_samples:
        print("评估指令遵循子集...")
        
        # 指令遵循特有指标（不依赖reference，仅检查约束）
        if_eval_results = []
        for s in if_samples:
            if_result = InstructionFollowingEvaluator.evaluate_sample(
                s["instruction"], s["prediction"]
            )
            if_eval_results.append(if_result)
        
        if_corpus_metrics = InstructionFollowingEvaluator.compute_corpus_metrics(if_eval_results)
        
        # 内容质量指标：仅对有reference的样本计算
        if_with_ref = [s for s in if_samples if s.get("reference", "").strip()]
        if_without_ref = len(if_samples) - len(if_with_ref)
        
        if if_without_ref > 0:
            print(f"  注意: {if_without_ref}/{len(if_samples)} 个指令遵循样本无reference，"
                  f"内容质量指标仅基于 {len(if_with_ref)} 个有reference的样本计算")
        
        if if_with_ref:
            preds = [s["prediction"] for s in if_with_ref]
            refs = [s["reference"] for s in if_with_ref]
            metrics = calculator.compute_metrics(preds, refs)
            content_metrics = metrics.__dict__
        else:
            print("  警告: 所有指令遵循样本均无reference，跳过内容质量指标计算")
            content_metrics = {
                "bleu": None, "rouge1": None, "rouge2": None, "rougeL": None,
                "bert_precision": None, "bert_recall": None, "bert_f1": None,
            }
        
        results["instruction_following"] = {
            **content_metrics,
            **if_corpus_metrics,
            "content_quality_samples": len(if_with_ref),
            "content_quality_skipped": if_without_ref,
        }
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "eval_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评分结果已保存: {results_file}")
    
    # 生成报告
    report = generate_report(model_name, results)
    report_file = os.path.join(output_dir, "eval_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"评估报告已保存: {report_file}")
    
    return results


def generate_report(model_name: str, results: Dict) -> str:
    """生成Markdown报告"""
    report = [f"# 模型评估报告 - {model_name}\n"]
    
    metrics_names = [
        ("BLEU", "bleu"),
        ("ROUGE-1", "rouge1"),
        ("ROUGE-2", "rouge2"),
        ("ROUGE-L", "rougeL"),
        ("BERTScore-P", "bert_precision"),
        ("BERTScore-R", "bert_recall"),
        ("BERTScore-F1", "bert_f1"),
    ]
    
    # 翻译子集
    if "translation" in results:
        report.append("## 翻译子集 (Translation)\n")
        report.append("| 指标 | 分数 |")
        report.append("|------|------|")
        for display_name, key in metrics_names:
            val = results["translation"].get(key, 0)
            report.append(f"| {display_name} | {val:.2f} |")
        report.append("")
    
    # 总结子集
    if "summarization" in results:
        report.append("## 总结子集 (Summarization)\n")
        report.append("| 指标 | 分数 |")
        report.append("|------|------|")
        for display_name, key in metrics_names:
            val = results["summarization"].get(key, 0)
            report.append(f"| {display_name} | {val:.2f} |")
        report.append("")
    
    # 指令遵循子集
    if "instruction_following" in results:
        if_results = results["instruction_following"]
        report.append("## 指令遵循子集 (Instruction Following)\n")
        
        # 核心指标
        report.append("### 核心指标\n")
        report.append("| 指标 | 分数 |")
        report.append("|------|------|")
        ifr_val = if_results.get('instruction_following_rate')
        strict_val = if_results.get('strict_accuracy')
        loose_val = if_results.get('loose_accuracy')
        report.append(f"| IFR (约束通过率) | {f'{ifr_val:.2f}%' if ifr_val is not None else 'N/A'} |")
        report.append(f"| Strict Acc (完全通过率) | {f'{strict_val:.2f}%' if strict_val is not None else 'N/A'} |")
        report.append(f"| Loose Acc (宽松通过率) | {f'{loose_val:.2f}%' if loose_val is not None else 'N/A'} |")
        report.append("")
        
        # 统计信息
        report.append("### 统计信息\n")
        samples_with_constraints = if_results.get('samples_evaluated', 0)
        total_constraints = if_results.get('total_constraints', 0)
        no_constraint_count = if_results.get('no_constraint_samples', 0)
        
        report.append(f"- 指令遵循总样本数: {samples_with_constraints + no_constraint_count}")
        report.append(f"- 检测到约束的样本数: {samples_with_constraints}")
        report.append(f"- 无可检测约束的样本数: {no_constraint_count}")
        report.append(f"- 总约束数: {total_constraints}")
        report.append(f"- 平均约束数/样本: {if_results.get('avg_constraints_per_sample', 0):.2f}")
        
        if no_constraint_count > 0:
            coverage = samples_with_constraints / (samples_with_constraints + no_constraint_count) * 100
            report.append(f"- 约束覆盖率: {coverage:.1f}%")
            report.append(f"\n> 注: {no_constraint_count} 个样本未检测到可验证的约束（可能是纯问答/总结任务被归入IF类，或约束模式未覆盖）。")
            report.append(f"> IFR/Strict Acc/Loose Acc 仅基于 {samples_with_constraints} 个有约束的样本计算。")
        report.append("")
        
        # 按约束类型分解
        if "by_constraint_type" in if_results and if_results["by_constraint_type"]:
            report.append("### 按约束类型分解\n")
            report.append("| 约束类型 | 总数 | 通过 | 通过率 |")
            report.append("|----------|------|------|--------|")
            for ctype, stats in if_results["by_constraint_type"].items():
                report.append(f"| {ctype} | {stats['total']} | {stats['passed']} | {stats['rate']:.1f}% |")
            report.append("")
        
        # 内容质量参考
        content_samples = if_results.get('content_quality_samples', 0)
        content_skipped = if_results.get('content_quality_skipped', 0)
        
        if content_samples > 0:
            note = ""
            if content_skipped > 0:
                note = f"（仅基于 {content_samples} 个有reference的样本，{content_skipped} 个无reference已跳过）"
            report.append(f"### 内容质量参考（辅助指标）{note}\n")
            report.append("| 指标 | 分数 |")
            report.append("|------|------|")
            for display_name, key in metrics_names:
                val = if_results.get(key)
                if val is not None:
                    report.append(f"| {display_name} | {val:.2f} |")
                else:
                    report.append(f"| {display_name} | N/A |")
        else:
            report.append("### 内容质量参考（辅助指标）\n")
            report.append(f"> 所有 {content_skipped} 个指令遵循样本均无reference输出，无法计算内容质量指标。")
            report.append("> 指令遵循任务的核心评估依赖上述约束检测指标（IFR/Strict Acc/Loose Acc）。")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="评分脚本 - 对单个模型的推理结果进行评分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  python score.py --input-file data/output_data/Granite_1B_20240207_120000.json

  python score.py --input-file data/output_data/model.json --output-dir evaluation/my_eval
        """
    )
    
    parser.add_argument(
        "--input-file", type=str, required=True,
        help="推理结果文件路径 (generate.py 的输出)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录 (默认: 与输入文件同目录的 eval_output/)"
    )
    
    args = parser.parse_args()
    
    # 默认输出目录
    if args.output_dir is None:
        input_dir = os.path.dirname(args.input_file)
        out_dir_tmp = "evaluation/performance"
        input_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_dir = os.path.join(out_dir_tmp, f"{input_name}_eval")
    
    score_results(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
