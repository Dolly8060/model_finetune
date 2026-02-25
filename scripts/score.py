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
from functools import lru_cache
from typing import List, Dict, Optional, Tuple, Any
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


def _first_alpha_char(text: str) -> str:
    """Return first latin alphabetic char for lightweight case constraints."""
    for ch in (text or "").strip():
        if 'a' <= ch.lower() <= 'z':
            return ch
    return ""


def _check_first_letter_lowercase(text: str, _param=None) -> bool:
    ch = _first_alpha_char(text)
    return (not ch) or ch.islower()


def _normalize_simple_reply(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r'^[\"\'“”‘’\\s]+|[\"\'“”‘’\\s\\.!?。！？]+$', '', t)
    return t.lower()


def _check_reply_only_choices(text: str, params) -> bool:
    if not isinstance(params, tuple) or len(params) < 2:
        return True
    pred = _normalize_simple_reply(text)
    options = [_normalize_simple_reply(x) for x in params if str(x).strip()]
    return pred in options if options else True


def _compare_number(actual: int, relation: Optional[str], target: int) -> bool:
    rel = (relation or "").strip().lower()
    if rel in ("at least", ">=", "ge", "no less than", "minimum"):
        return actual >= target
    if rel in ("at most", "<=", "le", "no more than", "maximum"):
        return actual <= target
    if rel in ("less than", "<", "lt"):
        return actual < target
    if rel in ("more than", "greater than", ">", "gt"):
        return actual > target
    if rel in ("exactly", "equal", "equals", "=="):
        return actual == target
    # default: lenient exact
    return actual == target


def _check_letter_frequency(text: str, params) -> bool:
    if not isinstance(params, tuple) or len(params) < 3:
        return True
    letter, relation, target = params[0], params[1], int(params[2])
    actual = (text or "").count(letter)
    return _compare_number(actual, relation, target)


def _check_capital_word_frequency(text: str, params) -> bool:
    if not isinstance(params, tuple) or len(params) < 2:
        return True
    relation, target = params[0], int(params[1])
    words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", text or "")
    # capitalized/uppercase words both count as "capital words"
    actual = sum(1 for w in words if w[0].isupper())
    return _compare_number(actual, relation, target)


def _check_nth_paragraph_first_word(text: str, params) -> bool:
    if not isinstance(params, tuple) or len(params) < 2:
        return True
    try:
        nth = int(params[0])
        expected = str(params[1]).strip().lower()
    except Exception:
        return True
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if not paras:
        paras = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if nth <= 0 or nth > len(paras):
        return False
    m = re.search(r"[A-Za-z0-9\u4e00-\u9fff\u3040-\u30ff\u0600-\u06ff\u0C80-\u0CFF]+", paras[nth - 1])
    if not m:
        return False
    return m.group(0).lower() == expected


def _check_end_with_phrase(text: str, param) -> bool:
    end_phrase = (param or "").strip()
    if not end_phrase:
        return True
    pred = (text or "").strip()
    return pred.endswith(end_phrase)


def _merge_constraints(base: List[tuple], extra: List[tuple]) -> List[tuple]:
    merged = list(base or [])
    seen = {(c, json.dumps(p, ensure_ascii=False, sort_keys=True) if isinstance(p, (dict, list, tuple)) else str(p))
            for c, p in merged}
    for c, p in (extra or []):
        key = (c, json.dumps(p, ensure_ascii=False, sort_keys=True) if isinstance(p, (dict, list, tuple)) else str(p))
        if key not in seen:
            merged.append((c, p))
            seen.add(key)
    return merged


def _candidate_ifeval_paths() -> List[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    return [
        os.path.join(cwd, "dataset", "IFEval", "input_data.jsonl"),
        os.path.join(os.path.dirname(script_dir), "dataset", "IFEval", "input_data.jsonl"),
    ]


def _candidate_mifeval_dirs() -> List[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    return [
        os.path.join(cwd, "dataset", "m-ifeval"),
        os.path.join(os.path.dirname(script_dir), "dataset", "m-ifeval"),
    ]


@lru_cache(maxsize=1)
def _load_ifeval_prompt_index() -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for path in _candidate_ifeval_paths():
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    prompt = (rec.get("prompt") or "").strip()
                    if prompt and prompt not in index:
                        index[prompt] = rec
            if index:
                break
        except Exception:
            continue
    return index


@lru_cache(maxsize=1)
def _load_mifeval_prompt_index() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Build prompt -> IFEval metadata map for each m-IFEval language source by:
    mifeval-en(key -> English prompt) --exact match--> IFEval metadata
    then transfer by shared key to all language files.
    Returns: {source_name: {prompt_text: ifeval_record}}
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    ifeval_map = _load_ifeval_prompt_index()
    mdir = None
    for cand in _candidate_mifeval_dirs():
        if os.path.isdir(cand):
            mdir = cand
            break
    if not mdir:
        return out

    en_path = os.path.join(mdir, "PMMEval-mifeval-en.json")
    if not os.path.exists(en_path):
        return out
    try:
        with open(en_path, "r", encoding="utf-8") as f:
            en_data = json.load(f)
    except Exception:
        return out

    key_to_ifeval: Dict[str, Dict[str, Any]] = {}
    for key, rec in en_data.items():
        try:
            prompt = (((rec or {}).get("origin_prompt") or [])[0] or {}).get("prompt", "")
        except Exception:
            prompt = ""
        prompt = (prompt or "").strip()
        if prompt and prompt in ifeval_map:
            key_to_ifeval[key] = ifeval_map[prompt]

    # Build per-language prompt maps
    for fn in os.listdir(mdir):
        if not fn.lower().startswith("pmmeval-mifeval-") or not fn.lower().endswith(".json"):
            continue
        path = os.path.join(mdir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        lang = fn.rsplit("-", 1)[-1].replace(".json", "").lower()
        source_name = f"mifeval_{lang}"
        prompt_map: Dict[str, Dict[str, Any]] = {}
        for key, rec in data.items():
            meta_rec = key_to_ifeval.get(key)
            if not meta_rec:
                continue
            try:
                prompt = (((rec or {}).get("origin_prompt") or [])[0] or {}).get("prompt", "")
            except Exception:
                prompt = ""
            prompt = (prompt or "").strip()
            if prompt:
                prompt_map[prompt] = meta_rec
        out[source_name] = prompt_map
    return out


def _parse_lria_choice_options_from_reference(sample: Dict) -> List[str]:
    ref = (sample.get("reference") or "").strip()
    if not ref:
        return []
    # Normalize common wrappers/spaces
    ref_norm = ref.replace("（", "(").replace("）", ")").strip()
    # Common binary choices
    binary_aliases = {
        "是/否": ["是", "否"],
        "对/错": ["对", "错"],
        "正确/错误": ["正确", "错误"],
        "yes/no": ["yes", "no"],
        "true/false": ["true", "false"],
    }
    lower = ref_norm.lower()
    for k, vals in binary_aliases.items():
        if lower == k.lower():
            return vals
    if "/" not in ref_norm:
        return []
    parts = [p.strip() for p in ref_norm.split("/") if p.strip()]
    if not (2 <= len(parts) <= 8):
        return []
    # Keep only short option-like tokens to avoid abusing reference as content label
    cleaned = []
    for p in parts:
        p2 = p.strip().strip("()[]{}")
        if len(p2) <= 16 and "\n" not in p2:
            cleaned.append(p2)
    return cleaned if len(cleaned) == len(parts) else []


def _looks_like_choice_judge_prompt(text: str) -> bool:
    t = (text or "").lower()
    patterns = [
        r"\bchoose\b", r"\bselect\b", r"\bonly (?:reply|answer|respond)\b",
        r"yes or no", r"true or false",
        r"\u9009\u51fa", r"\u9009\u62e9", r"\u4ec5\u56de\u7b54", r"\u53ea\u56de\u7b54",
        r"\u662f\u6216\u5426", r"\u6b63\u786e\u6216\u9519\u8bef", r"\u5224\u65ad",
    ]
    return any(re.search(p, t, re.IGNORECASE) for p in patterns)


def _normalize_lria_label(text: str) -> str:
    t = (text or "").strip()
    # remove think tags if any leaked into prediction
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.IGNORECASE | re.DOTALL).strip()
    # strip surrounding quotes/punctuation and whitespace
    t = re.sub(r'^[\s"\'“”‘’`]+|[\s"\'“”‘’`]+$', '', t)
    return t.casefold()


def _check_lria_reference_exact_short(text: str, param) -> bool:
    ref = _normalize_lria_label(str(param or ""))
    pred = _normalize_lria_label(text)
    if not ref:
        return False
    return pred == ref


def _check_lria_language_or_code(text: str, param) -> bool:
    label = (str(param or "").strip()).casefold()
    if not label:
        return False
    # natural language labels
    if label in ("english", "en", "英文", "英语"):
        return _check_language(text, "en")
    if label in ("chinese", "zh", "中文", "汉语", "华语"):
        return _check_language(text, "zh")
    if label in ("german", "de", "德语"):
        return _check_language(text, "de")
    if label in ("french", "fr", "法语"):
        return _check_language(text, "fr")
    if label in ("spanish", "es", "西班牙语"):
        return _check_language(text, "es")
    if label in ("japanese", "ja", "日语", "日本语"):
        # reuse CJK-heavy heuristic (not ideal, but better than exact label)
        return bool(re.search(r'[\u3040-\u30ff]', text or ""))
    # programming languages
    txt = (text or "")
    if label in ("go", "golang"):
        return bool(re.search(r"\bpackage\s+main\b|\bfunc\s+\w+\s*\(|time\.Sleep|fmt\.", txt))
    if label in ("python", "py"):
        return bool(re.search(r"\bdef\s+\w+\s*\(|import\s+\w+|print\s*\(", txt))
    if label in ("javascript", "js"):
        return bool(re.search(r"\bfunction\b|const\s+\w+|let\s+\w+|console\.log", txt))
    if label in ("java"):
        return bool(re.search(r"\bpublic\s+class\b|\bpublic\s+static\s+void\s+main\b|System\.out\.println", txt))
    if label in ("c++", "cpp"):
        return bool(re.search(r"#include\s*<|std::|int\s+main\s*\(", txt))
    if label in ("c",):
        return bool(re.search(r"#include\s*<|printf\s*\(|int\s+main\s*\(", txt))
    # fallback: exact short label
    return _check_lria_reference_exact_short(text, label)


def _candidate_strict_labeled_paths() -> List[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    return [
        os.path.join(cwd, "data", "qwen3_strict_test_labeled.json"),
        os.path.join(os.path.dirname(script_dir), "data", "qwen3_strict_test_labeled.json"),
    ]


@lru_cache(maxsize=1)
def _load_strict_labeled_if_index() -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for path in _candidate_strict_labeled_paths():
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for s in data:
                if s.get("task_type") != "instruction_following":
                    continue
                key = (
                    (s.get("instruction") or "").strip(),
                    (s.get("input") or "").strip(),
                    str(s.get("source") or ""),
                    str(s.get("language") or ""),
                )
                out[key] = s
            if out:
                break
        except Exception:
            continue
    return out


class InstructionFollowingEvaluator:
    """指令遵循评估器"""
    
    ENABLE_LRIA_FALLBACK = False

    VERIFIABLE_INSTRUCTIONS = {
        "response_language": {
            "patterns": [
                r"(?:your )?(?:ENTIRE )?response (?:should|must) be in (\w+)(?: language)?",
                r"answer (?:should|must) be (?:entirely )?in (\w+)",
                r"(?:please )?(?:reply|respond|answer) in (\w+)",
                r"in only (\w+)(?:,|\\s)",
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
                r"menos de (\d+) frases?",
                r"moins de (\d+) phrases?",
                r"不超过(\d+)(?:个)?句",
            ],
            "check": lambda text, param: _count_sentences(text) <= int(param),
        },
        "exact_sentences": {
            "patterns": [
                r"exactly (\d+) sentences?",
                r"\u6070\u597d\s*(\d+)\s*\u53e5",
            ],
            "check": lambda text, param: _count_sentences(text) == int(param),
        },
        "title_double_brackets": {
            "patterns": [
                r"title.*(?:wrapped|enclosed) in double angular brackets",
                r"title.*such as <<.*>>",
                r"<<[^<>]{0,40}>>",
                r"double angular brackets",
                r"paranteze unghiulare duble",
            ],
            "check": lambda text, _: bool(re.search(r'<<[^<>]+>>', text)),
        },
        "paragraph_divider": {
            "patterns": [
                r"paragraphs? (?:are |should be )?separated (?:with|by) (?:the )?(?:markdown )?divider[:\s]*\*\*\*",
                r"(?:paragraphs?|abs[aä]tze?) .*?(?:double )?line breaks?",
                r"\\n\\n",
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
        "letter_frequency": {
            "patterns": [
                r"(?:letter|character) ['\"](.+?)['\"].*(?:at least|at most|less than|more than|exactly) (\d+) times",
            ],
            "check": _check_letter_frequency,
        },
        "min_words": {
            "patterns": [
                r"at least (\d+) words",
                r"al menos (\d+) palabras",
                r"au moins (\d+) mots",
                r"pelo menos (\d+) palavras",
                r"almeno (\d+) parole",
                r"mindestens (\d+) w[oö]rter",
                r"cel pu(?:\u021bin|tin) (\d+) cuvinte",
                r"co najmniej (\d+) s[\u0142l][oó]w",
                r"\u5c11\u306a\u304f\u3068\u3082\s*(\d+)\s*\u8a9e",
                r"\u81f3\u5c11\s*(\d+)\s*\u4e2a?\u5b57",
                r"\u81f3\u5c11\s*(\d+)\s*\u4e2a?\u8bcd",
                r"不少于(\d+)个?(?:字|词)",
                r"字数要求[^\d]*(\d+)[到至\-~]+\d+",
            ],
            "check": lambda text, param: _count_words(text) >= int(param),
        },
        "max_words": {
            "patterns": [
                r"(?:at most|no more than|less than|under) (\d+) words",
                r"(?:anything )?longer than (\d+) words",
                r"(\d+) words? or less",
                r"no m[aá]s de (\d+) palabras",
                r"pas plus de (\d+) mots",
                r"nicht mehr als (\d+) w[oö]rter",
                r"cel mult (\d+) cuvinte",
                r"\u4e0d\u8d85\u8fc7\s*(\d+)\s*\u4e2a?\u5b57",
                r"\u4e0d\u8d85\u8fc7\s*(\d+)\s*\u4e2a?\u8bcd",
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
                r"exactamente (\d+) p[aá]rrafos?",
                r"exactement (\d+) paragraphes?",
                r"esattamente (\d+) paragrafi",
                r"genau (\d+) abs[aä]tze?",
                r"exact (\d+) paragrafe",
                r"\u0641\u064a\s*(\d+)\s*\u0641\u0642\u0631\u0627\u062a?.*?\u0628\u0627\u0644\u0636\u0628\u0637",
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
                r"don't (?:include |use |contain )?(?:the )?(?:word |letter )?['\"]([^'\"]+)['\"]",
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
                r"bullet points?",
                r"(?:请)?用列表形式",
            ],
            "check": lambda text, _: bool(re.search(r'[-•*●]\s', text)),
        },
        "exact_bullet_points": {
            "patterns": [
                r"exactly (\d+) bullet points?",
                r"exact (\d+) bullet points?",
                r"\u6070\u597d\s*(\d+)\s*\u4e2a?\u8981\u70b9",
            ],
            "check": lambda text, param: len(
                [ln for ln in (text or "").splitlines() if re.match(r'^\s*(?:[-*•]|\d+[.)])\s+', ln)]
            ) == int(param),
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
                r"format json",
                r"formato json",
                r"(?:用|以)json格式",
            ],
            "check": lambda text, _: ('{' in text and '}' in text),
        },
        "reply_only_choices": {
            "patterns": [
                r"(?:only|just) (?:reply|respond|answer) with ['\"]([^'\"]+)['\"]\s*(?:or|/)\s*['\"]([^'\"]+)['\"]",
                r"\u4ec5\u56de\u590d[\u201c\"]([^\"\u201d]+)[\u201d\"]\s*\u6216\s*[\u201c\"]([^\"\u201d]+)[\u201d\"]",
                r"\u53ea\u56de\u590d[\u201c\"]([^\"\u201d]+)[\u201d\"]\s*\u6216\s*[\u201c\"]([^\"\u201d]+)[\u201d\"]",
            ],
            "check": _check_reply_only_choices,
        },
        "first_letter_lowercase": {
            "patterns": [
                r"first letter (?:should be |must be )?lowercase",
                r"lowercase first letter",
                r"\u9996\u5b57\u6bcd\u5c0f\u5199",
            ],
            "check": _check_first_letter_lowercase,
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
        "end_with_phrase": {
            "patterns": [
                r"end with ['\"]([^'\"]+)['\"]",
                r"\u4ee5['\"\u201c]([^\"\u201d]+)['\"\u201d]\u7ed3\u5c3e",
            ],
            "check": _check_end_with_phrase,
        },
        "postscript": {
            "patterns": [
                r"(?:include|add|end with) a postscript",
                r"(?:add|include) (?:a )?P\.?S\.?",
                r"at the end.*(?:P\.S\.|postscript)",
                r"P\.?P\.?S\.?",
            ],
            "check": lambda text, _: bool(re.search(r'P\.?S\.?|P\.?P\.?S\.?', text)),
        },
        "highlight_sections": {
            "patterns": [
                r"[Hh]ighlight at least (\d+) sections?.*(?:markdown|with \*)",
                r"[Mm]arkier.*mindestens (\d+) abschnitte.*markdown",
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
                r"r[eé]p[eé]t(?:ez|e) d'abord",
                r"primeiro repita",
                r"per prima cosa,? ripeti",
                r"zuerst .*wiederhol",
                r"primero repite",
                r"najpierw powt[oó]rz",
                r"\u9996\u5148.*\u91cd\u590d",
                r"\u5148.*\u91cd\u590d",
                r"\u0623\u0648\u0644[\u0627\u064b]?\u061f?.*\u0643\u0631\u0631",
            ],
            "check": lambda text, _: True,  # 难以自动验证，默认通过
        },
        "quotation_wrap": {
            "patterns": [
                r"(?:[Ww]rap|[Ee]nclose).*(?:double )?quotation marks",
                r"double\s+quotes?",
                r"comillas dobles",
                r"guillemets doubles?|doubles guillemets",
                r"aspas duplas",
                r"virgolette doppie",
                r"ghilimele duble",
                r"doppelte[nr]?\s+anf[\u00fcu]hrungszeichen",
                r"\u53cc\u5f15\u53f7",
            ],
            "check": lambda text, _: text.strip().startswith('"') and text.strip().endswith('"'),
        },
        "all_uppercase": {
            "patterns": [
                r"(?:all|entire) (?:capital|uppercase) letters",
                r"response.*(?:all|only) (?:capital|uppercase)",
                r"capitalize all your words",
                r"all your words.*capitaliz",
            ],
            "check": lambda text, _: text.upper() == text or sum(1 for c in text if c.isupper()) > sum(1 for c in text if c.islower()) * 3,
        },
        "capital_word_frequency": {
            "patterns": [
                r"capital(?:ized)? words?.*(?:at least|at most|less than|more than) (\d+)",
            ],
            "check": _check_capital_word_frequency,
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
                r"\bmarkdown\b",
                r"用markdown(?:格式)?(?:写|输出|回答)",
            ],
            "check": lambda text, _: bool(re.search(r'(?:^#{1,6}\s|\*\*|```)', text, re.MULTILINE)),
        },
        "no_commas": {
            "patterns": [
                r"[Dd]o not use any commas",
                r"[Ww]ithout (?:using )?(?:any )?commas",
                r"[Nn]o comma",
                r"sin comas",
                r"sans virgules?",
                r"sem v[íi]rgulas?",
                r"ohne kommas?",
                r"senza virgole",
                r"bez przecink",
                r"nu folose[\u0219s]ti? virgule",
                r"folosirea virgulelor",
                r"f[ăa]r[ăa] virgule",
                r"\u4e0d\u8981.*\u9017\u53f7",
                r"\u4e0d.*\u4f7f\u7528.*\u9017\u53f7",
            ],
            "check": lambda text, _: ',' not in text,
        },
        "placeholder_count": {
            "patterns": [
                r"at least (\d+) placeholders?",
                r"(\d+) placeholders?",
                r"mindestens (\d+) platzhalter",
                r"(\d+) platzhalter",
                r"\u81f3\u5c11\u5305\u542b\s*(\d+)\s*\u4e2a.*\u5360\u4f4d\u7b26",
                r"\u81f3\u5c11\s*(\d+)\s*\u4e2a.*\u5360\u4f4d\u7b26",
            ],
            "check": lambda text, param: len(re.findall(r'\[.*?\]', text)) >= int(param),
        },
        # --- IFEval 高频约束 ---
        "separator_asterisks": {
            "patterns": [
                r"[Ss]eparate.*?(\d+)\s*asterisk",
                r"(\d+)\s*asterisk.*?[Ss]eparate",
                r"[Ss]eparated (?:by|with)\s*(?:\d+\s*)?(?:asterisk|\*{3,})",
                r"asterisc",
                r"ast[eé]risc",
                r"asterisco",
                r"\*{4,}",
            ],
            "check": lambda text, _: '***' in text or '******' in text,
        },
        "multiple_responses": {
            "patterns": [
                r"(?:exactly |give |provide )?(\d+) (?:different )?responses?",
                r"(\d+) different (?:responses?|answers?|ways?)",
                r"(\d+) respuestas? diferentes",
                r"(\d+) r[eé]ponses? diff[eé]rentes?",
                r"(\d+) respostas? diferentes",
                r"(\d+) risposte? diverse",
                r"(\d+) r\u0103spunsuri diferite",
                r"(\d+) verschiedene antworten",
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
        "lria_reference_exact_short": {
            "patterns": [],
            "check": _check_lria_reference_exact_short,
        },
        "lria_language_or_code": {
            "patterns": [],
            "check": _check_lria_language_or_code,
        },
        "nth_paragraph_first_word": {
            "patterns": [
                r"the first word of the (\d+)(?:st|nd|rd|th) paragraph.*?['\"]([^'\"]+)['\"]",
            ],
            "check": _check_nth_paragraph_first_word,
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
    def _map_ifeval_instruction(cls, instruction_id: str, kwargs: Dict[str, Any]) -> List[tuple]:
        iid = (instruction_id or "").strip()
        kw = kwargs or {}
        if iid == "punctuation:no_comma":
            return [("no_commas", None)]
        if iid == "detectable_format:number_highlighted_sections":
            n = kw.get("num_highlights")
            return [("highlight_sections", str(n))] if n is not None else []
        if iid == "length_constraints:number_words":
            n = kw.get("num_words")
            if n is None:
                return []
            rel = str(kw.get("relation") or "").lower()
            if rel == "at least":
                return [("min_words", str(n))]
            if rel in ("less than", "at most"):
                return [("max_words", str(n))]
            return [("exact_words", str(n))]
        if iid == "length_constraints:number_sentences":
            n = kw.get("num_sentences")
            if n is None:
                return []
            rel = str(kw.get("relation") or "").lower()
            if rel == "at least":
                return [("min_sentences", str(n))]
            if rel in ("less than", "at most"):
                return [("max_sentences", str(n))]
            return [("exact_sentences", str(n))]
        if iid == "keywords:forbidden_words":
            return [("keyword_exclude", str(w)) for w in (kw.get("forbidden_words") or []) if str(w).strip()]
        if iid == "keywords:existence":
            return [("keyword_include", str(w)) for w in (kw.get("keywords") or []) if str(w).strip()]
        if iid == "keywords:frequency":
            if kw.get("keyword") is None or kw.get("frequency") is None:
                return []
            return [("keyword_count", (str(kw.get("keyword")), str(kw.get("frequency"))))]
        if iid == "keywords:letter_frequency":
            if kw.get("letter") is None or kw.get("let_frequency") is None:
                return []
            return [("letter_frequency", (str(kw.get("letter")), str(kw.get("let_relation") or "at least"), str(kw.get("let_frequency"))))]
        if iid == "combination:repeat_prompt":
            return [("repeat_prompt", None)]
        if iid == "startend:quotation":
            return [("quotation_wrap", None)]
        if iid == "change_case:english_lowercase":
            return [("all_lowercase", None)]
        if iid == "change_case:english_capital":
            return [("all_uppercase", None)]
        if iid == "change_case:capital_word_frequency":
            n = kw.get("capital_frequency")
            return [("capital_word_frequency", (str(kw.get("capital_relation") or "at least"), str(n)))] if n is not None else []
        if iid == "detectable_format:title":
            return [("title_double_brackets", None)]
        if iid == "detectable_format:number_bullet_lists":
            n = kw.get("num_bullets")
            return [("exact_bullet_points", str(n))] if n is not None else []
        if iid == "language:response_language":
            lang = kw.get("language")
            return [("response_language", str(lang))] if lang else []
        if iid == "detectable_content:number_placeholders":
            n = kw.get("num_placeholders")
            return [("placeholder_count", str(n))] if n is not None else []
        if iid == "length_constraints:number_paragraphs":
            n = kw.get("num_paragraphs")
            return [("exact_paragraphs", str(n))] if n is not None else []
        if iid == "startend:end_checker":
            phrase = kw.get("end_phrase")
            return [("end_with_phrase", str(phrase))] if phrase else []
        if iid == "detectable_content:postscript":
            return [("postscript", None)]
        if iid == "combination:two_responses":
            return [("multiple_responses", "2")]
        if iid == "detectable_format:json_format":
            return [("json_format", None)]
        if iid == "detectable_format:multiple_sections":
            n = kw.get("num_sections")
            if n is None:
                return []
            splitter = str(kw.get("section_spliter") or "").upper()
            if splitter == "PARAGRAPH":
                return [("exact_paragraphs", str(n))]
            return [("section_markers", str(n))]
        if iid == "length_constraints:nth_paragraph_first_word":
            n = kw.get("nth_paragraph")
            first = kw.get("first_word")
            if n is None or not first:
                return []
            out = [("nth_paragraph_first_word", (str(n), str(first)))]
            if kw.get("num_paragraphs") is not None:
                out.append(("exact_paragraphs", str(kw.get("num_paragraphs"))))
            return out
        return []

    @classmethod
    def _extract_ifeval_constraints_from_sample(cls, sample: Dict) -> List[tuple]:
        if (sample.get("source") or "") != "ifeval_prompt_only":
            return []
        prompt = (sample.get("instruction") or "").strip()
        if not prompt:
            return []
        rec = _load_ifeval_prompt_index().get(prompt)
        if not rec:
            return []
        constraints: List[tuple] = []
        ids = rec.get("instruction_id_list") or []
        kwargs_list = rec.get("kwargs") or []
        for i, iid in enumerate(ids):
            kw = kwargs_list[i] if i < len(kwargs_list) and isinstance(kwargs_list[i], dict) else {}
            constraints.extend(cls._map_ifeval_instruction(iid, kw))
        return constraints

    @classmethod
    def _extract_mifeval_constraints_from_sample(cls, sample: Dict) -> List[tuple]:
        src = str(sample.get("source") or "").lower()
        # Support both strict source names (mifeval_xx) and older PMMEval-prefixed names.
        if src.startswith("pmmeval-mifeval-"):
            src = "mifeval_" + src.split("-")[-1]
        if not src.startswith("mifeval_"):
            return []
        prompt = (sample.get("instruction") or "").strip()
        if not prompt:
            return []
        rec = (_load_mifeval_prompt_index().get(src) or {}).get(prompt)
        if not rec:
            return []
        constraints: List[tuple] = []
        ids = rec.get("instruction_id_list") or []
        kwargs_list = rec.get("kwargs") or []
        for i, iid in enumerate(ids):
            kw = kwargs_list[i] if i < len(kwargs_list) and isinstance(kwargs_list[i], dict) else {}
            constraints.extend(cls._map_ifeval_instruction(iid, kw))
        return constraints

    @classmethod
    def _extract_lria_constraints_from_sample(cls, sample: Dict) -> List[tuple]:
        src = str(sample.get("source") or "")
        if not src.startswith("lria_follow_"):
            return []
        instr = (sample.get("instruction") or "")
        inp = str(sample.get("input") or "")
        full_text = (instr + "\n" + inp).strip() if inp else instr

        meta = sample.get("meta") or {}
        if not meta:
            lookup_key = (
                (sample.get("instruction") or "").strip(),
                (sample.get("input") or "").strip(),
                str(sample.get("source") or ""),
                str(sample.get("language") or ""),
            )
            meta = (_load_strict_labeled_if_index().get(lookup_key) or {}).get("meta") or {}
        l1 = str(meta.get("L1") or "")
        constraints: List[tuple] = []

        choice_opts = _parse_lria_choice_options_from_reference(sample)
        if choice_opts and (l1 == "\u9009\u62e9\u5224\u65ad" or _looks_like_choice_judge_prompt(full_text)):
            constraints.append(("reply_only_choices", tuple(choice_opts)))

        if re.search(r"yes\s*or\s*no|true\s*or\s*false|correct\s*or\s*incorrect", full_text, re.IGNORECASE):
            text_lower = full_text.lower()
            if "yes" in text_lower and "no" in text_lower:
                constraints.append(("reply_only_choices", ("yes", "no")))
            elif "true" in text_lower and "false" in text_lower:
                constraints.append(("reply_only_choices", ("true", "false")))
            else:
                constraints.append(("reply_only_choices", ("correct", "incorrect")))
        if re.search(r"\u662f\u6216\u5426|\u6b63\u786e\u6216\u9519\u8bef|\u5bf9\u6216\u9519", full_text):
            if "\u662f\u6216\u5426" in full_text:
                constraints.append(("reply_only_choices", ("\u662f", "\u5426")))
            elif "\u5bf9\u6216\u9519" in full_text:
                constraints.append(("reply_only_choices", ("\u5bf9", "\u9519")))
            else:
                constraints.append(("reply_only_choices", ("\u6b63\u786e", "\u9519\u8bef")))

        if cls.ENABLE_LRIA_FALLBACK:
            ref = (sample.get("reference") or "").strip()
            ref_short = bool(ref) and len(ref) <= 30 and len(ref.split()) <= 3 and "/" not in ref
            if ref_short and l1 == "\u8bed\u8a00":
                constraints.append(("lria_language_or_code", ref))
            exact_l1_allow = {
                "\u9009\u62e9\u5224\u65ad",
                "\u987a\u5e8f",
                "\u8f93\u51fa\u957f\u5ea6",
                "\u6b21\u6570",
                "\u683c\u5f0f",
                "\u7528\u8bcd",
                "\u91cd\u5b9a\u4e49",
            }
            if ref_short and l1 in exact_l1_allow:
                constraints.append(("lria_reference_exact_short", ref))
        return constraints

    @classmethod
    def extract_constraints_from_sample(cls, sample: Dict) -> List[tuple]:
        instr_text = (sample.get("instruction") or "")
        if sample.get("input"):
            instr_text = instr_text + "\n" + str(sample.get("input"))
        constraints = cls.extract_constraints(instr_text)
        constraints = _merge_constraints(constraints, cls._extract_ifeval_constraints_from_sample(sample))
        constraints = _merge_constraints(constraints, cls._extract_mifeval_constraints_from_sample(sample))
        constraints = _merge_constraints(constraints, cls._extract_lria_constraints_from_sample(sample))
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
    def evaluate_sample_with_constraints(cls, constraints: List[tuple], output: str) -> Dict:
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
    def evaluate_sample_from_sample(cls, sample: Dict) -> Dict:
        constraints = cls.extract_constraints_from_sample(sample)
        return cls.evaluate_sample_with_constraints(constraints, sample.get("prediction", ""))
    
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
        # BLEU tokenizer按语种选择：中文用zh，其他语种用13a
        self.bleu_zh = BLEU(effective_order=True, lowercase=False, tokenize='zh')
        self.bleu_default = BLEU(effective_order=True, lowercase=False, tokenize='13a')
    
    def _detect_lang(self, text: str) -> str:
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.replace(' ', ''))
        return "zh" if total_chars > 0 and chinese_chars / total_chars > 0.3 else "en"

    def _is_zh_text(self, text: str) -> bool:
        return self._detect_lang(text) == "zh"
    
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
        
        # BLEU（混合语料按参考文本语种分开计算，再加权平均）
        try:
            zh_preds, zh_refs = [], []
            other_preds, other_refs = [], []
            for pred, ref in zip(predictions, references):
                if self._is_zh_text(ref):
                    zh_preds.append(pred)
                    zh_refs.append(ref)
                else:
                    other_preds.append(pred)
                    other_refs.append(ref)

            bleu_parts = []
            total = len(references)
            if zh_refs:
                zh_score = self.bleu_zh.corpus_score(zh_preds, [zh_refs]).score
                bleu_parts.append((zh_score, len(zh_refs)))
            if other_refs:
                other_score = self.bleu_default.corpus_score(other_preds, [other_refs]).score
                bleu_parts.append((other_score, len(other_refs)))

            if bleu_parts:
                bleu_score = sum(score * cnt for score, cnt in bleu_parts) / total
            else:
                bleu_score = 0.0
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
        constraints = InstructionFollowingEvaluator.extract_constraints_from_sample(s)
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


def score_results(input_file: str, output_dir: str, reclassify: bool = False):
    """对推理结果进行评分"""
    
    # 加载推理结果
    print(f"加载推理结果: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    model_name = data.get("model_name", "unknown")
    samples = data.get("samples", [])
    
    print(f"模型名称: {model_name}")
    print(f"样本数量: {len(samples)}")
    
    # 可选：运行时重分类（严格测试集建议关闭，保持任务标签不变）
    if reclassify:
        samples = _reclassify_samples(samples)
    else:
        print("任务重分类: 关闭（使用原始 task_type）")
    
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
            if_result = InstructionFollowingEvaluator.evaluate_sample_from_sample(s)
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
    parser.add_argument(
        "--reclassify", action="store_true",
        help="启用运行时任务重分类（默认关闭，严格测试集建议关闭）"
    )
    
    parser.add_argument(
        "--enable-lria-fallback", action="store_true",
        help="Enable LRIA fallback judge for higher IF_labeled coverage (hybrid scoring mode)."
    )

    args = parser.parse_args()
    
    # 默认输出目录
    if args.output_dir is None:
        input_dir = os.path.dirname(args.input_file)
        out_dir_tmp = "evaluation/performance"
        input_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_dir = os.path.join(out_dir_tmp, f"{input_name}_eval")
    
    InstructionFollowingEvaluator.ENABLE_LRIA_FALLBACK = bool(args.enable_lria_fallback)
    print(
        "IF scoring mode: HYBRID (LRIA fallback enabled)"
        if InstructionFollowingEvaluator.ENABLE_LRIA_FALLBACK
        else "IF scoring mode: STRICT (LRIA fallback disabled)"
    )
    score_results(args.input_file, args.output_dir, args.reclassify)


if __name__ == "__main__":
    main()
