#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试约束检测器 - 验证新增的约束类型能被正确识别
"""

import sys
sys.path.insert(0, '.')

from scripts.evaluate import InstructionFollowingEvaluator, _check_language, _count_sentences


def test_constraint_detection():
    """测试约束提取功能"""
    
    test_cases = [
        # 语言约束
        {
            "instruction": "Your ENTIRE response should be in zh language, no other language is allowed.",
            "expected_types": ["response_language"],
            "description": "语言约束 - 中文"
        },
        {
            "instruction": "Your response should be in fr language. Write about travel.",
            "expected_types": ["response_language"],
            "description": "语言约束 - 法文"
        },
        # 句子数约束
        {
            "instruction": "Your response should contain at least 10 sentences. Write about AI.",
            "expected_types": ["min_sentences"],
            "description": "句子数约束 - 至少10句"
        },
        # 标题格式
        {
            "instruction": "Your answer must contain a title, wrapped in double angular brackets, such as <<诗的喜悦>>.",
            "expected_types": ["title_double_brackets"],
            "description": "标题格式 - 双尖括号"
        },
        # 段落分隔符
        {
            "instruction": "Your response must have 2 paragraphs. Paragraphs are separated with the markdown divider: ***",
            "expected_types": ["exact_paragraphs", "paragraph_divider"],
            "description": "段落分隔符 - ***"
        },
        # 词频约束
        {
            "instruction": "In your response, the word 世界 should appear at least 3 times.",
            "expected_types": ["word_frequency"],
            "description": "词频约束 - 世界出现3次"
        },
        # 复合约束（test_v4_enhanced.json 第一条样本）
        {
            "instruction": "Your ENTIRE response should be in zh language, no other language is allowed. Your response should contain at least 10 sentences. Your answer must contain a title, wrapped in double angular brackets, such as <<诗的喜悦>>. The response must have 2 paragraphs. Paragraphs are separated with the markdown divider: ***. In your response, the word 世界 should appear at least 3 times.",
            "expected_types": ["response_language", "min_sentences", "title_double_brackets", "exact_paragraphs", "paragraph_divider", "word_frequency"],
            "description": "复合约束 - 6种约束"
        },
    ]
    
    print("=" * 70)
    print("约束检测器测试")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for i, case in enumerate(test_cases, 1):
        instruction = case["instruction"]
        expected = set(case["expected_types"])
        description = case["description"]
        
        # 提取约束
        constraints = InstructionFollowingEvaluator.extract_constraints(instruction)
        detected = set(c[0] for c in constraints)
        
        # 检查是否检测到了预期的约束
        missing = expected - detected
        extra = detected - expected  # 额外检测到的（可能是正常的）
        
        if missing:
            print(f"\n[{i}] FAIL: {description}")
            print(f"    指令: {instruction[:80]}...")
            print(f"    期望: {expected}")
            print(f"    检测: {detected}")
            print(f"    缺失: {missing}")
            failed += 1
        else:
            print(f"\n[{i}] PASS: {description}")
            print(f"    检测到 {len(constraints)} 个约束: {[c[0] for c in constraints]}")
            passed += 1
    
    print("\n" + "=" * 70)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 70)
    
    return failed == 0


def test_language_check():
    """测试语言检测功能"""
    print("\n" + "=" * 70)
    print("语言检测测试")
    print("=" * 70)
    
    test_cases = [
        ("这是一段中文文本，用于测试语言检测功能。", "zh", True),
        ("This is an English text for testing language detection.", "en", True),
        ("Ceci est un texte français pour tester la détection de langue.", "fr", True),
        ("这是一段中文文本。", "en", False),  # 中文文本标注为英文应该失败
        ("This is English.", "zh", False),  # 英文文本标注为中文应该失败
    ]
    
    passed = 0
    for text, lang, expected in test_cases:
        result = _check_language(text, lang)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] 文本='{text[:30]}...', 期望语言={lang}, 检测结果={result}, 期望={expected}")
        if result == expected:
            passed += 1
    
    print(f"\n语言检测: {passed}/{len(test_cases)} 通过")


def test_sentence_count():
    """测试句子数计算功能"""
    print("\n" + "=" * 70)
    print("句子数计算测试")
    print("=" * 70)
    
    test_cases = [
        ("This is one sentence. This is another. And a third!", 3),
        ("这是第一句。这是第二句。这是第三句！", 3),
        ("Hello world", 1),
        ("First sentence. Second sentence? Third sentence!", 3),
    ]
    
    passed = 0
    for text, expected in test_cases:
        result = _count_sentences(text)
        # 允许一定误差
        status = "PASS" if abs(result - expected) <= 1 else "FAIL"
        print(f"  [{status}] 文本='{text[:40]}...', 句子数={result}, 期望约={expected}")
        if abs(result - expected) <= 1:
            passed += 1
    
    print(f"\n句子数计算: {passed}/{len(test_cases)} 通过")


def test_constraint_checking():
    """测试约束检查功能"""
    print("\n" + "=" * 70)
    print("约束检查测试")
    print("=" * 70)
    
    test_cases = [
        # (约束类型, 参数, 测试文本, 期望结果)
        ("response_language", "zh", "这是一段中文回复，内容全部使用中文书写。", True),
        ("response_language", "en", "This is an English response written entirely in English.", True),
        ("response_language", "zh", "This is English text.", False),
        ("title_double_brackets", None, "<<我的标题>>\n\n这是正文内容。", True),
        ("title_double_brackets", None, "这是没有标题的文本。", False),
        ("paragraph_divider", None, "第一段内容。\n\n***\n\n第二段内容。", True),
        ("paragraph_divider", None, "第一段内容。第二段内容。", False),
        ("word_frequency", ("世界", "3"), "世界很美好。这个世界充满希望。我们的世界需要和平。", True),
        ("word_frequency", ("世界", "3"), "地球很美好。", False),
    ]
    
    passed = 0
    for ctype, param, text, expected in test_cases:
        result = InstructionFollowingEvaluator.check_constraint(text, ctype, param)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] 约束={ctype}, 参数={param}, 期望={expected}, 结果={result}")
        if result == expected:
            passed += 1
    
    print(f"\n约束检查: {passed}/{len(test_cases)} 通过")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("指令遵循评估器 - 约束检测测试")
    print("=" * 70)
    
    all_passed = True
    
    all_passed &= test_constraint_detection()
    test_language_check()
    test_sentence_count()
    test_constraint_checking()
    
    print("\n" + "=" * 70)
    if all_passed:
        print("所有关键测试通过！约束检测器已就绪。")
    else:
        print("存在失败的测试，请检查约束检测器实现。")
    print("=" * 70)
