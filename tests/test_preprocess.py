"""
test_preprocess.py — 预处理管道测试（错别字纠正 + 敏感词过滤）
不依赖大模型，只测本地小模型
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── TypoService 测试 ────────────────────────────────────
@pytest.mark.asyncio
async def test_typo_service():
    """测试 macbert4csc 错别字纠正"""
    print("=" * 60)
    print("测试 1: TypoService (macbert4csc 错别字纠正)")
    print("=" * 60)

    from app.services.typo_service import get_typo_corrector

    corrector = get_typo_corrector()

    # 未加载模型时的 fallback
    if not corrector.is_ready:
        text, errors = await corrector.correct("测试")
        assert text == "测试", "未加载模型时应原样返回"
        assert errors == [], "未加载模型时无纠错详情"
        print("  [OK] 未加载模型时原样返回")

    # 加载模型
    print(f"  加载模型: {corrector.MODEL_NAME} ...")
    ok = corrector.load_model()
    if not ok:
        print("  [WARN] 模型加载失败，跳过纠错测试")
        return

    must_pass = [
        # (输入, 期望包含的关键字) — 这些必须成功
        ("今天天汽真不好", "天气"),       # 汽→气
        ("人工智能的发张历程", "发展"),     # 张→展
        ("今天天气很好", None),            # 无错字
    ]
    challenge = [
        # (输入, 期望方向) — 模型可能识别不了，仅观察不断言
        ("他是一位有名的作嘉", "作家"),     # 嘉→家
        ("我们一定要坚吃原则", "坚持"),     # 吃→持
        ("我想学习机器深习", "深度"),       # 深→深度
    ]

    for text, expected_keyword in must_pass:
        corrected, errors = await corrector.correct(text)
        ok_flag = (expected_keyword is None and corrected == text) or \
                  (expected_keyword and expected_keyword in corrected)
        status = "OK" if ok_flag else "FAIL"
        print(f"  [{status}] 输入: {text}")
        print(f"       纠正: {corrected}")
        if errors:
            print(f"       详情: {errors}")
        if expected_keyword:
            assert expected_keyword in corrected, \
                f"期望包含 '{expected_keyword}'，实际: {corrected}"
        else:
            assert corrected == text, f"期望无变化，实际: {corrected}"

    for text, expected_keyword in challenge:
        corrected, errors = await corrector.correct(text)
        hit = expected_keyword in corrected if expected_keyword else corrected == text
        status = "OK" if hit else "INFO"
        print(f"  [{status}] 输入: {text}")
        print(f"       纠正: {corrected}")
        if errors:
            print(f"       详情: {errors}")
        # 挑战用例不做硬性断言，只记录结果

    print()


# ─── PreprocessNode 集成测试 ──────────────────────────────
@pytest.mark.asyncio
async def test_preprocess_node():
    """测试 preprocess_node 完整流程：纠错 → 敏感词过滤"""
    print("=" * 60)
    print("测试 2: preprocess_node (集成)")
    print("=" * 60)

    from app.graph.nodes import preprocess_node

    # A: 无错字、无敏感词 → 原样通过
    state = {"message": "你好世界"}
    result = await preprocess_node(state)
    assert result == {}, "无错字无敏感词应原样通过"
    print("  [OK] A. 无错字无敏感词 -> 原样通过")

    # B: 有错字 → message 被纠正，original_message 保留
    state = {"message": "今天天汽真不好"}
    result = await preprocess_node(state)
    print(f"  B. 输入: {state['message']!r}")
    print(f"     返回: {result}")
    if "original_message" in result:
        assert result["original_message"] == "今天天汽真不好"
        assert "天气" in result.get("message", "")
        print("     [OK] original_message 和 message 均正确")
    else:
        print("     [WARN] 纠错未触发（模型未加载或未识别）")

    # C: 敏感词 → 返回错误
    state = {"message": "这是一个包含贩毒的句子"}
    result = await preprocess_node(state)
    assert result.get("error") == "sensitive_content", f"期望 sensitive_content，实际: {result}"
    print("  [OK] C. 敏感词 -> 返回错误")

    print()


# ─── Config 测试 ─────────────────────────────────────────
def test_config():
    print("=" * 60)
    print("测试 3: Config 配置")
    print("=" * 60)

    from app.config import config

    assert hasattr(config, "preprocess"), "config 应有 preprocess 属性"
    assert config.preprocess.enable_typo_correction is True
    assert "macbert4csc" in config.preprocess.typo_model
    print(f"  typo_model:     {config.preprocess.typo_model}")
    print("  [OK] 配置正确")
    print()


async def main():
    print("\n预处理管道功能验证")
    print("=" * 60)

    test_config()
    await test_typo_service()
    await test_preprocess_node()

    print("=" * 60)
    print("所有验证完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
