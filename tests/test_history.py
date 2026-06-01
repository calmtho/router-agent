"""测试对话历史剪枝逻辑"""
import asyncio
import pytest

from app.services.history_service import (
    append_history,
    clear_history,
    generate_summary,
    generate_title,
    get_history,
    get_summary,
    get_title,
    needs_summary,
    _session_history,
    _session_summaries,
    _session_titles,
    _session_turn_counter,
)


@pytest.fixture(autouse=True)
def clean_state():
    """每个测试前后清理内存状态"""
    _session_history.clear()
    _session_summaries.clear()
    _session_titles.clear()
    _session_turn_counter.clear()
    yield
    _session_history.clear()
    _session_summaries.clear()
    _session_titles.clear()
    _session_turn_counter.clear()


SESSION_ID = "test-session"


class TestNeedsSummary:
    """测试 needs_summary 阈值判断"""

    def test_no_history(self):
        """没有历史时不需要摘要"""
        assert needs_summary(SESSION_ID, threshold=10) is False

    def test_below_threshold(self):
        """低于阈值不需要摘要"""
        for i in range(10):  # 10 轮
            append_history(SESSION_ID, f"msg{i}", f"reply{i}")
        assert needs_summary(SESSION_ID, threshold=10) is False

    def test_at_threshold(self):
        """刚好等于阈值不需要摘要（> threshold，不是 >=）"""
        for i in range(10):
            append_history(SESSION_ID, f"msg{i}", f"reply{i}")
        assert needs_summary(SESSION_ID, threshold=10) is False

    def test_above_threshold(self):
        """超过阈值需要摘要"""
        for i in range(11):  # 11 轮
            append_history(SESSION_ID, f"msg{i}", f"reply{i}")
        assert needs_summary(SESSION_ID, threshold=10) is True


class TestGetHistory:
    """测试滑动窗口"""

    def test_empty(self):
        assert get_history(SESSION_ID) == []

    def test_within_window(self):
        for i in range(5):
            append_history(SESSION_ID, f"msg{i}", f"reply{i}")
        history = get_history(SESSION_ID, window=10)
        assert len(history) == 10  # 5 轮 × 2 条

    def test_exceeds_window(self):
        for i in range(15):
            append_history(SESSION_ID, f"msg{i}", f"reply{i}")
        history = get_history(SESSION_ID, window=10)
        assert len(history) == 20  # 只返回最近 10 轮
        # 确认是最新的 10 轮
        assert history[0]["content"] == "msg5"
        assert history[-1]["content"] == "reply14"


class TestGenerateSummary:
    """测试摘要生成和裁剪"""

    @pytest.mark.asyncio
    async def test_no_summary_needed_below_threshold(self):
        """低于阈值时不生成摘要"""
        for i in range(5):
            append_history(SESSION_ID, f"msg{i}", f"reply{i}")

        result = await generate_summary(SESSION_ID, keep_rounds=10)
        assert result == ""  # 不需要摘要，返回空
        assert get_summary(SESSION_ID) == ""

    @pytest.mark.asyncio
    async def test_summary_generated_above_threshold(self):
        """超过阈值时生成摘要（mock LLM）"""
        from unittest.mock import AsyncMock, patch

        for i in range(12):
            append_history(SESSION_ID, f"msg{i}", f"reply{i}")

        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            mock_client.chat = AsyncMock(return_value="这是关于测试的对话")
            mock_llm.return_value = mock_client

            result = await generate_summary(SESSION_ID, keep_rounds=10)

        assert result == "这是关于测试的对话"
        assert get_summary(SESSION_ID) == "这是关于测试的对话"

    @pytest.mark.asyncio
    async def test_history_trimmed_after_summary(self):
        """摘要生成后历史被裁剪到 keep_rounds 轮"""
        from unittest.mock import AsyncMock, patch

        for i in range(12):
            append_history(SESSION_ID, f"msg{i}", f"reply{i}")

        # 验证裁剪前有 24 条消息
        assert len(_session_history[SESSION_ID]) == 24

        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            mock_client.chat = AsyncMock(return_value="这是关于测试的对话")
            mock_llm.return_value = mock_client

            await generate_summary(SESSION_ID, keep_rounds=10)

        # 裁剪后只剩 20 条消息（10 轮）
        assert len(_session_history[SESSION_ID]) == 20
        # 确认保留的是最新的 10 轮
        assert _session_history[SESSION_ID][0]["content"] == "msg2"

    @pytest.mark.asyncio
    async def test_summary_can_be_updated(self):
        """摘要可以被覆盖更新，而不是只生成一次"""
        from unittest.mock import AsyncMock, patch

        for i in range(12):
            append_history(SESSION_ID, f"msg{i}", f"reply{i}")

        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            mock_client.chat = AsyncMock(return_value="第一次摘要")
            mock_llm.return_value = mock_client

            result1 = await generate_summary(SESSION_ID, keep_rounds=10)

        assert result1 == "第一次摘要"
        assert get_summary(SESSION_ID) == "第一次摘要"

        # 再加 2 轮，达到 12 轮（超过 10 轮阈值）
        append_history(SESSION_ID, "new_msg1", "new_reply1")
        append_history(SESSION_ID, "new_msg2", "new_reply2")

        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            mock_client.chat = AsyncMock(return_value="更新后的摘要")
            mock_llm.return_value = mock_client

            result2 = await generate_summary(SESSION_ID, keep_rounds=10)

        assert result2 == "更新后的摘要"
        assert get_summary(SESSION_ID) == "更新后的摘要"

    @pytest.mark.asyncio
    async def test_summary_includes_old_summary(self):
        """生成新摘要时会包含旧摘要作为上下文"""
        from unittest.mock import AsyncMock, patch

        for i in range(12):
            append_history(SESSION_ID, f"msg{i}", f"reply{i}")

        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            mock_client.chat = AsyncMock(return_value="第一次摘要")
            mock_llm.return_value = mock_client

            await generate_summary(SESSION_ID, keep_rounds=10)

        # 再加对话，触发第二次摘要
        append_history(SESSION_ID, "new_msg1", "new_reply1")
        append_history(SESSION_ID, "new_msg2", "new_reply2")

        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            # 捕获传入的 prompt，验证包含旧摘要
            call_args = {}
            original_chat = mock_client.chat

            async def capture_chat(messages):
                call_args["messages"] = messages
                return "更新后的摘要"

            mock_client.chat = capture_chat
            mock_llm.return_value = mock_client

            await generate_summary(SESSION_ID, keep_rounds=10)

        # 验证 prompt 中包含旧摘要
        prompt = call_args["messages"][1]["content"]
        assert "已有摘要" in prompt
        assert "第一次摘要" in prompt


class TestFullFlow:
    """测试完整的剪枝流程"""

    @pytest.mark.asyncio
    async def test_full_pruning_flow(self):
        """模拟完整流程：对话达到阈值 → 触发摘要 → 历史裁剪"""
        from unittest.mock import AsyncMock, patch

        # 1. 10 轮对话，不触发摘要
        for i in range(10):
            append_history(SESSION_ID, f"msg{i}", f"reply{i}")
        assert needs_summary(SESSION_ID, threshold=10) is False

        # 2. 第 11 轮，触发摘要
        append_history(SESSION_ID, "msg10", "reply10")
        assert needs_summary(SESSION_ID, threshold=10) is True

        # 3. 生成摘要并裁剪
        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            mock_client.chat = AsyncMock(return_value="这是关于前11轮的摘要")
            mock_llm.return_value = mock_client

            summary = await generate_summary(SESSION_ID, keep_rounds=10)

        # 4. 验证：摘要已生成，历史已裁剪
        assert summary == "这是关于前11轮的摘要"
        assert get_summary(SESSION_ID) == "这是关于前11轮的摘要"
        assert len(_session_history[SESSION_ID]) == 20  # 10 轮

        # 5. 验证：get_history 返回的窗口内容正确
        history = get_history(SESSION_ID, window=10)
        assert len(history) == 20

        # 6. 继续对话，加 1 轮后有 11 轮，仍然超过阈值会再次触发
        append_history(SESSION_ID, "msg11", "reply11")
        assert needs_summary(SESSION_ID, threshold=10) is True

        # 7. 再次生成摘要（应该覆盖旧摘要）
        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            mock_client.chat = AsyncMock(return_value="这是更新后的摘要")
            mock_llm.return_value = mock_client

            summary2 = await generate_summary(SESSION_ID, keep_rounds=10)

        assert summary2 == "这是更新后的摘要"
        assert get_summary(SESSION_ID) == "这是更新后的摘要"
        # 裁剪后又是 10 轮（20 条消息）
        assert len(_session_history[SESSION_ID]) == 20


class TestGenerateTitle:
    """测试标题生成"""

    @pytest.mark.asyncio
    async def test_title_generated_from_first_rounds(self):
        """根据前几轮对话生成标题"""
        from unittest.mock import AsyncMock, patch

        append_history(SESSION_ID, "什么是机器学习？", "机器学习是人工智能的一个分支...")

        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            mock_client.chat = AsyncMock(return_value="机器学习入门讨论")
            mock_llm.return_value = mock_client

            result = await generate_title(SESSION_ID)

        assert result == "机器学习入门讨论"
        assert get_title(SESSION_ID) == "机器学习入门讨论"

    @pytest.mark.asyncio
    async def test_title_not_regenerated_if_exists(self):
        """已有标题时不重复生成"""
        from unittest.mock import AsyncMock, patch

        # 先设置一个标题
        _session_titles[SESSION_ID] = "已有标题"
        append_history(SESSION_ID, "新的消息", "新的回复")

        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            mock_client.chat = AsyncMock(return_value="不应出现的标题")
            mock_llm.return_value = mock_client

            result = await generate_title(SESSION_ID)

        # 应返回已有标题，不调用 LLM
        assert result == "已有标题"
        mock_client.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_title_empty_when_no_history(self):
        """没有历史时返回空字符串"""
        from unittest.mock import AsyncMock, patch

        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            mock_llm.return_value = mock_client

            result = await generate_title(SESSION_ID)

        assert result == ""
        mock_client.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_title_cleans_quotes_and_prefix(self):
        """标题会清理引号和前缀"""
        from unittest.mock import AsyncMock, patch

        append_history(SESSION_ID, "Python怎么学？", "建议从基础语法开始...")

        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            # 模拟 LLM 返回带前缀和引号的标题
            mock_client.chat = AsyncMock(return_value='标题："Python学习指南"')
            mock_llm.return_value = mock_client

            result = await generate_title(SESSION_ID)

        assert result == "Python学习指南"

    @pytest.mark.asyncio
    async def test_title_strips_reasoning_leak(self):
        """清理 LLM 推理泄漏：如'用户需要我生成一个简短的会话标题：关于AI学习'"""
        from unittest.mock import AsyncMock, patch

        append_history(SESSION_ID, "我想学人工智能", "人工智能是一个广阔的领域...")

        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            # 模拟 LLM 泄漏推理过程
            mock_client.chat = AsyncMock(return_value="用户需要我生成一个简短的会话标题：关于AI学习")
            mock_llm.return_value = mock_client

            result = await generate_title(SESSION_ID)

        assert result == "关于AI学习"

    @pytest.mark.asyncio
    async def test_title_strips_reasoning_with_english_colon(self):
        """清理英文冒号分隔的推理泄漏"""
        from unittest.mock import AsyncMock, patch

        append_history(SESSION_ID, "什么是量子计算？", "量子计算利用量子力学...")

        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            mock_client.chat = AsyncMock(return_value="This is a conversation about: 量子计算入门")
            mock_llm.return_value = mock_client

            result = await generate_title(SESSION_ID)

        assert result == "量子计算入门"

    @pytest.mark.asyncio
    async def test_title_truncated_if_too_long(self):
        """标题过长时会被截断"""
        from unittest.mock import AsyncMock, patch

        append_history(SESSION_ID, "你好", "你好呀")

        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            mock_client.chat = AsyncMock(return_value="这是一个非常非常非常非常非常非常非常非常非常非常长的标题")
            mock_llm.return_value = mock_client

            result = await generate_title(SESSION_ID)

        assert len(result) <= 20

    @pytest.mark.asyncio
    async def test_title_cleared_with_clear_history(self):
        """clear_history 同时清理标题"""
        _session_titles[SESSION_ID] = "测试标题"
        _session_summaries[SESSION_ID] = "测试摘要"
        _session_history[SESSION_ID] = [{"role": "user", "content": "hi"}]

        clear_history(SESSION_ID)

        assert get_title(SESSION_ID) == ""
        assert get_summary(SESSION_ID) == ""
        assert get_history(SESSION_ID) == []


class TestTitleAndSummaryCoexist:
    """测试标题与摘要共存"""

    @pytest.mark.asyncio
    async def test_title_and_summary_independent(self):
        """标题和摘要互不影响"""
        from unittest.mock import AsyncMock, patch

        # 1. 先生成标题
        append_history(SESSION_ID, "什么是深度学习？", "深度学习是机器学习的子集...")

        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            mock_client.chat = AsyncMock(return_value="深度学习讨论")
            mock_llm.return_value = mock_client

            title = await generate_title(SESSION_ID)

        assert title == "深度学习讨论"
        assert get_title(SESSION_ID) == "深度学习讨论"
        assert get_summary(SESSION_ID) == ""  # 还没有摘要

        # 2. 继续对话到超过阈值，生成摘要
        for i in range(11):
            append_history(SESSION_ID, f"msg{i}", f"reply{i}")

        with patch("app.services.history_service.get_llm_client") as mock_llm:
            mock_client = AsyncMock()
            mock_client.chat = AsyncMock(return_value="关于深度学习的详细讨论")
            mock_llm.return_value = mock_client

            summary = await generate_summary(SESSION_ID, keep_rounds=10)

        # 标题和摘要同时存在，互不覆盖
        assert get_title(SESSION_ID) == "深度学习讨论"
        assert get_summary(SESSION_ID) == "关于深度学习的详细讨论"
