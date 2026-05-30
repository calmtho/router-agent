"""LangGraph 自测脚本 - 可以直接运行验证功能"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph import app_graph


async def test_chat_agent():
    """测试 Chat 代理"""
    print("=" * 50)
    print("测试 1: Chat 代理 (闲聊)")
    print("=" * 50)

    result = await app_graph.ainvoke({
        "message": "你好，介绍一下你自己",
        "session_id": "test",
        "file_ids": [],
        "target_agent": "",
        "cot_reasoning": "",
        "answer": "",
        "agent_used": "",
        "sources": [],
    })

    print(f"回答长度: {len(result['answer'])} 字符")
    print(f"使用的代理: {result['agent_used']}")
    print(f"CoT 推理: {result['cot_reasoning']}")
    print()


async def test_mcp_agent():
    """测试 MCP 代理（工具调用）"""
    print("=" * 50)
    print("测试 2: MCP 代理 (工具调用)")
    print("=" * 50)

    result = await app_graph.ainvoke({
        "message": "计算 123 + 456",
        "session_id": "test",
        "file_ids": [],
        "target_agent": "",
        "cot_reasoning": "",
        "answer": "",
        "agent_used": "",
        "sources": [],
    })

    print(f"回答: {result['answer']}")
    print(f"使用的代理: {result['agent_used']}")
    print(f"CoT 推理: {result['cot_reasoning']}")
    print()


async def test_rag_agent():
    """测试 RAG 代理（需要 file_ids）"""
    print("=" * 50)
    print("测试 3: RAG 代理 (文档检索)")
    print("=" * 50)

    result = await app_graph.ainvoke({
        "message": "这份文档说了什么？",
        "session_id": "test",
        "file_ids": ["test-file-id"],  # 模拟有文件
        "target_agent": "",
        "cot_reasoning": "",
        "answer": "",
        "agent_used": "",
        "sources": [],
    })

    print(f"回答长度: {len(result['answer'])} 字符")
    print(f"使用的代理: {result['agent_used']}")
    print(f"CoT 推理: {result['cot_reasoning']}")
    print("注意: RAG 测试可能失败，因为没有实际文件和 Milvus 连接")
    print()


async def test_router_only():
    """单独测试路由逻辑"""
    print("=" * 50)
    print("测试 4: 路由逻辑 (CoT)")
    print("=" * 50)

    from app.graph.nodes import router_node

    # 测试需要工具的查询
    result = await router_node({
        "message": "计算 100 * 200",
        "session_id": "test",
        "file_ids": [],
    })
    print(f"查询: '计算 100 * 200'")
    print(f"路由结果: {result['target_agent']}")
    print(f"推理: {result['cot_reasoning']}")
    print()

    # 测试需要文档的查询
    result = await router_node({
        "message": "这份文档总结一下",
        "session_id": "test",
        "file_ids": ["file1"],
    })
    print(f"查询: '这份文档总结一下' (有 file_ids)")
    print(f"路由结果: {result['target_agent']}")
    print(f"推理: {result['cot_reasoning']}")
    print()


async def test_graph_structure():
    """测试图结构"""
    print("=" * 50)
    print("测试 5: 图结构")
    print("=" * 50)

    graph = app_graph.get_graph()
    print(f"图节点: {list(graph.nodes.keys())}")

    # 不打印 ASCII 图（需要 grandalf 依赖）
    # print("\n图结构:")
    # graph.print_ascii()
    print("(图结构 visualization 需要安装 grandalf，跳过显示)")
    print()


async def main():
    """运行所有测试"""
    print("\nLangGraph 自测脚本")
    print("开始测试...\n")

    try:
        await test_graph_structure()
        await test_router_only()
        await test_chat_agent()
        await test_mcp_agent()
        await test_rag_agent()

        print("=" * 50)
        print("所有测试完成!")
        print("=" * 50)

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
