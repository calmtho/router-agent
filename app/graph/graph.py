"""LangGraph 图定义 - 构建主子代理图"""

from langgraph.graph import END, StateGraph

from .nodes import chat_node, mcp_node, preprocess_node, rag_node, router_node
from .state import AgentState


def router_decision(state: AgentState) -> str:
    """
    条件边函数 - 根据 target_agent 决定下一个节点

    Args:
        state: 当前图状态

    Returns:
        下一个节点名称（chat, rag, mcp）
    """
    return state.get("target_agent", "chat")


def preprocess_decision(state: AgentState) -> str:
    """
    预处理决策函数 - 判断是否通过预处理

    Args:
        state: 当前图状态

    Returns:
        "router" - 通过预处理，继续路由
        "blocked" - 检测到敏感内容，拒绝
    """
    if state.get("error") == "sensitive_content":
        return "blocked"
    return "router"


def build_graph() -> StateGraph:
    """
    构建主子代理图

    图结构：
        __start__ -> preprocess -> router -> chat/rag/mcp -> END
                                -> blocked (拒绝) -> END

    Returns:
        编译后的图实例
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("router", router_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("mcp", mcp_node)
    workflow.add_node("blocked", lambda _state: {"answer": "抱歉，您的请求包含不合适的内容，我无法处理。"})

    # 设置入口节点
    workflow.add_edge("__start__", "preprocess")

    # 预处理决策
    workflow.add_conditional_edges(
        "preprocess",
        preprocess_decision,
        {
            "router": "router",
            "blocked": "blocked",
        },
    )

    # 添加条件边 - 根据 router 的输出决定路由到哪个代理
    workflow.add_conditional_edges(
        "router",
        router_decision,
        {
            "chat": "chat",
            "rag": "rag",
            "mcp": "mcp",
        },
    )

    # 所有代理节点都结束
    for agent in ["chat", "rag", "mcp"]:
        workflow.add_edge(agent, END)

    workflow.add_edge("blocked", END)

    # 编译图
    return workflow.compile()


# 全局图实例
app_graph = build_graph()


def get_graph() -> StateGraph:
    """获取全局图实例"""
    return app_graph
