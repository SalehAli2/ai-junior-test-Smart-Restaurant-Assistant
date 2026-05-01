from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, AIMessage
from state import NovaState
from agents import orchestrator_node, rag_node, operations_node


def assembler_node(state: NovaState) -> dict:
    if state.get("rag_answer"):
        response = state["rag_answer"]
    elif state.get("tool_output"):
        response = state["tool_output"]
    else:
        response = state.get("final_response", "I'm not sure how to help with that.")
    new_messages = [
        HumanMessage(content=state["user_question"]),
        AIMessage(content=response)
    ]
    return {
        "final_response": response,
        "chat_history": new_messages,
        "rag_answer": "",
        "rag_context": "",
        "tool_output": "",
        "user_intent": ""
    }



def fallback_node(state: NovaState) -> dict:
    response = (
        "I'm Nova, NovaBite's assistant. I can help you with:\n"
        "- Table reservations\n"
        "- Menu and allergen information\n"
        "- Loyalty points and today's specials\n"
        "How can I help you?"
    )
    return {
        "final_response": response,
        "chat_history": [
            HumanMessage(content=state["user_question"]),
            AIMessage(content=response)
        ]
    }


def build_graph():
    builder = StateGraph(NovaState)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("rag", rag_node)
    builder.add_node("operations", operations_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("assembler", assembler_node)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        lambda state: state.get("user_intent", "fallback"),
        {
            "rag": "rag",
            "operations": "operations",
            "fallback": "fallback"
        }
    )
    builder.add_edge("rag", "assembler")
    builder.add_conditional_edges(
        "operations",
        lambda state: "rag" if state.get("user_intent") == "rag" else "assembler",
        {
            "rag": "rag",
            "assembler": "assembler"
        }
    )
    builder.add_edge("fallback", "assembler")
    builder.add_edge("assembler", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)

graph = build_graph()