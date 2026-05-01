from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages



class NovaState(TypedDict):
    user_question: str
    user_intent: str
    chat_history: Annotated[list[BaseMessage], add_messages]
    rag_context: str
    rag_answer: str
    tool_output: str
    final_response: str


