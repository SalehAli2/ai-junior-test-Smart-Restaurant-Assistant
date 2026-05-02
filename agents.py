from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings, ChatOllama
from tools import check_table_availability, book_table, get_today_special, check_loyalty_points
from rag import retrieve_docs
from langchain_openai import ChatOpenAI
from state import NovaState
from datetime import date
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()



llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com",
    temperature=0,
)

embedder = OllamaEmbeddings(model="nomic-embed-text")

class IntentClassification(BaseModel):
    intent: Literal["rag", "operations", "fallback"]
    confidence: float = Field(ge=0.0, le=1.0)
    needs_clarification: bool = False
    clarification_question: str = ""

orchestrator_prompt = """
You are the NovaBite Orchestrator. Your job is to analyze the user's question 
and route it to the correct sub-agent.

Available routes:
- "rag": Questions about menu items, ingredients, allergens, dietary preferences, 
         loyalty program rules, opening hours, policies, or any knowledge-based question.
- "operations": Questions about table reservations, booking, availability, 
                today's specials, or checking loyalty points balance.
- "fallback": Greetings, unrelated topics, weather, politics, or anything else.

Analyze the current question in the context of the chat history.
Return structured output only.
"""

def orchestrator_node(state: NovaState, config: RunnableConfig):
    chat_history = state.get("chat_history", [])
    
    messages = [
        SystemMessage(content=orchestrator_prompt),
        *chat_history,
        HumanMessage(content=state["user_question"])
    ]
    
    structured_llm = llm.with_structured_output(IntentClassification)
    
    result: IntentClassification = structured_llm.invoke(messages)
    
    final_intent = result.intent if result.confidence >= 0.5 else "fallback"
    
    return {
        "user_intent": final_intent
    }








rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are the NovaBite Knowledge Specialist. Answer using ONLY the provided context.\n\n"
     "RULES:\n"
     "1. Use ONLY the context. Never contradict yourself.\n"
     "2. List allergens exactly as stated — do not negate then confirm.\n"
     "3. When listing menu items, list ALL items in the context. Never truncate.\n"
     "4. If context does not contain the answer, say: "
     "'I'm sorry, I only have information regarding our menu and loyalty program.'\n\n"
     "CONTEXT:\n{context}"
     "Sources:\n{sources}"),
    ("placeholder", "{chat_history}"),
    ("human", "{question}")
])

rag_chain = (rag_prompt | llm | StrOutputParser())

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query rewriter for a restaurant assistant. "
     "Given a conversation history and a follow-up question, rewrite it into a standalone query.\n\n"
     "RULES:\n"
     "1. Expand any shorthand or nickname to its full name as mentioned in the conversation history. "
     "For example: 'the Carbonara' → 'Spaghetti Carbonara', 'the first one' → the actual dish name, "
     "'the top tier' → the actual loyalty tier name.\n"
     "2. Resolve all pronouns (it, they, those, that) using the conversation history.\n"
     "3. If the pronoun refers to a LIST of items, expand the query to mention each item by name explicitly. "
     "For example: 'What allergens does it have?' after a list of 6 pasta dishes → "
     "'What allergens are in Spaghetti Carbonara, Spicy Tomato Penne, Clam Linguine, "
     "Porcini Mushroom Pappardelle, Bolognese Tagliatelle, and Eggplant and Ricotta Rigatoni?'\n"
     "4. If 'the first one', 'the second one', etc. refers to a numbered list in the conversation, "
     "resolve it to the actual item at that position.\n"
     "5. Return ONLY the rewritten query, nothing else."),
    ("placeholder", "{chat_history}"),
    ("human", "{question}")
])

rewrite_chain = (rewrite_prompt | llm | StrOutputParser())


def rag_node(state: NovaState, config: RunnableConfig):
    question = state["user_question"]
    chat_history = state.get("chat_history", [])

    standalone_question = (
        rewrite_chain.invoke({"question": question, "chat_history": chat_history}, config)
        if chat_history else question
    )

    context, sources = retrieve_docs(standalone_question)

    if context == "No relevant information found.":
        answer = "I'm sorry, I only have information regarding our menu and loyalty program."
        return {
            "rag_context": context,
            "rag_answer": answer
        }

    answer = rag_chain.invoke(
        {"context": context, "question": question, "sources": str(sources), "chat_history": chat_history},
        config
    )

    return {
        "rag_context": context,
        "rag_answer": answer
    }




operations_prompt = """
    You are NovaBite's Operations Assistant.
    You handle table reservations, availability checks, today's specials, and loyalty points.

    Rules:
    - If the user asks about menu, ingredients, allergens, dietary restrictions, or loyalty program rules, reply with exactly: REDIRECT_TO_RAG
    - Otherwise, use the available tools to help the user.
    - Ask for missing information (name, date, time, branch, etc.) before calling tools.
    - Today's date is {today}.
    """.format(today=date.today().strftime("%Y-%m-%d"))

operations_agent = create_agent(
    model=llm,
    tools=[check_table_availability, book_table, get_today_special, check_loyalty_points],
    system_prompt=operations_prompt
)

def operations_node(state: NovaState, config: RunnableConfig):
    question = state["user_question"]
    chat_history = state.get("chat_history", [])
    
    messages = [*chat_history, HumanMessage(content=question)]
    result = operations_agent.invoke({"messages": messages}, config)
    answer = result["messages"][-1].content

    if "REDIRECT_TO_RAG" in answer.strip():
        return {"user_intent": "rag", "tool_output": ""}

    return {"tool_output": answer}
