import uuid
import streamlit as st
from langchain_core.runnables import RunnableConfig
from rag import ingest
from graph import graph

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NovaBite Assistant",
    layout="wide"
)

# ── Initialize Vector Store ───────────────────────────────────────────────────

@st.cache_resource
def initialize():
    ingest()
    return True

initialize()

# ── Session State ─────────────────────────────────────────────────────────────

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_intent" not in st.session_state:
    st.session_state.last_intent = "—"

if "last_tool" not in st.session_state:
    st.session_state.last_tool = "—"


# ── Main Chat ─────────────────────────────────────────────────────────────────

st.markdown("## NovaBite Restaurant Assistant")
st.markdown("Ask me about our menu, make a reservation, or check your loyalty points.")
st.markdown("---")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke graph
    with st.chat_message("assistant"):
        with st.spinner("Nova is thinking..."):
            config = RunnableConfig(
                configurable={"thread_id": st.session_state.thread_id},
                recursion_limit=10,
                tags=["novabite"]
            )

            state = {
                "user_question": prompt,
                "user_intent": "",
                "rag_context": "",
                "rag_answer": "",
                "tool_output": "",
                "final_response": ""
            }

            result = graph.invoke(state, config=config)
            response = result.get("final_response", "I'm not sure how to help with that.")

            # Update sidebar info
            st.session_state.last_intent = result.get("user_intent", "—")

            # Detect last tool called from tool_output content
            tool_output = result.get("tool_output", "")
            if "Booking Confirmed" in tool_output:
                st.session_state.last_tool = "book_table"
            elif "available" in tool_output.lower() and "branch" in tool_output.lower():
                st.session_state.last_tool = "check_table_availability"
            elif "Today's special" in tool_output:
                st.session_state.last_tool = "get_today_special"
            elif "Points Balance" in tool_output:
                st.session_state.last_tool = "check_loyalty_points"
            elif result.get("rag_answer"):
                st.session_state.last_tool = "retrieve_docs"
            else:
                st.session_state.last_tool = "—"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})