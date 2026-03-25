"""
app.py — Streamlit chat UI for the Customer Support Q&A Chatbot.

Run with:
    streamlit run app.py
"""

import streamlit as st
from chain import CustomerSupportChain

# --- Page config ---
st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="💬",
    layout="centered",
)

# --- Custom CSS for a cleaner look ---
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    .source-tag {
        display: inline-block;
        background-color: #f0f2f6;
        color: #555;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 4px;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_chain():
    """Load the RAG chain (cached so it only loads once)."""
    return CustomerSupportChain()

def format_sources(sources: list[dict]) -> str:
    """Format retrieved source intents as small tags."""
    unique_intents = sorted(
        set(s["intent"].replace("_", " ").title() for s in sources)
    )
    tags = "".join(
        f'<span class="source-tag">{intent}</span>' for intent in unique_intents
    )
    return f"<div style='margin-top: 8px;'>{tags}</div>"

def main():
    # --- Header ---
    st.title("💬 Customer Support Chatbot")
    st.caption(
        "Ask me anything about orders, payments, refunds, accounts, and more. "
        "Powered by RAG with FAISS + Claude."
    )

    # --- Initialize session state ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            st.session_state.chain = load_chain()

    chain = st.session_state.chain

    # --- Render chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # --- Chat input ---
    if prompt := st.chat_input("How can I help you today?"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                result = chain.ask(prompt)

            response_text = result["response"]
            source_html = format_sources(result["sources"])
            full_content = response_text + "\n\n" + source_html

            st.markdown(full_content, unsafe_allow_html=True)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_content}
        )


if __name__ == "__main__":
    main()