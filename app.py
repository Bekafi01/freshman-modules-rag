import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.rag import rephrase_chain, answer_chain, format_docs, vectorstore

# Page configuration
st.set_page_config(
    page_title= "Freshman RAG Tutor",
    page_icon = "🎓",
    layout = "centered"
)

st.title("🎓 Freshman Modules RAG Tutor")
st.caption("Grounded academic assistance for Ethiopian university fresham students")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")

    # Module filter
    selected_module = st.selectbox(
        "Focus on specific module:",
        options = ["All Modules", "History", "Physics", "Logic"],
        index = 0
    )

    # Database stats
    chunk_count = vectorstore._collection.count()
    st.metric("Total Chunks in Database", chunk_count)

    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width = True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Dynamic retriever configuration based on sidebar selection
search_kwargs = {"k": 3}
if selected_module != "All Modules":
    # Match both lowecase and capitalized module names in Chroma
    search_kwargs["filter"] = {"module": {"$in": [selected_module, selected_module.lower()]}}

retriever = vectorstore.as_retriever(search_kwargs = search_kwargs)

# Initalize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg and msg["metadata"]:
            meta = msg["metadata"]
            st.caption(f"📖 **{meta['module']}** | Pages: {meta['pages']} | Confidence: `{meta['confidence']}`")
            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 View Retrieved Sources"):
                    st.text(msg["sources"])

# User Input & RAG execution
user_query = st.chat_input("Ask a question from History, Physics, or Logic .....")

if user_query:
    # Display user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching texbooks and reasoning..."):
            try:
                # Repherase if we have conversation history
                if st.session_state.chat_history:
                    rephrased = rephrase_chain.invoke({
                        "chat_history": st.session_state.chat_history,
                        "question": user_query
                    })
                    search_query = str(rephrased).strip()
                else: 
                    search_query = user_query
                
                # Retrieve relevant textbook chunks
                docs = retriever.invoke(search_query)
                context = format_docs(docs)

                # Generate cited answer
                response = answer_chain.invoke({
                    "context" : context,
                    "chat_history": st.session_state.chat_history,
                    "question": user_query
                })

                # Render answer and citations
                st.markdown(response.answer)
                st.caption(f"📖 **{response.module}** | Pages: {response.page_numbers} | Confidence: `{response.confidence}`")

                with st.expander("🔍 View Retrieved Sources"):
                    st.text(context)

                # Update state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "metadata": {
                        "module": response.module,
                        "pages": response.page_numbers,
                        "confidence": response.confidence
                    },
                    "sources": context
                })

                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content = response.answer))

            except Exception as e:
                st.error(f"Error generation response: {e}")