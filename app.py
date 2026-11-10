import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage, AIMessage
from src.rag import rephrase_chain, answer_chain, format_docs, vectorstore

def scroll_to_bottom():
    """Smoothly scrolls the page to the bottom of the chat container."""
    components.html(
        """
        <script>
            function autoScroll() {
                const main = window.parent.document.querySelector('.main') 
                          || window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
                if (main) {
                    main.scrollTo({ top: main.scrollHeight, behavior: 'smooth' });
                }
            }
            // Trigger immediately and once more after DOM rendering finishes
            autoScroll();
            setTimeout(autoScroll, 200);
        </script>
        """,
        height=0,
    )

# 1. Page Configuration & Custom CSS
st.set_page_config(
    page_title="Freshman RAG Tutor",
    page_icon="🎓",
    layout="centered"
)

# Subtle styling improvements
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #4a90e2;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Freshman Modules RAG Tutor")
st.caption("Grounded academic assistant for Ethiopian university freshman textbooks.")

# 2. Sidebar Controls
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Module filter (Supports all 4 modules)
    selected_module = st.selectbox(
        "Focus on specific module:",
        options=["All Modules", "History", "Physics", "Logic", "Emerging Technologies"],
        index=0
    )
    
    # Adjustable Top-K
    top_k = st.slider("Passages to retrieve (k):", min_value=1, max_value=5, value=3)
    
    # Database stats
    chunk_count = vectorstore._collection.count()
    st.metric("Total Chunks in Database", f"{chunk_count:,}")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# 3. Dynamic Retriever Configuration
search_kwargs = {"k": top_k}
if selected_module != "All Modules":
    # Match various filename/capitalization conventions
    mod_variants = [
        selected_module,
        selected_module.lower(),
        selected_module.replace(" ", "_"),
        selected_module.lower().replace(" ", "_")
    ]
    search_kwargs["filter"] = {"module": {"$in": mod_variants}}

retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

# 4. Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Helper: Render confidence badge
def render_confidence_badge(confidence: str) -> str:
    conf = confidence.lower()
    if conf == "high":
        return "🟢 :green-background[High Confidence]"
    elif conf == "medium":
        return "🟡 :orange-background[Medium Confidence]"
    return "🔴 :red-background[Low Confidence]"

# 5. Display Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg and msg["metadata"]:
            meta = msg["metadata"]
            st.caption(
                f"📖 **{meta['module']}** | Pages: {meta['pages']} | {render_confidence_badge(meta['confidence'])}"
            )
            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 View Retrieved Sources"):
                    for passage in msg["sources"]:
                        st.markdown(f"**[{passage['module']} | Page {passage['page']}]**")
                        st.info(passage["text"])

# 6. Starter Prompts (Shown only when conversation is empty)
prompt_to_submit = None
if not st.session_state.messages:
    st.markdown("##### 💡 Try asking:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📜 When was the Solomonic dynasty restored?", use_container_width=True):
            prompt_to_submit = "When was the Solomonic dynasty restored?"
        if st.button("🧠 What makes an argument deductive vs inductive?", use_container_width=True):
            prompt_to_submit = "What makes an argument deductive vs inductive?"
    with col2:
        if st.button("⚡ What equations calculate Archimedes buoyant force?", use_container_width=True):
            prompt_to_submit = "What equations are used to calculate Archimedes' principle buoyant force?"
        if st.button("🤖 What are the core drivers of the 4th Industrial Revolution?", use_container_width=True):
            prompt_to_submit = "What are the core drivers of the 4th Industrial Revolution?"

# 7. User Input Handling
user_query = st.chat_input("Ask a question from History, Physics, Logic, or Emerging Tech...")

# Use either typed query or clicked starter button
active_prompt = prompt_to_submit or user_query

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching textbooks and reasoning..."):
            try:
                # A. Rephrase query if history exists
                if st.session_state.chat_history:
                    rephrased = rephrase_chain.invoke({
                        "chat_history": st.session_state.chat_history,
                        "question": active_prompt
                    })
                    search_query = str(rephrased).strip()
                    st.caption(f"🔎 *Searching database for: '{search_query}'*")
                else:
                    search_query = active_prompt

                # B. Retrieve relevant chunks
                docs = retriever.invoke(search_query)
                context = format_docs(docs)

                # Format structured source passages for clean UI display
                sources_data = [
                    {
                        "module": d.metadata.get("module", "Textbook"),
                        "page": d.metadata.get("page_number", "Unknown"),
                        "text": d.page_content.strip()
                    }
                    for d in docs
                ]

                # C. Generate cited answer
                response = answer_chain.invoke({
                    "context": context,
                    "chat_history": st.session_state.chat_history,
                    "question": active_prompt
                })

                # D. Render answer and badges
                st.markdown(response.answer)
                st.caption(
                    f"📖 **{response.module}** | Pages: {response.page_numbers} | {render_confidence_badge(response.confidence)}"
                )
                
                with st.expander("🔍 View Retrieved Sources"):
                    for passage in sources_data:
                        st.markdown(f"**[{passage['module']} | Page {passage['page']}]**")
                        st.info(passage["text"])
                
                scroll_to_bottom()

                # E. Update session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "metadata": {
                        "module": response.module,
                        "pages": response.page_numbers,
                        "confidence": response.confidence
                    },
                    "sources": sources_data
                })

                st.session_state.chat_history.append(HumanMessage(content=active_prompt))
                st.session_state.chat_history.append(AIMessage(content=response.answer))

                # 🚀 Keep viewport scrolled to bottom whenever conversation exists
                if st.session_state.messages:
                    scroll_to_bottom()

            except Exception as e:
                st.error(f"Error generating response: {e}")