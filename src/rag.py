import re
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(
    collection_name="freshman_modules",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page_number", "Unkown")
        module = doc.metadata.get("module", "Module")
        formatted.append(f"[{module} | Page {page}]\n {doc.page_content}")
   
    return "\n\n --- \n\n".join(formatted)    

# Build an in-memory index of (module, chunk_index) -> id for instant neighbor lookup
def get_chunk_map(vectorstore):
    all_ids = vectorstore.get(include=[])["ids"]
    chunk_map = {}
    for doc_id in all_ids:
        m = re.match(r"([a-z_]+)_p\d+_\d+_c(\d+)", doc_id)
        if m:
            chunk_map[(m.group(1), int(m.group(2)))] = doc_id
    return chunk_map

CHUNK_MAP = get_chunk_map(vectorstore)

def expand_with_neighbors(docs, vectorstore, forward_steps=2):
    """Expands retrieved chunks with successor chunks to prevent boundary cutoffs."""
    if not docs:
        return docs

    expanded_ids = []
    seen = set()

    for doc in docs:
        if doc.id and doc.id not in seen:
            expanded_ids.append(doc.id)
            seen.add(doc.id)

        # Look up next chunks (e.g. idx + 1, idx + 2)
        if doc.id:
            m = re.match(r"([a-z_]+)_p\d+_\d+_c(\d+)", doc.id)
            if m:
                module, idx = m.group(1), int(m.group(2))
                for step in range(1, forward_steps + 1):
                    next_id = CHUNK_MAP.get((module, idx + step))
                    if next_id and next_id not in seen:
                        expanded_ids.append(next_id)
                        seen.add(next_id)

    fetched = vectorstore.get(ids=expanded_ids, include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta, id=doc_id)
        for doc_id, text, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])
    ]

class CitedAnswer(BaseModel):
    """Answer to the student's question with exact textbook citations."""
    answer: str = Field(description="The direct answer to the question based strictly on the context.")
    module: str = Field(description="The textbook module where the answer was found, e.g. 'History', 'Physics'.")
    page_numbers: list[int] = Field(description="List of exact page numbers containing the evidence, e.g. [79].")
    confidence: str = Field(description="Confidence level: 'high', 'medium', or 'low'.")


prompt_template = """You are an academic tutor for Ethiopian university freshman students.
Answer the question clearly and concisely based ONLY on the following textbook context.
Always cite the module and page number where you found the information.
If the answer cannot be found in the context, say "I cannot find this information in the provided textbook."

Context:
{context}

Question: {question}

Answer (with citation):"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# 1. Models
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
structured_llm = llm.with_structured_output(CitedAnswer)

# 2. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Contextualize Chain (for history rephrasing)
contextualize_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given a chat history and the latest user question which might reference context "
        "in the chat history, formulate a standalone question that can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it if needed "
        "and otherwise return it as is."
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])
rephrase_chain = contextualize_prompt | llm | StrOutputParser()

# 4. QA Answer Chain
qa_system_prompt = """You are an academic tutor for Ethiopian university freshman students.
Answer the question clearly and concisely based ONLY on the following textbook context.
Always cite the module and page number where you found the information.
If the answer cannot be found in the context, say "I cannot find this information in the provided textbook."

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])
answer_chain = qa_prompt | structured_llm

def get_rag_chain(module: str | None = None):
    search_kwargs = {"k": 3}
    if module:
        search_kwargs["filter"] = {"module": module}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | structured_llm
    )
    return rag_chain


if __name__ == "__main__":
    chat_history = []
    print("Freshman RAG Tutor is ready! (Type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ["exit", "quit", "q"]:
            break

        try:
            # 1. Rephrase if we have history
            if chat_history:
                search_query = rephrase_chain.invoke({
                    "chat_history": chat_history,
                    "question": user_input
                })
                search_query = str(search_query).strip()
                print(f"   (Searching database for: '{search_query}')")
            else:
                search_query = user_input

            # 2. Retrieve relevant chunks & expand boundaries
            raw_docs = retriever.invoke(search_query)
            docs = expand_with_neighbors(raw_docs, vectorstore)
            context = format_docs(docs)

            # 3. Generate grounded answer
            response = answer_chain.invoke({
                "context": context,
                "chat_history": chat_history,
                "question": user_input
            })

            # 4. Print response
            print(f"\nTutor: {response.answer}")
            print(f"[{response.module} | Pages: {response.page_numbers} | Confidence: {response.confidence}]\n")

            # 5. Append to history for the next turn
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response.answer))

        except Exception as e:
            print(f"\nTransient API error ({e}). Please try asking again!\n")  