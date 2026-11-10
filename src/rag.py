from operator import mod
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(
    collection_name="freshman_modules",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


def format_docs(docs):
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page_number", "Unkown")
        module = doc.metadata.get("module", "Module")
        formatted.append(f"[{module} | Page {page}]\n {doc.page_content}")
   
    return "\n\n --- \n\n".join(formatted)     

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

llm = ChatGoogleGenerativeAI(model = "gemini-3.6-flash")

structured_llm = llm.with_structured_output(CitedAnswer)

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

    question = "What is Archimedes' principle and what does buoyant force depend on?"
    print(f"Question: {question}")

    chain = get_rag_chain()

    response = chain.invoke(question)
    print(response.answer)       
    print(response.module)       
    print(response.page_numbers)  
    print(response.confidence)    