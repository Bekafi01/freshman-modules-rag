from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(
    collection_name="freshman_modules",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs = {"k": 3})

def format_docs(docs):
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page_number", "Unkown")
        module = doc.metadata.get("module", "Module")
        formatted.append(f"[{module} | Page {page}]\n {doc.page_content}")
   
    return "\n\n --- \n\n".join(formatted)     

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

rag_chain = (
    {"context" : retriever | format_docs, "question" : RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    question = "When did Yekuno-Amlak send embassy to Egypt, and what was the purpose?"
    print(f"Question: {question}")
    response = rag_chain.invoke(question)
    print(f"Response: \n {response}")