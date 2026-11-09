from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = Chroma(
    collection_name="freshman_modules",
    embedding_function= embeddings,
    persist_directory="./chroma_db"
)

def search_history(query: str, k: int = 3):
    """
    Retrieve the top_k semantically similar documents to the query from the vectorstore
    """
    results = vectorstore.similarity_search_with_score(query, k = k)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- Rank {i} (Distance Score: {score:.4f}) ---")
        print(f"Page Number: {doc.metadata['page_number']}")
        print(f"Content: {doc.page_content[:250]}...\n")
        

    return results
    
if __name__ == "__main__":
    search_history("When did Yekuno-Amlak send embassy to Egypt?")