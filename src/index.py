from pathlib import Path
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.load import load_single_pdf
from src.chunk import split_textbooks
from dotenv import load_dotenv
import time

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = Chroma(
    collection_name= "freshman_modules",
    embedding_function= embeddings,
    persist_directory= "./chroma_db"
)

def index_documents(vectorstore, chunks, batch_size = 100):

    ids = [
    f"{doc.metadata['module'].lower()}_p{doc.metadata['page_number']}_{doc.metadata.get('start_index', i)}_c{i}"
    for i, doc in enumerate(chunks)
    ]

    existing_ids = set(vectorstore.get(include=[])["ids"])

    new_chunks = []
    new_ids = []

    # Loop over the zipped pairs and check the condition
    for chunk, doc_id in zip(chunks, ids):
        if doc_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(doc_id)
        
    if not new_chunks:
        print("All chunks already indexed! Skipping API calls.")
        return vectorstore._collection.count()


    total = len(new_chunks)
    for start_idx in range(0, total, batch_size):
        end_idx = start_idx + batch_size

        # 1. Slice both lists using [start_idx : end_idx]
        batch_chunks = new_chunks[start_idx : end_idx]
        batch_ids = new_ids[start_idx : end_idx]
    
        # 2. Call vectorstore.add_documents(...) with just this batch
        vectorstore.add_documents(documents = batch_chunks, ids = batch_ids)

        # 3. Print progress
        print(f"Indexed {min(end_idx, total)}/{total} chunks")

        # 4. Pause briefly so you don't hit Gemini API rate limits
        if end_idx < total:
            time.sleep(62)


    return vectorstore._collection.count()

if __name__=="__main__":
    path = Path("data/raw/History.pdf")
    doc = load_single_pdf(path)
    docs = split_textbooks(doc)
    total_indexed = index_documents(vectorstore, docs)
    print(f"Total chunks in database: {total_indexed}")
