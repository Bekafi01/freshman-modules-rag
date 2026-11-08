from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.load import load_all_textbooks

def split_textbooks(docs: list[Document], 
                    chunk_size: int = 1000, 
                    chunk_overlap: int = 150
                    ) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        add_start_index = True,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

if __name__ == "__main__":
    path = Path("data/raw")
    doc = load_all_textbooks(path)
    docs = split_textbooks(doc)
    print(f"length of the document: {len(doc)}")
    print(f"length after chunking: {len(docs)}")  

