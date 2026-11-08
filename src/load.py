from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from langchain_core.documents import Document
import re

def clean_text(text: str) -> str:
    # 1. Rejoin hyphenated words split across lines
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # 2. Collapse 3+ newlines into a standard paragraph break (\n\n)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    # 3. Collapse multiple spaces and tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def load_single_pdf(pdf_path: Path) -> list[Document]:
    loader = PyMuPDFLoader(str(pdf_path))

    docs = loader.load()

    cleaned_docs = []
    for doc in docs:
        cleaned = clean_text(doc.page_content)
        doc.page_content = cleaned
        doc.metadata["module"] = pdf_path.stem
        doc.metadata["file_name"] = pdf_path.name
        doc.metadata["page_number"] = doc.metadata["page"] + 1
        cleaned_docs.append(doc)

    return cleaned_docs

def load_all_textbooks(data_dir: Path) -> list[Document]:
    all_docs = []
    pdf_files = sorted(data_dir.glob("*.pdf"))

    for pdf_path in pdf_files:
        docs = load_single_pdf(pdf_path)
        all_docs.extend(docs)
        print(f"Loaded {len(docs)} pages from {pdf_path.name}")

    print(f"Total pages loaded across all modules: {len(all_docs)}")
    return all_docs



if __name__ == "__main__":
    path = Path("data/raw")
    docs = load_all_textbooks(path)

