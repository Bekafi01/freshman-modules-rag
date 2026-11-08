from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from langchain_core.documents import Document


def load_single_pdf(pdf_path: Path) -> list[Document]:
    loader = PyMuPDFLoader(str(pdf_path))

    docs = loader.load()

    for doc in docs:
        doc.metadata["module"] = pdf_path.stem
        doc.metadata["file_name"] = pdf_path.name
        doc.metadata["page_number"] = doc.metadata["page"] + 1

    return docs

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

