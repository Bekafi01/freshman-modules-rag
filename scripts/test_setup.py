from langchain_core import embeddings
from dotenv import load_dotenv
load_dotenv()

# Verify importss
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

print("All packages imported successfully!")

# Test LLM invocation
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
response = llm.invoke("Say 'Environment setup complete' in three words." )
print(f"LLM Response: {response.text}")

#Test Embedding generation
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
sample_vector = embeddings.embed_query("Ethiopian freshman physics and logic")
print(f"Embeddding Dimension: {len(sample_vector)}(expected: 3072)")