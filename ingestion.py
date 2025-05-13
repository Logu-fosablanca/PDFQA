from utils import load_and_split_docs
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import os
import shutil
import pickle
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "data/faiss_index"
DOCS_PKL_PATH = "data/docs.pkl"

def ingest_docs(directory: str):
    # Delete old vector store and docs if they exist
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    if os.path.exists(DOCS_PKL_PATH):
        os.remove(DOCS_PKL_PATH)

    # Load and split the documents
    docs = load_and_split_docs(directory)

    # Save docs for BM25 (sparse retriever)
    os.makedirs("data", exist_ok=True)
    with open(DOCS_PKL_PATH, "wb") as f:
        pickle.dump(docs, f)

    # Load embeddings model
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
    embeddings = OllamaEmbeddings(model=model_name)

    # Create vector store and save it
    db = FAISS.from_documents(docs, embedding=embeddings)
    db.save_local(DB_DIR)

    return f"Ingested {len(docs)} chunks and saved for dense + sparse retrieval."

if __name__ == "__main__":
    print(ingest_docs("docs"))
