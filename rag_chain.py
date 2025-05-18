import os
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import pickle


DB_DIR = "data/faiss_index"

def build_rag_chain(memory):
    # Load Ollama model and embeddings
    llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama3.1:latest"))
    embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_MODEL", "llama3.1:latest"))

    # Load FAISS vector DB (dense retriever)
    db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    dense_retriever = db.as_retriever()

    # Load docs and set up sparse retriever (BM25)
    with open("data/docs.pkl", "rb") as f:
        docs = pickle.load(f)
    sparse_retriever = BM25Retriever.from_documents(docs)

    # Combine both into an ensemble retriever
    retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.7, 0.3],  # You can tune these weights
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    return chain

