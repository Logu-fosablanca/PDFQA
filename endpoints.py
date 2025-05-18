from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
import uuid

from rag_chain import build_rag_chain
from memory import get_memory, save_message

app = FastAPI()

# Keep RAG chains in memory per session
active_chains = {}

class QueryRequest(BaseModel):
    query: str

# ✅ Dependency to properly expose session_id header in Swagger
def get_session_id(session_id: str = Header(..., convert_underscores=False)):
    return session_id

@app.post("/start")
def start_session():
    session_id = str(uuid.uuid4())
    memory = get_memory(session_id)
    chain = build_rag_chain(memory)
    active_chains[session_id] = chain
    return {"session_id": session_id, "message": "RAG is ready!"}

@app.post("/query")
def query_rag(req: QueryRequest, session_id: str = Depends(get_session_id)):
    if session_id not in active_chains:
        raise HTTPException(status_code=400, detail="Invalid or expired session_id")
    
    chain = active_chains[session_id]
    query = req.query

    # Run RAG
    response = chain.run(query)

    # Save chat history
    save_message(session_id, "user", query)
    save_message(session_id, "ai", response)

    return {"response": response}
