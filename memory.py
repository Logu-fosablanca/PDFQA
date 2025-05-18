from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from db import SessionLocal, ChatHistory

def get_message_history(session_id: str):
    db = SessionLocal()
    records = db.query(ChatHistory).filter(ChatHistory.session_id == session_id).all()
    history = ChatMessageHistory()

    for rec in records:
        if rec.role == "user":
            history.add_user_message(rec.content)
        else:
            history.add_ai_message(rec.content)
    return history

def save_message(session_id: str, role: str, content: str):
    db = SessionLocal()
    msg = ChatHistory(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()

def get_memory(session_id: str):
    history = get_message_history(session_id)
    memory = ConversationBufferMemory(chat_memory=history, return_messages=True)
    return memory
