from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True)
    session_id = Column(String)
    role = Column(String)  # 'user' or 'ai'
    content = Column(Text)

engine = create_engine('sqlite:///chat.db')
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(engine)
