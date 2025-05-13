import chainlit as cl
from rag_chain import build_rag_chain

@cl.on_chat_start
async def start():
    rag_chain = build_rag_chain()
    cl.user_session.set("rag_chain", rag_chain)
    await cl.Message(content="Hybrid RAG is ready! Ask me something from your PDFs.").send()

@cl.on_message
async def main(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    response = rag_chain.run(message.content)
    await cl.Message(content=response).send()
