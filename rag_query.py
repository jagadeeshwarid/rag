import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.callbacks.tracers import LangChainTracer

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Rag_project"

# ---------------------------
# Load saved vectorstore
# ---------------------------
def load_vectorstore(path: str = "models/faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# ---------------------------
# Initialize QA Chain
# ---------------------------
def init_qa_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-8b-instant")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
    )
    return qa_chain

# ---------------------------
# Ask a question using the chain
# ---------------------------
def ask_question(query: str, qa_chain=None):
    if qa_chain is None:
        vectorstore = load_vectorstore()
        qa_chain = init_qa_chain(vectorstore)
    tracer = LangChainTracer(project_name="Rag_project")
    response = qa_chain.run(query, callbacks=[tracer])
    return response

# ---------------------------
# Test locally (run directly)
# ---------------------------
if __name__ == "__main__":
    vs = load_vectorstore()
    chain = init_qa_chain(vs)
    q = "How A2A protocol works?"
    print(ask_question(q, chain))
