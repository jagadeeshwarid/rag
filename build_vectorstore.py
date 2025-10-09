import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

def build_vectorstore(url: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    # Load documents from the web
    loader = WebBaseLoader(url)
    docs = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build FAISS vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore locally
    vectorstore.save_local("models/faiss_index")
    print("Vectorstore built and saved successfully!")
    return vectorstore

if __name__ == "__main__":
    build_vectorstore("https://www.ibm.com/think/topics/agent2agent-protocol")
