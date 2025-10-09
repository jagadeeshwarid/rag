import streamlit as st
from rag_query import load_vectorstore, init_qa_chain, ask_question

st.title("📚 RAG Question Answering")

# Input box
query = st.text_input("Enter your question:")

if query:
    vectorstore = load_vectorstore()
    qa_chain = init_qa_chain(vectorstore)
    answer = ask_question(query, qa_chain)
    
    st.subheader("Answer:")
    st.write(answer)
