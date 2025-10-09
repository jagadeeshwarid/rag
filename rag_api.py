from fastapi import FastAPI
from pydantic import BaseModel
from rag_query import load_vectorstore, init_qa_chain, ask_question

app = FastAPI()

vectorstore = load_vectorstore()
qa_chain = init_qa_chain(vectorstore)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    answer = ask_question(query.question, qa_chain)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
