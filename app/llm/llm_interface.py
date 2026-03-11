from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class LLMInterface:
    def __init__(self):
        self.llm=ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )
    
    def generate_answer(self, query, retrieved_chunks):

        context=""

        for item in retrieved_chunks:
            context+=item["chunk"] + "\n"

        prompt = f"""
You are a helpful AI that explains code.

Context:
{context}

Question:
{query}

Answer clearly. If you do not know the answer-say I don't have sufficient info.
"""
        
        response=self.llm.invoke(prompt)

        return response.content
    




# Test block
if __name__ == "__main__":

    from app.ingestion.repo_loader import load_repository
    from app.ingestion.chunker import chunk_code
    from app.embeddings.embedding_model import EmbeddingModel
    from app.vectorstore.vector_db import VectorDB
    from app.retrieval.retriever import Retriever

    repo_path = "data/repositories/sample_repo"

    # 1 Load repo
    files = load_repository(repo_path)

    # 2 Chunk code
    chunks = chunk_code(files)

    # 3 Create embeddings
    model = EmbeddingModel()
    embedded_chunks = model.create_embeddings(chunks)

    # 4 Build vector DB
    vector_db = VectorDB()
    vector_db.build_index(embedded_chunks)

    # 5 Retrieve relevant chunks
    retriever = Retriever(vector_db, model)

    query = "Explain the Employee class"

    retrieved_chunks = retriever.retrieve(query)

    # 6 Generate LLM answer
    llm = LLMInterface()

    answer = llm.generate_answer(query, retrieved_chunks)

    print("\nLLM Answer:\n")
    print(answer)