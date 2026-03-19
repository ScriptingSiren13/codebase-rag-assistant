from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class LLMInterface:

    def __init__(self):

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

    def generate_answer(self, query, retrieved_chunks):

        # Build context from retrieved chunks
        context = ""

        for item in retrieved_chunks:

            context += f"File: {item['file_path']}\n"
            context += item["chunk"]
            context += "\n\n"

        prompt = f"""
You are an AI assistant that explains code from a repository.

Use the provided code context to answer the question.

Important rules:
- Use the code snippets to infer the answer.
- The answer may not be explicitly written; infer it from the code.
- Mention relevant files if useful.
- Only say you don't have enough information if the context is completely unrelated.

Code Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)

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
    retriever = Retriever(vector_db, model, chunks)

    query = "Explain the Employee class"

    retrieved_chunks = retriever.retrieve(query)

    # 6 Generate LLM answer
    llm = LLMInterface()

    answer = llm.generate_answer(query, retrieved_chunks)

    print("\nLLM Answer:\n")
    print(answer)