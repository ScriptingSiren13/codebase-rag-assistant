# Step 1: Import required modules
import os
import hashlib

from app.ingestion.github_loader import clone_repository
from app.ingestion.repo_loader import load_repository
from app.ingestion.chunker import chunk_code
from app.embeddings.embedding_model import EmbeddingModel
from app.vectorstore.vector_db import VectorDB
from app.retrieval.retriever import Retriever
from app.llm.llm_interface import LLMInterface


class RAGPipeline:

    def __init__(self, repo_url):

        # Step 2: Clone the repository
        repo_path = clone_repository(repo_url)
        print("Repository path:", repo_path)

        # Step 3: Load files
        files = load_repository(repo_path)
        print("Files loaded:", len(files))

        if len(files) == 0:
            raise ValueError("No files loaded from repository.")

        # Step 4: Chunk code
        chunks = chunk_code(files)
        print("Chunks created:", len(chunks))

        if len(chunks) == 0:
            raise ValueError("Chunking failed.")

        # Step 5: Initialize embedding model
        self.embedding_model = EmbeddingModel()

        # Step 6: Create vector DB
        self.vector_db = VectorDB()

        # ✅ NEW: Create repo-specific index path
        repo_hash = hashlib.md5(repo_url.encode()).hexdigest()

        index_dir = f"data/index/{repo_hash}"
        os.makedirs(index_dir, exist_ok=True)

        index_path = f"{index_dir}/faiss.index"
        metadata_path = f"{index_dir}/metadata.pkl"

        # Step 7: Check if index already exists
        if os.path.exists(index_path):

            print("Loading existing FAISS index...")
            self.vector_db.load_index(index_path, metadata_path)

        else:

            print("Building new FAISS index...")

            embedded_chunks = self.embedding_model.create_embeddings(chunks)

            print("Embeddings created:", len(embedded_chunks))

            self.vector_db.build_index(embedded_chunks)

            # Save index for future runs
            self.vector_db.save_index(index_path, metadata_path)

        # Step 8: Initialize hybrid retriever
        self.retriever = Retriever(self.vector_db, self.embedding_model, chunks)

        # Step 9: Initialize LLM
        self.llm = LLMInterface()


    def ask(self, query):

        # Step 10: Hybrid retrieval
        retrieved_chunks = self.retriever.retrieve(query)

        # DEBUG: Print retrieved chunks
        print("\nRetrieved Chunks:\n")
        for chunk in retrieved_chunks:
            print(chunk["file_path"])

        # Step 11: Generate answer using LLM
        answer = self.llm.generate_answer(query, retrieved_chunks)

        return answer


# ---------------- TEST BLOCK ----------------

if __name__ == "__main__":

    repo_url = "https://github.com/pallets/flask"

    rag = RAGPipeline(repo_url)

    query = "Explain routing in Flask"

    answer = rag.ask(query)

    print("\nFinal Answer:\n")
    print(answer)