from app.ingestion.github_loader import clone_repository
from app.ingestion.repo_loader import load_repository
from app.ingestion.chunker import chunk_code
from app.embeddings.embedding_model import EmbeddingModel
from app.vectorstore.vector_db import VectorDB
from app.retrieval.retriever import Retriever
from app.llm.llm_interface import LLMInterface


class RAGPipeline:

    def __init__(self, repo_url):

        # Clone repo
        repo_path = clone_repository(repo_url)

        print("Repository path:", repo_path)

        # Load files
        files = load_repository(repo_path)

        print("Files loaded:", len(files))

        if len(files) == 0:
            raise ValueError("No files loaded from repository. Check loader extensions.")

        # Chunk code
        chunks = chunk_code(files)

        print("Chunks created:", len(chunks))

        if len(chunks) == 0:
            raise ValueError("Chunking failed. No chunks created.")

        # Embeddings
        self.embedding_model = EmbeddingModel()

        embedded_chunks = self.embedding_model.create_embeddings(chunks)

        print("Embeddings created:", len(embedded_chunks))

        # Vector DB
        self.vector_db = VectorDB()

        self.vector_db.build_index(embedded_chunks)

        # Retriever
        self.retriever = Retriever(self.vector_db, self.embedding_model)

        # LLM
        self.llm = LLMInterface()

    def ask(self, query):

        retrieved_chunks = self.retriever.retrieve(query)

        answer = self.llm.generate_answer(query, retrieved_chunks)

        return answer


# Test block
if __name__ == "__main__":

    repo_url = "https://github.com/pallets/flask"

    rag = RAGPipeline(repo_url)

    query = "Explain routing in Flask"

    answer = rag.ask(query)

    print("\nFinal Answer:\n")
    print(answer)