import faiss
import numpy as np


class VectorDB:

    def __init__(self):
        self.index = None
        self.metadata = []

    def build_index(self, embedded_chunks):

        embeddings = []

        for item in embedded_chunks:

            embeddings.append(item["embedding"])

            self.metadata.append({
                "file_path": item["file_path"],
                "chunk": item["chunk"]
            })

        # Safety check
        if len(embeddings) == 0:
            raise ValueError("No embeddings created. Check repository files or loader.")

        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)

    def search(self, query_embedding, top_k=3):

        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for idx in indices[0]:
            results.append(self.metadata[idx])

        return results


# Test block
if __name__ == "__main__":

    from app.ingestion.repo_loader import load_repository
    from app.ingestion.chunker import chunk_code
    from app.embeddings.embedding_model import EmbeddingModel

    repo_path = "data/repositories/sample_repo"

    files = load_repository(repo_path)

    print("Files loaded:", len(files))

    chunks = chunk_code(files)

    print("Chunks created:", len(chunks))

    model = EmbeddingModel()

    embedded_chunks = model.create_embeddings(chunks)

    print("Embeddings created:", len(embedded_chunks))

    vector_db = VectorDB()

    vector_db.build_index(embedded_chunks)

    query = "authentication login function"

    query_embedding = model.model.encode(query)

    results = vector_db.search(query_embedding)

    print("Search Results:\n")

    for r in results:
        print(r["file_path"])
        print(r["chunk"][:150])
        print("-" * 40)