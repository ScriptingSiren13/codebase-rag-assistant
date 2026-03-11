class Retriever:

    def __init__(self, vector_db, embedding_model):
        self.vector_db=vector_db
        self.embedding_model= embedding_model

    
    def retrieve(self, query, top_k=3):
        query_embedding=self.embedding_model.model.encode(query)

        results=self.vector_db.search(query_embedding, top_k)

        return results
    

    


# Test block
if __name__ == "__main__":

    from app.ingestion.repo_loader import load_repository
    from app.ingestion.chunker import chunk_code
    from app.embeddings.embedding_model import EmbeddingModel
    from app.vectorstore.vector_db import VectorDB

    repo_path = "data/repositories/sample_repo"

    files = load_repository(repo_path)

    chunks = chunk_code(files)

    model = EmbeddingModel()

    embedded_chunks = model.create_embeddings(chunks)

    vector_db = VectorDB()

    vector_db.build_index(embedded_chunks)

    retriever = Retriever(vector_db, model)

    query = "authentication login function"

    results = retriever.retrieve(query)

    print("Retrieved Chunks:\n")

    for r in results:
        print(r["file_path"])
        print(r["chunk"][:150])
        print("-" * 40)