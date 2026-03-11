from sentence_transformers import SentenceTransformer

class EmbeddingModel:

    def __init__(self):
        self.model=SentenceTransformer("all-MiniLM-L6-v2")

    def create_embeddings(self, chunks):

        embedded_chunks=[]

        for item in chunks:
            chunk_text=item["chunk"]

            embedding=self.model.encode(chunk_text)

            embedded_chunks.append({
                "file_path":item["file_path"],
                "chunk":chunk_text,
                "embedding":embedding
            })
        return embedded_chunks
    

# Testing block
if __name__ == "__main__":

    from app.ingestion.repo_loader import load_repository
    from app.ingestion.chunker import chunk_code

    repo_path = "data/repositories/sample_repo"

    files = load_repository(repo_path)

    chunks = chunk_code(files)

    model = EmbeddingModel()

    embedded_chunks = model.create_embeddings(chunks)

    print("Total chunks embedded:", len(embedded_chunks))

    print("\nExample embedding length:")
    print(len(embedded_chunks[0]["embedding"]))