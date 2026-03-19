import faiss
import numpy as np
import os
import pickle


class VectorDB:
    def __init__(self):
        self.index = None   #for embeddings
        self.metadata = []  #for meta data(file info + chunks)



    #Building the fais index
    def build_index(self, embedded_chunks):

        embeddings = []

        for item in embedded_chunks:

            #collecting embeddings:
            embeddings.append(item["embedding"])


            #storing metadata:
            self.metadata.append({
                "file_path": item["file_path"],
                "chunk": item["chunk"]
            })

        # Safety check
        if len(embeddings) == 0:
            raise ValueError("No embeddings created. Check repository files or loader.")

        #FAISS requires data- embeddings in this format:
        embeddings = np.array(embeddings).astype("float32")

        #Extracting size of each embedding vector - no of features each chunk is represented by - vectors in the index must have the same dimensionality.
        dimension = embeddings.shape[1]

        #creates a FAISS index configured to store vectors of exact dimension and compare them using L2 distance (Euclidean distance)
        self.index = faiss.IndexFlatL2(dimension)

        #storing all vectors into the index.
        self.index.add(embeddings)



    #Searching
    def search(self, query_embedding, top_k=3):

        #Formatting the query accordng to the FAISS required format:
        query_embedding = np.array([query_embedding]).astype("float32")

        #FAISS performs similarity search: indices are positions of closest vectors &  distances are similarity scores
        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        #Retrieve corresponding metadata as the emnedding - on the basis of position
        for idx in indices[0]:
            results.append(self.metadata[idx])

        return results


    #Saving the index
    def save_index(self, index_path, metadata_path):

        #Checking if the directory where the fais index will be saved exists:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        #Writing the entire vector index (all embeddings and structure) to disk.
        faiss.write_index(self.index, index_path)

        #Saving metadata:
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)




    #Loading using file paths
    def load_index(self, index_path, metadata_path):

        #Reading saved file from disk and reconstructing the FAISS index in memory.
        self.index = faiss.read_index(index_path)

        # Reconstructing the original Python object (metadata list - file paths and chunk content)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)





# ---------------- TEST BLOCK ----------------
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