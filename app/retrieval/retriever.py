from app.retrieval.keyword_retriever import KeywordRetriever


class Retriever:

    def __init__(self, vector_db, embedding_model, chunks):

        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.keyword_retriever = KeywordRetriever(chunks)


    def retrieve(self, query, top_k=10):

        # Encode query:
        query_embedding = self.embedding_model.model.encode(query)

        #Semantic (vector) retrieval is performed-chunks that are semantically similar:
        vector_results = self.vector_db.search(query_embedding, top_k)

        # Keyword retrieval- returns chunks where exact words or tokens match:
        keyword_results = self.keyword_retriever.search(query, top_k)

        rrf_scores = {}    #for storing combined scores
        chunk_lookup = {}  #maps chunk_id to actual chunk

        k = 60      #constant used in RRF scoring




        # Vector scoring
        for rank, chunk in enumerate(vector_results): #Loop through vector results with ranking.
            chunk_id = (chunk["file_path"], chunk["chunk"]) #Create a unique identifier (tuple) for each chunk.

            chunk_lookup[chunk_id] = chunk   #in lookup, there is chunk id: chunk

            score = 1 / (k + rank + 1)     #rff score for each rank of chunk

            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + score  #combines scores from multiple retrieval methods- chunk id: score



        # Keyword scoring
        for rank, chunk in enumerate(keyword_results):

            chunk_id = (chunk["file_path"], chunk["chunk"])

            chunk_lookup[chunk_id] = chunk

            score = 1 / (k + rank + 1)

            # Keyword boost
            score *= 1.3

            # Token normalization
            query_tokens = (
                query.lower()
                .replace("-", " ")
                .replace("_", " ")
                .split()
            )

            # Boost filename token matches
            for token in query_tokens:
                if token in chunk["file_path"].lower():
                    score *= 3

            # Exact filename boost
            if query.lower() in chunk["file_path"].lower():
                score *= 4

            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + score


        # Rank results
        ranked_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Remove duplicates while preserving order
        seen_files = set()
        final_results = []

        for chunk_id, _ in ranked_chunks:
            chunk = chunk_lookup[chunk_id]
            file_path = chunk["file_path"]

            if file_path not in seen_files:
                final_results.append(chunk)
                seen_files.add(file_path)

            if len(final_results) == top_k:
                break

        return final_results


# -------------------------
# TEST BLOCK
# -------------------------
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

    retriever = Retriever(vector_db, model, chunks)

    query = "authentication login function"

    results = retriever.retrieve(query)

    print("Hybrid Retrieval Results:\n")

    for r in results:
        print(r["file_path"])
        print(r["chunk"][:150])
        print("-" * 40)