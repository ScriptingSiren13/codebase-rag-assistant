from rank_bm25 import BM25Okapi


class KeywordRetriever:
    def __init__(self, chunks):

        self.chunks = chunks

        # Combine file path + chunk text
        self.documents = [
            chunk["file_path"] + " " + chunk["chunk"]
            for chunk in chunks
        ]

        # Tokenization + normalization
        tokenized_docs = [
            doc.lower()
            .replace("-", " ")
            .replace("_", " ")
            .split()
            for doc in self.documents     #each document becomes a list of normalized tokens (words).
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)



    def search(self, query, top_k=5):
        # Normalize query tokens
        tokenized_query = (
            query.lower()
            .replace("-", " ")
            .replace("_", " ")
            .split()
        )

        #Compute BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        #sort indices based on their scores
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        #Retrieve the top k chunk
        results = [self.chunks[i] for i in ranked_indices[:top_k]]

        return results