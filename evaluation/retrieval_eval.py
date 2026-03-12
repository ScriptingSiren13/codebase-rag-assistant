import json
from app.services.rag_pipeline import RAGPipeline

# Load evaluation dataset
with open("evaluation/dataset/rag_repo_dataset.json") as f:
    dataset = json.load(f)

# Repository to evaluate
repo_url = "https://github.com/ScriptingSiren13/RAG"

rag = RAGPipeline(repo_url)

k = 3

recall_hits = 0
precision_total = 0
mrr_total = 0

for item in dataset:

    query = item["query"]
    expected_doc = item["expected_doc"]

    results = rag.retriever.retrieve(query, top_k=k)

    retrieved_files = [r["file_path"] for r in results]

    # ---------- Recall ----------
    if any(expected_doc in file for file in retrieved_files):
        recall_hits += 1

    # ---------- Precision ----------
    correct_in_top_k = sum(expected_doc in file for file in retrieved_files)
    precision_total += correct_in_top_k / k

    # ---------- MRR ----------
    rank = None
    for i, file in enumerate(retrieved_files):
        if expected_doc in file:
            rank = i + 1
            break

    if rank is not None:
        mrr_total += 1 / rank


# Final metrics
recall_at_k = recall_hits / len(dataset)
precision_at_k = precision_total / len(dataset)
mrr = mrr_total / len(dataset)

print(f"Recall@{k}: {recall_at_k}")
print(f"Precision@{k}: {precision_at_k}")
print(f"MRR: {mrr}")