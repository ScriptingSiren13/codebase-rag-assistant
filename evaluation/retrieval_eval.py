import json
import os
from app.services.rag_pipeline import RAGPipeline

with open("evaluation/dataset/daily_commit_dataset.json") as f:
    dataset = json.load(f)

repo_url = "https://github.com/ScriptingSiren13/The-Daily-Commit"
rag = RAGPipeline(repo_url)

k = 5

recall_hits = 0
precision_total = 0
mrr_total = 0

failures = []

for item in dataset:

    query = item["query"]
    expected_doc = item["expected_doc"].replace("\\", "/").lower()

    results = rag.retriever.retrieve(query, top_k=k)

    # Keep full path (normalized)
    retrieved_files = [
        r["file_path"].replace("\\", "/").lower()
        for r in results
    ]

    print(f"\nQuery: {query}")
    print(f"Expected: {expected_doc}")
    print(f"Retrieved: {retrieved_files}")

    # ---------- Recall ----------
    if any(expected_doc in file for file in retrieved_files):
        recall_hits += 1
    else:
        failures.append({
            "query": query,
            "expected": expected_doc,
            "retrieved": retrieved_files
        })

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


recall_at_k = recall_hits / len(dataset)
precision_at_k = precision_total / len(dataset)
mrr = mrr_total / len(dataset)

print(f"\nRecall@{k}: {recall_at_k}")
print(f"Precision@{k}: {precision_at_k}")
print(f"MRR: {mrr}")

print("\nFailures:")
for f in failures:
    print(f)