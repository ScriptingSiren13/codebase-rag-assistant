import mlflow

mlflow.set_experiment("Codebase_RAG_Evaluation")


# -----------------------------
# Run 1 — RAG Repo (v2)
# -----------------------------
with mlflow.start_run(run_name="hybrid_rag_rag_repo_v2"):

    mlflow.log_param("rag_version", "v2_hybrid_rag")
    mlflow.log_param("repository", "RAG Repo")
    mlflow.log_param("retrieval_method", "hybrid (vector + BM25 + RRF)")
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("top_k", 5)

    # Retrieval
    mlflow.log_metric("recall_at_5", 1.0)
    mlflow.log_metric("precision_at_5", 0.20)
    mlflow.log_metric("mrr", 0.545)

    # Generation (RAGAS)
    mlflow.log_metric("faithfulness", 0.8981)
    mlflow.log_metric("answer_relevancy", 0.9519)
    mlflow.log_metric("context_precision", 0.5187)


# -----------------------------
# Run 2 — Daily Commit Repo (v2)
# -----------------------------
with mlflow.start_run(run_name="hybrid_rag_daily_commit_v2"):

    mlflow.log_param("rag_version", "v2_hybrid_rag")
    mlflow.log_param("repository", "Daily Commit Repo")
    mlflow.log_param("retrieval_method", "hybrid (vector + BM25 + RRF)")
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("top_k", 5)

    # Retrieval
    mlflow.log_metric("recall_at_5", 1.0)
    mlflow.log_metric("precision_at_5", 0.20)
    mlflow.log_metric("mrr", 1.0)

    # Generation (RAGAS)
    mlflow.log_metric("faithfulness", 0.8997)
    mlflow.log_metric("answer_relevancy", 0.8707)
    mlflow.log_metric("context_precision", 0.0417)


print("All v2 experiments logged to MLflow.")