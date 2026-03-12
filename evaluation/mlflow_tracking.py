import mlflow

# Set experiment
mlflow.set_experiment("Codebase_RAG_Evaluation")


# -----------------------------
# Run 1 — RAG Repository
# -----------------------------
with mlflow.start_run(run_name="vector_rag_rag_repo"):

    # Version of system
    mlflow.log_param("rag_version", "v1_vector_rag")

    # System parameters
    mlflow.log_param("repository", "RAG Repo")
    mlflow.log_param("retrieval_method", "vector_search")
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("top_k", 3)

    # Retrieval metrics
    mlflow.log_metric("recall_at_3", 1.0)
    mlflow.log_metric("precision_at_3", 0.366)
    mlflow.log_metric("mrr", 0.783)

    # Generation metrics
    mlflow.log_metric("faithfulness", 0.7766)
    mlflow.log_metric("answer_relevancy", 0.9062)
    mlflow.log_metric("context_precision", 0.0)


# -----------------------------
# Run 2 — Daily Commit Repo
# -----------------------------
with mlflow.start_run(run_name="vector_rag_daily_commit_repo"):

    # Version of system
    mlflow.log_param("rag_version", "v1_vector_rag")

    # System parameters
    mlflow.log_param("repository", "Daily Commit Repo")
    mlflow.log_param("retrieval_method", "vector_search")
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("top_k", 3)

    # Retrieval metrics
    mlflow.log_metric("recall_at_3", 0.0)
    mlflow.log_metric("precision_at_3", 0.0)
    mlflow.log_metric("mrr", 0.0)

    # Generation metrics
    mlflow.log_metric("faithfulness", 0.90)
    mlflow.log_metric("answer_relevancy", 0.0)
    mlflow.log_metric("context_precision", 0.0)


print("Both experiments logged to MLflow.")