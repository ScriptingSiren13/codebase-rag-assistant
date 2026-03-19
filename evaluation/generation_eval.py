import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.services.rag_pipeline import RAGPipeline


# Load evaluation dataset
with open("evaluation/dataset/rag_repo_dataset.json") as f:
    dataset = json.load(f)


# Repository to evaluate
repo_url = "https://github.com/ScriptingSiren13/RAG"

rag = RAGPipeline(repo_url)


questions = []
answers = []
contexts = []
ground_truths = []


for item in dataset:

    query = item["query"]

    # Use same top_k as retrieval eval
    retrieved_chunks = rag.retriever.retrieve(query, top_k=5)

    context_text = [chunk["chunk"] for chunk in retrieved_chunks]

    # Generate answer
    answer = rag.llm.generate_answer(query, retrieved_chunks)

    # -----------------------------
    # FIX: Proper ground truth text
    # -----------------------------
    ground_truth_text = ""

    for chunk in retrieved_chunks:
        if item["expected_doc"].lower() in chunk["file_path"].lower():
            ground_truth_text += chunk["chunk"] + "\n"

    questions.append(query)
    answers.append(answer)
    contexts.append(context_text)
    ground_truths.append(ground_truth_text if ground_truth_text else "")


# Create dataset for RAGAS
eval_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})


# LLM + Embeddings for evaluation
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()


# Run evaluation
result = evaluate(
    eval_dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision()
    ],
    llm=llm,
    embeddings=embeddings
)


print(result)