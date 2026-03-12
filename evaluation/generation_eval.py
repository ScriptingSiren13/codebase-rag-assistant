import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.services.rag_pipeline import RAGPipeline


# Load evaluation dataset
with open("evaluation/dataset/daily_commit_dataset.json") as f:
    dataset = json.load(f)


repo_url = "https://github.com/ScriptingSiren13/The-Daily-Commit"

rag = RAGPipeline(repo_url)


questions = []
answers = []
contexts = []
ground_truths = []


for item in dataset:

    query = item["query"]

    retrieved_chunks = rag.retriever.retrieve(query)

    context_text = [chunk["chunk"] for chunk in retrieved_chunks]

    answer = rag.llm.generate_answer(query, retrieved_chunks)

    questions.append(query)
    answers.append(answer)
    contexts.append(context_text)
    ground_truths.append(item["expected_doc"])



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

