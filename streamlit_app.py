import streamlit as st
from app.services.rag_pipeline import RAGPipeline

st.title("Codebase RAG Assistant")

repo_url = st.text_input("Enter GitHub Repository URL")

question = st.text_input("Ask a question about the repository")

if st.button("Run RAG"):

    if repo_url and question:

        st.write("Processing repository...")

        rag = RAGPipeline(repo_url)

        answer = rag.ask(question)

        st.subheader("Answer")
        st.write(answer)

    else:
        st.warning("Please enter both a repository URL and a question.")