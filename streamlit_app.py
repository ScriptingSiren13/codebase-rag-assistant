import streamlit as st
import base64
from app.services.rag_pipeline import RAGPipeline


def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


bg_image = get_base64_image("assets/background.jpg")


st.set_page_config(
    page_title="Codebase RAG Assistant",
    layout="wide"
)


st.markdown(
    f"""
    <style>

    /* ===== Background ===== */
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    header {{
        background: transparent !important;
    }}

    .block-container {{
        padding-top: 3rem;
        background: transparent;
    }}

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {{
        background-color: #0b1220;
    }}

    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}

    /* ===== Title (grey-white + shadow) ===== */
    h1 {{
        color: #e5e7eb !important;  /* soft grey-white */
        text-shadow: 2px 2px 12px rgba(0,0,0,0.9);
        font-weight: 700;
    }}

    /* ===== Subheaders (like "Answer") ===== */
   h2, h3 {{
    color: #e5e7eb !important;
    text-shadow: 1px 1px 8px rgba(0,0,0,0.9);
}}

    p {{
        color: white;
    }}

    label {{
        color: white !important;
    }}

    /* ===== INPUTS (keep your original dark look) ===== */
    input, textarea {{
        background-color: #2d2d2d !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 6px !important;
        padding: 10px !important;
        caret-color: white !important;  /* FIX cursor */
    }}

    input:focus, textarea:focus {{
        outline: none !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
    }}

    ::placeholder {{
        color: #cccccc !important;
    }}

    /* ===== BUTTON (dark, simple, clean) ===== */
    .stButton > button {{
        background-color: #2d2d2d;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 6px;
        padding: 8px 18px;
        font-weight: 500;
    }}

    .stButton > button:hover {{
        background-color: #3a3a3a;
    }}

    /* ===== ANSWER BOX (subtle, not heavy) ===== */
    .answer-box {{
        background: rgba(0, 0, 0, 0.35);
        padding: 14px;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ===== Sidebar =====
st.sidebar.title("About")
st.sidebar.write("Codebase RAG Assistant")

st.sidebar.markdown("---")
st.sidebar.subheader("Connect")

# Cleaner professional icons (no childish look)
st.sidebar.markdown(
"""
<div style="display:flex; gap:20px; align-items:center; margin-top:10px;">

<a href="https://www.linkedin.com/in/zarnain-723a31325/" target="_blank">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="28" style="filter: brightness(0.85);">
</a>

<a href="https://github.com/ScriptingSiren13" target="_blank">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="28" style="filter: invert(1) brightness(0.9);">
</a>

<a href="mailto:zarnain.work@gmail.com">
<img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" width="28" style="filter: brightness(0.85);">
</a>

</div>
""",
unsafe_allow_html=True
)


# ===== Main =====
st.title("Codebase RAG Assistant")

st.write("Enter a GitHub repository and ask questions about the codebase.")


repo_url = st.text_input(
    "GitHub Repository URL",
    placeholder="https://github.com/user/repository"
)

question = st.text_input(
    "Question",
    placeholder="Example: How does routing work in Flask?"
)

run_button = st.button("Run RAG")


if run_button:

    if repo_url and question:

        with st.spinner("Processing repository and generating answer..."):

            if "rag_pipeline" not in st.session_state or st.session_state.get("current_repo") != repo_url:
                st.session_state.rag_pipeline = RAGPipeline(repo_url)
                st.session_state.current_repo = repo_url

            rag = st.session_state.rag_pipeline

            answer = rag.ask(question)

        st.subheader("Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    else:
        st.warning("Please enter both a repository URL and a question.")