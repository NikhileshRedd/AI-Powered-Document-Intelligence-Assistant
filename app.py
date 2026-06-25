import streamlit as st
import pdfplumber
import faiss
import numpy as np
import re

from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color:#f5f7fa;
}

.hero {
    background: linear-gradient(135deg,#1e3c72,#2a5298);
    padding:40px;
    border-radius:20px;
    text-align:center;
    color:white;
    margin-bottom:25px;
}

.answer-box{
    background:white;
    padding:20px;
    border-radius:15px;
    border-left:8px solid #2a5298;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
    margin-top:20px;
}

.upload-box{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}

.stButton>button{
    width:100%;
    background:#2a5298;
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1e3c72;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
<h1>🤖 AI Document Assistant</h1>
<p>Upload PDF Documents and Ask Questions Instantly</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --------------------------------------------------
# PDF EXTRACTION
# --------------------------------------------------

def extract_text_from_pdf(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text

# --------------------------------------------------
# SMART CHUNKING
# --------------------------------------------------

def split_text(text):

    # clean spaces
    text = re.sub(r'\s+', ' ', text)

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if len(current_chunk) + len(sentence) < 700:

            current_chunk += sentence + " "

        else:

            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# --------------------------------------------------
# VECTOR STORE
# --------------------------------------------------

def create_faiss_index(chunks):

    embeddings = model.encode(
        chunks,
        show_progress_bar=False
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    return index

# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

def get_answer(question, chunks, index):

    query_embedding = model.encode(
        [question]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    D, I = index.search(
        query_embedding,
        k=min(5, len(chunks))
    )

    retrieved_answers = []

    for idx in I[0]:

        if idx < len(chunks):

            text = chunks[idx].strip()

            if text not in retrieved_answers:
                retrieved_answers.append(text)

    return retrieved_answers

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📌 Features")

    st.success("PDF Upload")
    st.success("Semantic Search")
    st.success("Sentence Transformers")
    st.success("FAISS Vector Search")
    st.success("Document Question Answering")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="upload-box">',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "📄 Upload PDF Document",
    type=["pdf"]
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# PROCESS DOCUMENT
# --------------------------------------------------

if uploaded_file:

    with st.spinner(
        "Processing document..."
    ):

        text = extract_text_from_pdf(
            uploaded_file
        )

        chunks = split_text(
            text
        )

        index = create_faiss_index(
            chunks
        )

    st.success(
        "✅ Document Indexed Successfully!"
    )

    col1, col2 = st.columns(
        [3,1]
    )

    with col1:

        question = st.text_input(
            "🔍 Ask a Question"
        )

    with col2:

        st.write("")
        st.write("")

        ask_button = st.button(
            "Generate Answer"
        )

    if ask_button:

        if question:

            answers = get_answer(
                question,
                chunks,
                index
            )

            html_output = ""

            for ans in answers:

                html_output += f"""
                <li style="
                    margin-bottom:18px;
                    line-height:1.8;
                    font-size:16px;">
                    {ans}
                </li>
                """

            st.markdown(
                f"""
                <div class="answer-box">
                    <h3>📄 Retrieved Answer</h3>
                    <ul>
                        {html_output}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.warning(
                "Please enter a question."
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown("""
<center>
<h5>
🚀 AI Document Assistant |
Built with Streamlit, FAISS & Sentence Transformers
</h5>
</center>
""",
unsafe_allow_html=True)
