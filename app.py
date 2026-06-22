import streamlit as st
import pdfplumber
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# --------------------------
# CUSTOM CSS
# --------------------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.hero {
    background: linear-gradient(135deg,#1e3c72,#2a5298);
    padding:40px;
    border-radius:20px;
    text-align:center;
    color:white;
    margin-bottom:25px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.15);
}

.hero h1{
    font-size:42px;
    margin-bottom:10px;
}

.hero p{
    font-size:18px;
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

</style>
""", unsafe_allow_html=True)

# --------------------------
# HEADER
# --------------------------
st.markdown("""
<div class="hero">
    <h1>🤖 AI Document Assistant</h1>
    <p>Upload PDF Documents and Ask Questions Instantly</p>
</div>
""", unsafe_allow_html=True)

# --------------------------
# LOAD MODEL
# --------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --------------------------
# PDF READING
# --------------------------
def extract_text_from_pdf(pdf_file):
    text = ""

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text

# --------------------------
# CHUNKING
# --------------------------
def split_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks

# --------------------------
# VECTOR STORE
# --------------------------
def create_faiss_index(chunks):

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    return index, embeddings

# --------------------------
# RETRIEVAL
# --------------------------
def get_answer(question, chunks, index):

    query_embedding = model.encode([question])

    D, I = index.search(
        np.array(query_embedding).astype("float32"),
        k=3
    )

    retrieved_chunks = []

    for idx in I[0]:
        retrieved_chunks.append(chunks[idx])

    answer = "\n\n".join(retrieved_chunks)

    return answer

# --------------------------
# SIDEBAR
# --------------------------
with st.sidebar:
    st.header("📌 Features")

    st.success("PDF Upload")
    st.success("Semantic Search")
    st.success("FAISS Vector Database")
    st.success("Sentence Transformers")

# --------------------------
# FILE UPLOAD
# --------------------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📄 Upload Your PDF Document",
    type=["pdf"]
)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# PROCESS PDF
# --------------------------
if uploaded_file:

    with st.spinner("Processing document..."):

        text = extract_text_from_pdf(uploaded_file)

        chunks = split_text(text)

        index, embeddings = create_faiss_index(chunks)

    st.success("✅ Document Indexed Successfully!")

    col1, col2 = st.columns([2,1])

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

            answer = get_answer(
                question,
                chunks,
                index
            )

            st.markdown(
                f"""
                <div class="answer-box">
                <h3>📄 Retrieved Answer</h3>
                <p>{answer}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.warning("Please enter a question.")

# --------------------------
# FOOTER
# --------------------------
st.markdown("---")

st.markdown(
"""
<center>
<h5>🚀 AI Document Assistant | Built with Streamlit, FAISS & Sentence Transformers</h5>
</center>
""",
unsafe_allow_html=True
)