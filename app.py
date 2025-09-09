
import streamlit as st
import torch
import fitz  # PyMuPDF
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# Load models (lightweight + CPU safe)
# -------------------------------
@st.cache_resource
def load_models():
    # Embedding model
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Small LLM (safe for CPU, not gated)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    llm = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b",
        torch_dtype=torch.float32
    ).to("cpu")

    # ASR pipeline (Granite speech)
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="ibm-granite/granite-speech-3.3-8b"
    )

    return embedder, tokenizer, llm, asr_pipe


embedder, tokenizer, llm, asr_pipe = load_models()


# -------------------------------
# PDF Text Extraction
# -------------------------------
def extract_text_from_pdfs(files):
    all_text = ""
    for file in files:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            all_text += page.get_text("text") + "\n"
    return all_text


# -------------------------------
# Ask LLM
# -------------------------------
def ask_llm(question, context, tokenizer, llm):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = llm.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="StudyMate AI", layout="wide")
st.title("📘 StudyMate AI – PDF + Speech Q&A")

tab1, tab2 = st.tabs(["📄 Chat with PDFs", "🎤 Speech to Text"])

# --- PDF Q&A ---
with tab1:
    pdfs = st.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)
    query = st.text_input("Ask a question about the documents")

    if pdfs and query:
        text = extract_text_from_pdfs(pdfs)
        answer = ask_llm(query, text, tokenizer, llm)
        st.markdown("### 💡 Answer")
        st.write(answer)

# --- Speech Recognition ---
with tab2:
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
    if audio_file:
        st.audio(audio_file)
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.read())

        result = asr_pipe("temp_audio.wav")
        st.markdown("### 📝 Transcription")
        st.write(result["text"])
