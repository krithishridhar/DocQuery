import os
import re

import numpy as np
import pymupdf
import streamlit as st
from dotenv import load_dotenv
from google import genai


# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY was not found.")
    st.stop()

client = genai.Client(api_key=api_key)


# Session state
if "answer" not in st.session_state:
    st.session_state.answer = ""


# Page setup
st.title("📄 DocQuery AI")
st.write("Ask questions about your PDF documents.")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


def create_chunks(pdf, chunk_size=500, overlap=100):
    """Split PDF text into overlapping word-based chunks."""

    chunks = []

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text().strip()

        if not text:
            continue

        words = re.findall(r"\S+", text)
        start = 0

        while start < len(words):
            chunk_words = words[start:start + chunk_size]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "page": page_number,
                "text": chunk_text
            })

            start += chunk_size - overlap

    return chunks


@st.cache_data
def create_embeddings(chunks):
    """Create embeddings for all PDF chunks."""

    embedded_chunks = []

    for chunk in chunks:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk["text"]
        )

        embedded_chunks.append({
            "page": chunk["page"],
            "text": chunk["text"],
            "embedding": result.embeddings[0].values
        })

    return embedded_chunks


def retrieve_chunks(question, chunks, top_k=3, threshold=0.45):
    """Retrieve the most relevant PDF chunks using cosine similarity."""

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )

    question_embedding = np.array(
        result.embeddings[0].values
    )

    scored_chunks = []

    for chunk in chunks:
        chunk_embedding = np.array(
            chunk["embedding"]
        )

        similarity = np.dot(
            question_embedding,
            chunk_embedding
        ) / (
            np.linalg.norm(question_embedding)
            * np.linalg.norm(chunk_embedding)
        )

        if similarity >= threshold:
            scored_chunks.append({
                "page": chunk["page"],
                "text": chunk["text"],
                "score": similarity
            })

    scored_chunks.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    unique_chunks = []
    seen_pages = set()

    for chunk in scored_chunks:
        if chunk["page"] not in seen_pages:
            unique_chunks.append(chunk)
            seen_pages.add(chunk["page"])

        if len(unique_chunks) == top_k:
            break

    return unique_chunks


if uploaded_file:

    st.success("PDF uploaded successfully!")

    pdf_bytes = uploaded_file.getvalue()

    pdf = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    st.write(f"Pages: {len(pdf)}")

    text = ""

    for page in pdf:
        text += page.get_text()

    st.write(
        f"Characters extracted: {len(text)}"
    )

    chunks = create_chunks(pdf)

    if not chunks:
        st.error(
            "No readable text was found in this PDF. "
            "It may be a scanned or image-only PDF."
        )
        st.stop()

    st.write(
        f"Chunks created: {len(chunks)}"
    )

    chunks = create_embeddings(chunks)

    st.success(
        "PDF is ready for questions! ✅"
    )

    with st.expander("📄 View extracted PDF text"):
        st.text_area(
            "Extracted text",
            text,
            height=300
        )

    question = st.text_input(
        "Ask a question about your PDF:",
        placeholder="e.g. What are the two types of inductive bias?",
        key="question"
    )

    if st.button("Clear Answer"):
        st.session_state.answer = ""
        st.rerun()

    if question:

        st.markdown(
            f"**You:** {question}"
        )

        relevant_chunks = retrieve_chunks(
            question,
            chunks
        )

        if not relevant_chunks:

            st.warning(
                "I couldn't find that information in the PDF."
            )

        else:

            st.subheader("📚 Relevant Sources")

            for chunk in relevant_chunks:
                st.write(
                    f"Page {chunk['page']} "
                    f"(similarity: {chunk['score']:.3f})"
                )

                st.text(
                    chunk["text"][:500]
                )

            context = ""

            for chunk in relevant_chunks:
                context += (
                    f"\n[Page {chunk['page']}]\n"
                )
                context += chunk["text"]
                context += "\n"

            prompt = f"""
You are a helpful PDF assistant.

Answer the user's question using ONLY the information
provided in the relevant sections of the PDF.

Instructions:
- Give a clear and concise answer.
- Use bullet points when useful.
- Do not add information that is not present in the PDF.
- If the answer cannot be found in the provided sections, say:
  "I couldn't find that information in the PDF."

Relevant PDF sections:

{context}

User question:

{question}
"""

            if st.button("Generate Answer"):

                with st.spinner("Generating answer..."):

                    try:

                        response = client.models.generate_content(
                            model="gemini-3.6-flash",
                            contents=prompt
                        )

                        st.session_state.answer = response.text

                    except Exception:

                        st.error(
                            "Sorry, I couldn't generate an answer "
                            "right now. Please try again later."
                        )

            if st.session_state.answer:

                st.subheader("🤖 Answer")

                st.info(
                    st.session_state.answer
                )