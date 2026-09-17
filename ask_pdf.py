import os
import pymupdf
import numpy as np
import re
from dotenv import load_dotenv
from google import genai


# -----------------------------
# 1. Load API key
# -----------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY was not found.")
    print("Please check your .env file.")
    exit()


# -----------------------------
# 2. Connect to Gemini
# -----------------------------

client = genai.Client(api_key=api_key)

chat = client.chats.create(
    model="gemini-3.6-flash"
)


# -----------------------------
# 3. Read PDF
# -----------------------------

pdf_path = input("Enter PDF filename: ")

try:
    pdf = pymupdf.open(pdf_path)

except Exception:
    print("Error: Could not open the PDF.")
    print("Please check the filename and try again.")
    exit()


print(f"\nPDF: {pdf_path}")
print(f"Pages: {len(pdf)}")


# -----------------------------
# 4. Split PDF into chunks
# -----------------------------

def create_chunks(pdf, chunk_size=500, overlap=100):

    chunks = []

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text().strip()

        if not text:
            continue

        words = re.findall(r'\S+', text)

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


chunks = create_chunks(pdf)


if not chunks:

    print("Error: No readable text was found in this PDF.")
    print("This may be a scanned/image-only PDF.")
    exit()


print(f"PDF loaded and split into {len(chunks)} chunks")


# -----------------------------
# 5. Create embeddings
# -----------------------------

def create_embeddings(chunks):

    for chunk in chunks:

        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk["text"]
        )

        chunk["embedding"] = result.embeddings[0].values

    return chunks


chunks = create_embeddings(chunks)

print("Embeddings created successfully!")


# -----------------------------
# 6. Semantic retrieval
# -----------------------------

def retrieve_chunks(question, chunks, top_k=3, threshold=0.45):

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


    # Keep only one chunk from each page

    unique_chunks = []

    seen_pages = set()

    for chunk in scored_chunks:

        if chunk["page"] not in seen_pages:

            unique_chunks.append(chunk)

            seen_pages.add(chunk["page"])


        if len(unique_chunks) == top_k:

            break


    return unique_chunks


# -----------------------------
# 7. Ask questions
# -----------------------------

print("\nAsk questions about the PDF.")
print("Type 'exit' to quit.\n")

conversation_history = []

while True:

    question = input("You: ")


    if question.lower() == "exit":

        print("Goodbye!")

        break


    # Retrieve relevant chunks

    relevant_chunks = retrieve_chunks(
        question,
        chunks
    )


    # No relevant information

    if not relevant_chunks:

        print("\nAssistant:")

        print(
            "I couldn't find that information in the PDF.\n"
        )

        continue


    # -----------------------------
    # Show retrieved chunks
    # -----------------------------

    print("\nRetrieved chunks:")


    for chunk in relevant_chunks:

        print(
            f"\n[Page {chunk['page']}] "
            f"(similarity: {chunk['score']:.3f})"
        )

        print(
            chunk["text"][:300]
        )


    # -----------------------------
    # Build context
    # -----------------------------

    context = ""


    for chunk in relevant_chunks:

        context += (
            f"\n[Page {chunk['page']}]\n"
        )

        context += chunk["text"]

        context += "\n"


    # -----------------------------
# Create conversation history
# -----------------------------

history = ""

for item in conversation_history:

    history += f"""
Previous question: {item["question"]}
Previous answer: {item["answer"]}
"""


# -----------------------------
# Create prompt
# -----------------------------

prompt = f"""
You are a helpful PDF assistant.

Answer the user's question using ONLY the information
provided in the relevant sections of the PDF.

Instructions:
- Give a clear and concise answer.
- Use bullet points when useful.
- Do not add information that is not present in the PDF.
- Use the conversation history to understand follow-up questions.
- If the answer cannot be found in the PDF, say:
  "I couldn't find that information in the PDF."

Previous conversation:

{history}

Relevant PDF sections:

{context}

User question:

{question}
"""


    # -----------------------------
    # Generate answer
    # -----------------------------

try:

    response = chat.send_message(prompt)

    answer = response.text

    print("\nAssistant:")
    print(answer)

    conversation_history.append({
        "question": question,
        "answer": answer
    })


except Exception as e:

    print("\nAssistant:")

    print(
        "Sorry, I couldn't generate an answer right now."
    )

    print(
        "Please try again in a moment."
    )

    print(
        f"\nAPI error: {e}"
    )


# -----------------------------
# Show sources
# -----------------------------

print("\nSources:")

for chunk in relevant_chunks:

    print(
        f"- Page {chunk['page']} "
        f"(similarity: {chunk['score']:.3f})"
    )

print()