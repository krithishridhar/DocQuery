📄 DocQuery AI



"AI-powered PDF question answering using Retrieval-Augmented Generation (RAG)"


Live Demo

🚀 [Try DocQuery AI](https://docqueryai2.streamlit.app/)

![DocQuery AI Screenshot](screenshot.png)

DocQuery AI is a Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its contents.
The application uses semantic embeddings to retrieve relevant sections of the document and Gemini to generate answers grounded in the retrieved content.



✨ Features


* Upload and process PDF documents

* Extract text using PyMuPDF

* Split documents into overlapping chunks

* Generate semantic embeddings using Gemini

* Retrieve relevant sections using cosine similarity

* Display source pages and similarity scores

* Generate PDF-grounded answers using Gemini

* Streamlit-based web interface

* Cached embeddings for improved performance



🧠 How It Works



PDF Upload

&#x20;   ↓

Text Extraction

&#x20;   ↓

Text Chunking

&#x20;   ↓

Gemini Embeddings

&#x20;   ↓

User Question

&#x20;   ↓

Question Embedding

&#x20;   ↓

Cosine Similarity Retrieval

&#x20;   ↓

Relevant PDF Chunks

&#x20;   ↓

Gemini

&#x20;   ↓

Grounded Answer



The application follows a basic "Retrieval-Augmented Generation (RAG)" pipeline:



1. The uploaded PDF is converted into text.

2. The text is divided into overlapping chunks.

3\. Each chunk is converted into an embedding using `gemini-embedding-001`.

4. The user's question is also converted into an embedding.

5. Cosine similarity is used to identify the most relevant chunks.

6. The retrieved content is provided to Gemini as context.

7. Gemini generates an answer using the retrieved PDF content.



🛠️ Tech Stack



* Python

* Streamlit

* Google Gemini API

* Gemini Embeddings

* PyMuPDF

* NumPy

* python-dotenv



🎯 Project Objective



This project was built to gain practical experience with \*\*Generative AI, embeddings, semantic search, and Retrieval-Augmented Generation (RAG)\*\* by developing an end-to-end document question-answering application.



