from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
from pdf2image import convert_from_path
import pytesseract
import re
import faiss
import pickle
from rank_bm25 import BM25Okapi
import argparse
import torch
from rich.console import Console
from rich.table import Table
import time
import streamlit as st
try:
    import gradio as gr
except ImportError:
    gr = None

# Configure Tesseract path if needed (Windows users typically need this)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def pdf_to_text(pdf_path, output_txt_path=None, dpi=300, lang='eng'):
    """
    Convert a scanned PDF to plain text using OCR.
    
    Args:
        pdf_path (str): Path to the input PDF file.
        output_txt_path (str): Path to save extracted text (optional).
        dpi (int): Resolution for converting PDF to image (default: 300).
        lang (str): Language for OCR (default: 'eng').
    
    Returns:
        str: Extracted text from the PDF.
    """
    # Convert PDF pages to images
    print(f"Converting PDF to images...")
    pages = convert_from_path(pdf_path, dpi=dpi)

    text = ""
    for i, page in enumerate(pages):
        print(f"OCR on page {i + 1}...")
        text += pytesseract.image_to_string(page, lang=lang)
        text += "\n\n"  # Separate pages

    if output_txt_path:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text saved to {output_txt_path}")
    
    return text

def clean_ocr_text(text):
    """
    Clean OCR text by removing headers, footers, page numbers, and fixing spaces.
    """

    # 1. Remove page numbers (standalone numbers or "Page X of Y")
    text = re.sub(r'\n?\s*\d+\s*\n', '\n', text)
    text = re.sub(r'Page\s+\d+(\s+of\s+\d+)?', '', text, flags=re.IGNORECASE)

    # 2. Remove common headers/footers (Annual Report, Company name, Statutory Reports)
    headers_footers = [
        r'Annual Report\s*\d{4}-\d{2}',
        r'Indian Railway Finance Corporation Ltd',
        r'Statutory Reports\s*Corporate Overview\s*Financial Statements',
        r'Corporate Overview',
    ]
    for pattern in headers_footers:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 3. Fix broken words/numbers (F Y 2 3 → FY23)
    text = re.sub(r'F\s*Y\s*(\d{2})\s*(\d{2})', r'FY\1-\2', text)

    # 4. Remove multiple newlines and extra spaces
    text = re.sub(r'\n{2,}', '\n\n', text)  # collapse multiple newlines
    text = re.sub(r'\s{2,}', ' ', text)     # collapse multiple spaces

    # 5. Strip leading/trailing whitespace
    text = text.strip()

    return text

def split_into_chunks(text, chunk_size):
    """
    Split text into chunks of approximately chunk_size tokens (words).
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def add_metadata_to_chunks(chunks, chunk_size):
    """
    Add metadata to each chunk with unique ID and size.
    """
    metadata_chunks = []
    for idx, chunk in enumerate(chunks):
        metadata = f"CHUNK_ID: {idx}, SIZE: {chunk_size}\n"
        metadata_chunks.append(metadata + chunk)
    return metadata_chunks

def simple_tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def build_and_save_faiss(embeddings: np.ndarray, dim: int, index_path: str):
    index = faiss.IndexFlatIP(dim)
    # Normalize embeddings for cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normed = embeddings / norms
    index.add(normed.astype(np.float32))
    faiss.write_index(index, index_path)
    return index

def build_and_save_bm25(docs_tokens, bm25_path: str):
    bm25 = BM25Okapi(docs_tokens)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    return bm25

def load_faiss(index_path):
    return faiss.read_index(index_path)

def search_dense(query: str, model: SentenceTransformer, index, chunks_texts, top_k=5):
    q_emb = model.encode([query])
    # Normalize for cosine/IP
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    D, I = index.search(q_emb.astype(np.float32), top_k)
    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        results.append({"rank": rank+1, "chunk_id": int(idx), "score": float(score), "text": chunks_texts[idx]})
    return results

def load_bm25(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def search_sparse(query: str, bm25, chunks_texts, top_k=5):
    q_tokens = simple_tokenize(query)
    scores = bm25.get_scores(q_tokens)
    top_idx = np.argsort(-scores)[:top_k]
    return [{"rank": i+1, "chunk_id": int(idx), "score": float(scores[idx]), "text": chunks_texts[idx]} for i, idx in enumerate(top_idx)]

def preprocess_query(query):
    stopwords = {"the", "is", "at", "which", "on", "and", "a", "an", "of", "for", "in", "to", "with"}
    query = query.lower()
    query = re.sub(r'[^a-z0-9 ]+', '', query)
    tokens = query.split()
    filtered_tokens = [t for t in tokens if t not in stopwords]
    return " ".join(filtered_tokens)

def validate_query(query):
    """
    Validate the query to check for harmful or irrelevant content.
    Raises ValueError if invalid.
    """
    if not query or not query.strip():
        raise ValueError("Query is empty. Please provide a valid query.")
    offensive_words = {"offensiveword1", "offensiveword2", "offensiveword3"}  # Replace with actual offensive words
    lowered = query.lower()
    for word in offensive_words:
        if word in lowered:
            raise ValueError("Query contains inappropriate content. Please modify your query.")
    # Additional checks can be added here

def hybrid_retrieve(query, model, faiss_index, bm25, chunks_texts, top_k=5, alpha=0.5):
    # Preprocess query
    proc_query = preprocess_query(query)

    # Encode query for dense retrieval and normalize
    q_emb = model.encode([proc_query])
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

    # Search FAISS for top_k results
    D, I = faiss_index.search(q_emb.astype(np.float32), top_k)
    dense_results = [(int(idx), float(score)) for idx, score in zip(I[0], D[0])]

    # Search BM25 for top_k results
    q_tokens = simple_tokenize(proc_query)
    bm25_scores = bm25.get_scores(q_tokens)
    top_bm25_idx = np.argsort(-bm25_scores)[:top_k]
    bm25_results = [(int(idx), bm25_scores[idx]) for idx in top_bm25_idx]

    # Normalize BM25 scores by max score to [0,1]
    max_bm25_score = max([score for _, score in bm25_results]) if bm25_results else 1.0
    normalized_bm25_scores = {idx: score / max_bm25_score if max_bm25_score > 0 else 0.0 for idx, score in bm25_results}

    # Combine results by weighted score fusion
    combined_scores = {}
    # Add dense scores weighted by alpha
    for idx, score in dense_results:
        combined_scores[idx] = alpha * score
    # Add BM25 scores weighted by (1-alpha)
    for idx, norm_score in normalized_bm25_scores.items():
        combined_scores[idx] = combined_scores.get(idx, 0.0) + (1 - alpha) * norm_score

    # Sort combined results by combined_score descending
    sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Prepare final results list
    results = []
    for rank, (chunk_id, combined_score) in enumerate(sorted_combined, start=1):
        results.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "combined_score": combined_score,
            "text": chunks_texts[chunk_id]
        })
    return results

def rerank_with_cross_encoder(query, results, cross_encoder_model, top_k=None):
    pairs = [(query, r["text"]) for r in results]
    scores = cross_encoder_model.predict(pairs)
    for r, score in zip(results, scores):
        r["rerank_score"] = float(score)
    sorted_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    if top_k is not None:
        sorted_results = sorted_results[:top_k]
    return sorted_results

def filter_output(answer):
    """
    Filter the generated answer for placeholder phrases.
    """
    lowered = answer.lower()
    placeholders = ["i don't know", "not available"]
    for phrase in placeholders:
        if phrase in lowered:
            return "Answer uncertain. Please verify."
    return answer


# --- Helper functions for answering and UI ---
def summarize_with_gpt2(prompt, tokenizer, model_gpt2, max_new_tokens=60):
    # Build inputs leaving room for generation
    max_context_length = model_gpt2.config.n_positions
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_context_length - max_new_tokens
    )
    available_space = max_context_length - inputs["input_ids"].shape[1]
    if available_space < max_new_tokens:
        max_new_tokens = max(10, available_space)
    outputs = model_gpt2.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split("Answer:")[-1].strip()
    sentences = answer.split(". ")
    short_answer = ". ".join(sentences[:2]) + "."
    return filter_output(short_answer)

def compute_confidence_from_rerank(reranked_results):
    if not reranked_results:
        return 0.0
    scores = [r.get("rerank_score", 0.0) for r in reranked_results]
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-6:
        return 0.5
    # Normalize top score to [0,1]
    top = scores[0]
    return float((top - mn) / (mx - mn))

def answer_query_rag(query, retrieval_mode, top_k, chunks_texts, faiss_index, bm25, sbert_model, cross_encoder_model, tokenizer, model_gpt2):
    t0 = time.time()
    # Retrieve
    if retrieval_mode == "dense":
        base_results = search_dense(query, sbert_model, faiss_index, chunks_texts, top_k=top_k)
    elif retrieval_mode == "sparse":
        base_results = search_sparse(query, bm25, chunks_texts, top_k=top_k)
    else:
        base_results = hybrid_retrieve(query, sbert_model, faiss_index, bm25, chunks_texts, top_k=top_k, alpha=0.5)
    # Rerank
    reranked = rerank_with_cross_encoder(query, base_results, cross_encoder_model, top_k=top_k)
    # Build prompt with concise instruction
    context = "\n\n".join([r["text"] for r in reranked])
    prompt = (
        "Summarize the answer to the question in a short and concise way (max 3 sentences) "
        "based on the following context:\n\n"
        f"{context}\n\nQuestion: {query}\nAnswer:"
    )
    # Generate
    answer = summarize_with_gpt2(prompt, tokenizer, model_gpt2, max_new_tokens=60)
    latency = time.time() - t0
    confidence = compute_confidence_from_rerank(reranked)
    method = f"{retrieval_mode} + cross-encoder re-rank + GPT-2"
    return {"answer": answer, "confidence": confidence, "method": method, "latency": latency, "retrieved": reranked}

def answer_query_ft(query, tokenizer, model_gpt2):
    t0 = time.time()
    prompt = (
        "Answer the question briefly and factually in at most 2 sentences.\n\n"
        f"Question: {query}\nAnswer:"
    )
    answer = summarize_with_gpt2(prompt, tokenizer, model_gpt2, max_new_tokens=60)
    latency = time.time() - t0
    return {"answer": answer, "confidence": 0.5, "method": "Fine-Tuned (simulated: GPT-2 baseline)", "latency": latency, "retrieved": []}

def launch_gradio(chunks_texts, faiss_index, bm25, sbert_model, cross_encoder_model, tokenizer, model_gpt2):
    if gr is None:
        raise RuntimeError("Gradio is not installed. Please `pip install gradio` and try again.")
    def run(query, pipeline, retrieval_mode, chunk_size, top_k):
        try:
            validate_query(query)
        except ValueError as e:
            return str(e), 0.0, "validation", 0.0
        if pipeline == "RAG":
            res = answer_query_rag(query, retrieval_mode, int(top_k), chunks_texts, faiss_index, bm25, sbert_model, cross_encoder_model, tokenizer, model_gpt2)
        else:
            res = answer_query_ft(query, tokenizer, model_gpt2)
        return res["answer"], float(res["confidence"]), res["method"], float(res["latency"])
    with gr.Blocks() as demo:
        gr.Markdown("## 📚 RAG vs FT — QA Demo")
        with gr.Row():
            query = gr.Textbox(label="Your query", placeholder="e.g., net profit after tax")
        with gr.Row():
            pipeline = gr.Radio(choices=["RAG", "FT"], value="RAG", label="Pipeline")
            retrieval_mode = gr.Radio(choices=["dense", "sparse", "hybrid"], value="hybrid", label="Retrieval (RAG)")
            chunk_size = gr.Dropdown(choices=[100, 400], value=400, label="Chunk size (loaded at start)")
            top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K")
        btn = gr.Button("Run")
        answer = gr.Textbox(label="Answer")
        confidence = gr.Number(label="Confidence (0-1)")
        method = gr.Textbox(label="Method used")
        latency = gr.Number(label="Response time (s)")
        btn.click(run, inputs=[query, pipeline, retrieval_mode, chunk_size, top_k], outputs=[answer, confidence, method, latency])
    demo.launch()


def streamlit_ui(chunks_texts, faiss_index, bm25, sbert_model, cross_encoder_model, tokenizer, model_gpt2):
    st.title("📚 RAG vs FT — QA Demo")
    
    query = st.text_input("Enter your query:")
    pipeline = st.radio("Choose pipeline:", ["RAG", "FT"])
    retrieval_mode = st.selectbox("Retrieval mode (for RAG):", ["dense", "sparse", "hybrid"])
    chunk_size = st.selectbox("Chunk size:", [100, 400])
    top_k = st.slider("Top-K:", 1, 10, 5)
    
    if st.button("Run"):
        if not query.strip():
            st.warning("Please enter a valid query.")
            return
        
        t0 = time.time()
        if pipeline == "RAG":
            res = answer_query_rag(query, retrieval_mode, top_k, chunks_texts, faiss_index, bm25, sbert_model, cross_encoder_model, tokenizer, model_gpt2)
        else:
            res = answer_query_ft(query, tokenizer, model_gpt2)
        
        latency = time.time() - t0
        st.subheader("Answer")
        st.write(res["answer"])
        st.metric("Confidence", f"{res['confidence']:.2f}")
        st.metric("Method", res["method"])
        st.metric("Response Time", f"{latency:.2f} s")

if __name__ == "__main__":
    pdf_file = "Annual_Report_2023_24.pdf"  # Replace with your PDF file
    output_file = "output.txt"
    extracted_text = pdf_to_text(pdf_file, output_txt_path=output_file)
    print("OCR Extraction Completed!")

    # Clean the extracted text
    cleaned_text = clean_ocr_text(extracted_text)
    with open("cleaned_output.txt", "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    
    print("Text cleaned and saved to cleaned_output.txt")

    # Split into chunks
    chunks_100 = split_into_chunks(cleaned_text, 100)
    chunks_400 = split_into_chunks(cleaned_text, 400)
    chunks_100_with_metadata = add_metadata_to_chunks(chunks_100, 100)
    chunks_400_with_metadata = add_metadata_to_chunks(chunks_400, 400)
    with open("chunks_100.txt", "w", encoding="utf-8") as f:
        f.write("\n\n---CHUNK---\n\n".join(chunks_100_with_metadata))
    with open("chunks_400.txt", "w", encoding="utf-8") as f:
        f.write("\n\n---CHUNK---\n\n".join(chunks_400_with_metadata))
    print("Chunks created and saved to chunks_100.txt and chunks_400.txt")
    parser = argparse.ArgumentParser(description="OCR text retrieval CLI")
    parser.add_argument("--query", type=str, required=True, help="Query string for retrieval")
    parser.add_argument("--mode", type=str, choices=["dense", "sparse", "hybrid"], default="hybrid", help="Retrieval mode")
    parser.add_argument("--size", type=int, choices=[100, 400], default=400, help="Chunk size to use (100 or 400)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--ui", type=str, choices=["none", "gradio"], default="none", help="Launch a simple UI (gradio)")
    parser.add_argument("--pipeline", type=str, choices=["RAG", "FT"], default="RAG", help="Choose between Retrieval-Augmented (RAG) or Fine-Tuned (FT) mode")
    args = parser.parse_args()

    query = args.query
    try:
        validate_query(query)
    except ValueError as e:
        print(f"Query validation error: {e}")
        exit(1)

    # Load model once
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embedding_100 = model.encode(chunks_100)
    embedding_400 = model.encode(chunks_400_with_metadata)
    np.save("embeddings_100.npy", embedding_100)
    np.save("embeddings_400.npy", embedding_400)
    print("Embeddings created and saved to embeddings_100.npy and embeddings_400.npy")
    # Build and save FAISS index
    dim = embedding_100.shape[1]
    print("Building FAISS index for 100 chunks...")
    build_and_save_faiss(embedding_100, dim, "faiss_100.index")
    print(" Saved FAISS index to faiss_100.index")
    print("Building FAISS index for 400 chunks...")
    build_and_save_faiss(embedding_400, dim, "faiss_400.index")
    print("Saved FAISS index to faiss_400.index")

    # Build and save BM25 index
    print("Tokenizing chunks for BM25...")
    docs_tokens_100 = [simple_tokenize(chunk) for chunk in chunks_100]
    docs_tokens_400 = [simple_tokenize(chunk) for chunk in chunks_400]

    print("Building BM25 index for 100 chunks...")
    build_and_save_bm25(docs_tokens_100, "bm25_100.pkl")
    print("Saved BM25 index to bm25_100.pkl")

    print("Building BM25 index for 400 chunks...")
    build_and_save_bm25(docs_tokens_400, "bm25_400.pkl")
    print("Saved BM25 index to bm25_400.pkl")

    # Save chunk text and ID mappings for retrieval reconstruction
    with open("chunks_100.pkl", "wb") as f:
        pickle.dump({"ids": list(range(len(chunks_100))), "texts": chunks_100, "sizes": 100}, f)

    with open("chunks_400.pkl", "wb") as f:
        pickle.dump({"ids": list(range(len(chunks_400))), "texts": chunks_400, "sizes": 400}, f)
    
    print("Chunk texts and IDs saved to chunks_100.pkl and chunks_400.pkl")

    # Load chunks and indexes based on chunk size
    if args.size == 100:
        chunks_pkl = "chunks_100.pkl"
        faiss_index_path = "faiss_100.index"
        bm25_pkl = "bm25_100.pkl"
    else:
        chunks_pkl = "chunks_400.pkl"
        faiss_index_path = "faiss_400.index"
        bm25_pkl = "bm25_400.pkl"

    with open(chunks_pkl, "rb") as f:
        data = pickle.load(f)
    chunks_texts = data["texts"]

    faiss_index = load_faiss(faiss_index_path)
    bm25 = load_bm25(bm25_pkl)

    top_k = args.top_k
    mode = args.mode

    # Load cross-encoder and GPT-2 once
    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if args.ui == "gradio":
        # Launch interactive UI
        launch_gradio(chunks_texts, faiss_index, bm25, model, cross_encoder_model, tokenizer, model_gpt2)
        exit(0)

    # CLI single-run path
    pipeline_choice = args.pipeline
    if pipeline_choice == "RAG":
        res = answer_query_rag(query, mode, top_k, chunks_texts, faiss_index, bm25, model, cross_encoder_model, tokenizer, model_gpt2)
    else:
        res = answer_query_ft(query, tokenizer, model_gpt2)

    print("\n=== Result ===")
    print(f"Answer: {res['answer']}")
    print(f"Confidence: {res['confidence']:.3f}")
    print(f"Method: {res['method']}")
    print(f"Response time: {res['latency']:.2f}s")

    console = Console()
    table = Table(title="QA Result")
    table.add_column("Answer", style="cyan")
    table.add_column("Confidence", style="green")
    table.add_column("Method", style="magenta")
    table.add_column("Latency (s)", style="yellow")
    table.add_row(res["answer"], f"{res['confidence']:.2f}", res["method"], f"{res['latency']:.2f}")
    console.print(table)
