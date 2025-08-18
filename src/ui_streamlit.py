import streamlit as st
import time
from ocr_parser import answer_query_rag, answer_query_ft

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