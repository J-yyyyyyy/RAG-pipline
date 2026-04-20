# ===== Streamlit RAG Mental Health Chatbot =====
# Enhanced Version: Hybrid Search + Reranking + Query Rewriting
# Local Models Only (No OpenAI API)

import streamlit as st
import os
import numpy as np
import faiss
from typing import List, Tuple

# ============================
# Dependencies:
# pip install sentence-transformers langchain-text-splitters rank-bm25 ollama
# ============================

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import ollama

# ============================
# Configuration
# ============================
# Local Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast embedding model
CHAT_MODEL = "llama3.2:1b"               # Or "mistral", "qwen2.5", "phi"

# Chunking Configuration
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# Retrieval Configuration
VECTOR_TOP_K = 5
BM25_TOP_K = 5
FUSION_TOP_K = 5

# Reranking Configuration
RERANK_TOP_K = 3

st.set_page_config(page_title="Mental Health RAG Chatbot (Local)")

st.title("🧠 Mental Health Chatbot (RAG)")
st.write("⚠️ This is not medical advice. Please consult professionals for serious concerns.")

# ============================
# 0. Model Loading (Local)
# ============================
@st.cache_resource
def load_models():
    """
    Load local models.
    - Embedding: sentence-transformers
    - Chat: Ollama
    - Reranker: CrossEncoder
    """
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder('BAAI/bge-reranker-large')
    return embed_model, reranker

# ============================
# 1. Text Chunking
# ============================
def chunk_documents(raw_texts: List[str]) -> List[str]:
    """
    Split raw documents into overlapping semantic chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n", "\n", ". ", "! ", "? ", ", ", " "
        ],
        length_function=len
    )
    chunks = text_splitter.split_text("\n".join(raw_texts))
    return chunks

def get_query_embedding(embed_model, query: str) -> np.ndarray:
    """Generate query embedding using local model."""
    embedding = embed_model.encode(query)
    return embedding

def get_document_embeddings(embed_model, texts: List[str]) -> np.ndarray:
    """Batch get embeddings using local model."""
    embeddings = embed_model.encode(texts)
    return embeddings

# ============================
# 2. Hybrid Search
# ============================
def reciprocal_rank_fusion(
    vector_results: List[Tuple[str, float]], 
    bm25_results: List[Tuple[str, float]], 
    k: int = 60
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion for merging retrieval results."""
    rrf_scores = {}
    
    for rank, (doc, _) in enumerate(vector_results):
        score = 1.0 / (k + rank + 1)
        rrf_scores[doc] = rrf_scores.get(doc, 0) + score
    
    for rank, (doc, _) in enumerate(bm25_results):
        score = 1.0 / (k + rank + 1)
        rrf_scores[doc] = rrf_scores.get(doc, 0) + score
    
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

def build_index(chunks: List[str], embed_model) -> Tuple[faiss.Index, np.ndarray, BM25Okapi, List[str]]:
    """Build vector index and BM25 index."""
    doc_embeddings = get_document_embeddings(embed_model, chunks)
    dim = doc_embeddings.shape[1]
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    doc_embeddings_normalized = doc_embeddings / norms
    
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings_normalized.astype('float32'))
    
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    
    return index, doc_embeddings, bm25_index, chunks

def vector_search(
    query: str, index: faiss.Index, doc_embeddings: np.ndarray,
    chunks: List[str], embed_model, top_k: int = VECTOR_TOP_K
) -> List[Tuple[str, float]]:
    """Vector search (semantic retrieval)."""
    q_emb = get_query_embedding(embed_model, query)
    q_emb = q_emb / np.linalg.norm(q_emb)  # Normalize
    
    scores, indices = index.search(q_emb.reshape(1, -1).astype('float32'), top_k)
    
    results = [(chunks[idx], float(scores[0][i])) 
               for i, idx in enumerate(indices[0]) if idx < len(chunks)]
    return results

def bm25_search(
    query: str, bm25_index: BM25Okapi, chunks: List[str], top_k: int = BM25_TOP_K
) -> List[Tuple[str, float]]:
    """BM25 keyword search."""
    query_tokens = query.split()
    scores = bm25_index.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = [(chunks[idx], float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    return results

def hybrid_search(
    query: str, index: faiss.Index, doc_embeddings: np.ndarray,
    bm25_index: BM25Okapi, chunks: List[str], embed_model, top_k: int = FUSION_TOP_K
) -> List[str]:
    """Hybrid search combining vector and BM25."""
    vector_results = vector_search(query, index, doc_embeddings, chunks, embed_model, VECTOR_TOP_K)
    bm25_results = bm25_search(query, bm25_index, chunks, BM25_TOP_K)
    
    fused_results = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    return [doc for doc, _ in fused_results[:top_k]]

# ============================
# 3. Reranking
# ============================
@st.cache_resource
def get_reranker():
    """Get and cache reranker model using CrossEncoder."""
    return CrossEncoder('BAAI/bge-reranker-large')

def rerank(query: str, chunks: List[str], reranker, top_n: int = RERANK_TOP_K) -> List[str]:
    """Rerank documents using CrossEncoder."""
    if not chunks:
        return []
    
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    
    if isinstance(scores, float):
        scores = [scores]
    
    indexed_scores = list(enumerate(scores))
    sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    
    return [chunks[idx] for idx, _ in sorted_scores[:top_n]]

# ============================
# 4. Query Rewriting (Local)
# ============================
def rewrite_query(raw_query: str) -> str:
    """
    Rewrite conversational query into retrieval-friendly format using Ollama.
    """
    system_prompt = """You are a query rewriting assistant. Transform natural language 
questions into concise retrieval-friendly format (max 30 chars, keywords only).

Examples:
- "I can't sleep lately" → insomnia treatment methods
- "how to stop anxiety" → anxiety relief techniques
- "does mood affect health" → mental health physical effects"""

    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_query}
            ],
            options={"temperature": 0.3, "num_predict": 50}
        )
        
        rewritten = response['message']['content'].strip()
        return rewritten if rewritten else raw_query
        
    except Exception as e:
        print(f"Query rewriting failed: {e}")
        return raw_query

# ============================
# 5. Response Generation (Local)
# ============================
def generate_response(query: str, context: str) -> str:
    """Generate response using Ollama chat model."""
    system_prompt = """You are a friendly mental health assistant. Answer based on context 
in a warm, supportive tone. Keep answers concise and helpful.

Guidelines:
1. Answer based on the provided context
2. Use clear, easy-to-understand language
3. If context doesn't have relevant info, say so
4. Always include disclaimer: "⚠️ This is not medical advice."
"""

    user_prompt = f"""Context information:
{context}

User question: {query}

Please provide a helpful answer:"""

    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.7, "num_predict": 500}
        )
        
        return response['message']['content'].strip()
        
    except Exception as e:
        return f"Sorry, an error occurred: {str(e)}\n\n⚠️ This is not medical advice."

# ============================
# 6. Safety Layer
# ============================
def safety_check(query: str) -> bool:
    """Check for urgent/dangerous content."""
    risky_keywords = ["suicide", "kill myself", "self-harm", "want to die", "end my life"]
    return any(keyword in query.lower() for keyword in risky_keywords)

# ============================
# 7. Main RAG Function
# ============================
def rag_chat(query: str, index, doc_embeddings, bm25_index, chunks, embed_model, reranker):
    """Main RAG pipeline."""
    if safety_check(query):
        return """🆘 You seem to be going through a difficult time.

Please remember, you are not alone:
📞 National Suicide Prevention Lifeline: 988
📞 Crisis Text Line: Text HOME to 741741

⚠️ Please seek professional help immediately."""

    # Query rewriting
    rewritten_query = rewrite_query(query)
    
    # Hybrid search
    candidate_chunks = hybrid_search(rewritten_query, index, doc_embeddings, bm25_index, chunks, embed_model)
    
    if not candidate_chunks:
        return "Sorry, I couldn't find relevant information.\n\n⚠️ This is not medical advice."

    # Reranking
    final_chunks = rerank(query, candidate_chunks, reranker, top_n=RERANK_TOP_K)
    
    # Generate response
    context = "\n".join(final_chunks)
    return generate_response(query, context)

# ============================
# 8. Initialize System
# ============================
@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with local models."""
    embed_model, reranker = load_models()
    
    raw_docs = [
        "Anxiety can cause restlessness, fatigue, and difficulty sleeping. Breathing exercises and mindfulness may help reduce anxiety symptoms.",
        "Depression is characterized by persistent sadness, loss of interest in activities, and changes in appetite or sleep patterns. Cognitive Behavioral Therapy (CBT) has been shown to be effective.",
        "Cognitive Behavioral Therapy (CBT) helps people identify and change negative thought patterns and behaviors that contribute to mental health issues.",
        "Regular sleep (7-9 hours for adults), moderate exercise, and strong social support are important foundations for mental well-being and resilience.",
        "Mindfulness meditation involves focusing attention on the present moment without judgment, which can reduce stress and improve emotional regulation.",
        "Stress management techniques include deep breathing, progressive muscle relaxation, time management, and setting healthy boundaries.",
        "Signs of burnout include exhaustion, cynicism, and reduced professional efficacy. Prevention includes work-life balance and self-care practices.",
        "Social connections and meaningful relationships are protective factors against mental health problems and can improve overall life satisfaction."
    ]
    
    chunks = chunk_documents(raw_docs)
    index, doc_embeddings, bm25_index, chunks = build_index(chunks, embed_model)
    
    return embed_model, reranker, index, doc_embeddings, bm25_index, chunks

# Initialize
embed_model, reranker, index, doc_embeddings, bm25_index, chunks = initialize_rag_system()

# ============================
# 9. Streamlit UI
# ============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for speaker, text in st.session_state.chat_history:
    role = "🧑 You" if speaker == "You" else "🤖 Bot"
    st.markdown(f"**{role}:** {text}")

user_input = st.text_input("💬 Ask me about mental health:", "", key="user_input")

if st.button("Send", type="primary"):
    if user_input.strip():
        response = rag_chat(user_input, index, doc_embeddings, bm25_index, chunks, embed_model, reranker)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
        st.rerun()
    else:
        st.warning("Please enter a message.")

# ============================
# 10. Tests
# ============================
if __name__ == "__main__":
    print("=" * 60)
    print("Local RAG Pipeline Test")
    print("=" * 60)
    
    print("\n[1] Testing RAG pipeline...")
    test_query = "How can I manage anxiety?"
    response = rag_chat(test_query, index, doc_embeddings, bm25_index, chunks, embed_model, reranker)
    print(f"Query: {test_query}")
    print(f"Response: {response[:300]}...")
    print("\n✓ Done!")
