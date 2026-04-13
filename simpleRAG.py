# ===== Streamlit RAG Mental Health Chatbot =====

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

st.set_page_config(page_title="Mental Health RAG Chatbot")

st.title("🧠 Mental Health Chatbot (RAG)")
st.write("⚠️ This is not medical advice.")

# ---------------------------
# 1. Documents
# ---------------------------
docs = [
    "Anxiety can cause restlessness, fatigue, and difficulty sleeping. Breathing exercises may help.",
    "Depression is characterized by persistent sadness and loss of interest in activities.",
    "Cognitive Behavioral Therapy (CBT) helps people reframe negative thoughts.",
    "Regular sleep, exercise, and social support are important for mental well-being.",
    "Mindfulness meditation can reduce stress and improve emotional regulation."
]

# ---------------------------
# 2. Load models (cache for speed)
# ---------------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-mpnet-base-v2")
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        max_new_tokens=100
    )
    return embed_model, generator

embed_model, generator = load_models()

# ---------------------------
# 3. Build vector DB
# ---------------------------
@st.cache_resource
def build_index(docs):
    embeddings = embed_model.encode(docs)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

index, doc_embeddings = build_index(docs)

# ---------------------------
# 4. Safety layer
# ---------------------------
def safety_check(query):
    risky_words = ["suicide", "kill myself", "self-harm"]
    return any(word in query.lower() for word in risky_words)

# ---------------------------
# 5. RAG function
# ---------------------------
def rag_chat(query, k=2):

    if safety_check(query):
        return "⚠️ Please seek professional help or contact a trusted person."

    q_emb = embed_model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    retrieved_docs = [docs[i] for i in I[0]]

    context = "\n".join(retrieved_docs)

    prompt = f"""
You are a helpful mental health assistant.

Context:
{context}

User: {query}
Answer:
"""

    result = generator(prompt)[0]["generated_text"]

    return "⚠️ This is not medical advice.\n\n" + result


# ---------------------------
# 6. Chat UI
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    response = rag_chat(user_input)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display chat
for speaker, text in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")