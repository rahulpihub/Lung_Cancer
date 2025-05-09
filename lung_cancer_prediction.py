import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load symptom dataset and FAISS index
faiss_index_symptom = faiss.read_index("Faiss/lung_faiss.index")
with open("Faiss/lung_mapping.pkl", "rb") as f:
    symptom_dataset = pickle.load(f)

# Load treatment dataset and FAISS index
faiss_index_treatment = faiss.read_index("Faiss/lung_treatment.index")
with open("Faiss/lung_treatment.pkl", "rb") as f:
    treatment_dataset = pickle.load(f)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to search FAISS index
def search_faiss(query, faiss_index, k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    D, I = faiss_index.search(query_embedding, k)
    return D, I

# Streamlit UI
st.title("Lung Cancer Risk Prediction")
st.write("Enter your symptoms to check the risk and receive a treatment suggestion:")

# User input
user_input = st.text_input("Enter symptoms or details here:")

if user_input:
    # --- Symptom search ---
    symptom_distances, symptom_indices = search_faiss(user_input, faiss_index_symptom)
    risk_percentage = (1 - np.mean(symptom_distances[0])) * 100

    st.write(f"⚠️ **Risk Estimate**: {risk_percentage:.2f}% match with known lung cancer symptoms")

    st.subheader("🔍 Matched Symptom Descriptions:")
    for i, idx in enumerate(symptom_indices[0]):
        st.markdown(f"**Match {i + 1}**:")
        st.write(f"Text: {symptom_dataset[idx]}")
        st.write(f"Cosine Similarity: {symptom_distances[0][i]:.4f}")
        st.write("---")

    # --- Treatment plan search ---
    st.subheader("💊 Suggested Treatment Plan:")
    treatment_distances, treatment_indices = search_faiss(user_input, faiss_index_treatment, k=1)
    top_treatment = treatment_dataset[treatment_indices[0][0]]
    st.write(top_treatment)
