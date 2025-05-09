import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load dataset and FAISS index
faiss_index = faiss.read_index("Faiss/lung_faiss.index")
with open("Faiss/lung_mapping.pkl", "rb") as f:
    dataset = pickle.load(f)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to search in FAISS index
def search(query, k=5):
    # Embed the query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Normalize the query embedding
    faiss.normalize_L2(query_embedding)
    
    # Search in FAISS index
    D, I = faiss_index.search(query_embedding, k)  # D: distances, I: indices
    
    return D, I

# Streamlit UI
st.title("Lung Cancer Risk Prediction")
st.write("Enter your symptoms to check the risk of lung cancer:")

# User input for symptoms or details
user_input = st.text_input("Enter symptoms or details here:")

# Check if the user has entered input
if user_input:
    # Perform the FAISS search
    distances, indices = search(user_input)
    
    # Calculate the percentage match based on cosine similarity
    risk_percentage = (1 - np.mean(distances[0])) * 100  # Risk estimation based on similarity (improve this formula if needed)
    
    # Display the risk percentage
    st.write(f"⚠️ Risk Estimate: {risk_percentage:.2f}% match with known lung cancer symptoms")
    
    # Display matched data from the dataset
    st.subheader("Matched Dataset Segments:")
    for i, idx in enumerate(indices[0]):
        st.write(f"Match {i + 1}:")
        st.write(f"Text: {dataset[idx]}")
        st.write(f"Cosine Similarity: {distances[0][i]:.4f}")
        st.write("\n")