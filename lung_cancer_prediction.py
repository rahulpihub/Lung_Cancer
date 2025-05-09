import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import google.generativeai as genai

# Configure the Gemini API key
genai.configure(api_key="AIzaSyAjBwYfXyCJ0SPzrV0meK2dPvhbLNMrqIs")
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-001")

# Load symptom dataset and FAISS index
faiss_index_symptom = faiss.read_index("Faiss/lung_faiss.index")
with open("Faiss/lung_mapping.pkl", "rb") as f:
    symptom_dataset = pickle.load(f)

# Load treatment dataset and FAISS index
faiss_index_treatment = faiss.read_index("Faiss/lung_treatment.index")
with open("Faiss/lung_treatment.pkl", "rb") as f:
    treatment_dataset = pickle.load(f)

# Load embedding model
model_embed = SentenceTransformer('all-MiniLM-L6-v2')

# Function to search FAISS index
def search_faiss(query, faiss_index, k=5):
    query_embedding = model_embed.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    D, I = faiss_index.search(query_embedding, k)
    return D, I

# Function to generate treatment suggestion using Gemini AI
def get_gemini_treatment_suggestion(user_input, symptom_matches, treatment_matches):
    # Create the prompt based on the user input and matched symptom data
    symptom_texts = "\n".join([f"Text: {symptom_dataset[idx]} \nCosine Similarity: {distance:.4f}" 
                              for idx, distance in zip(symptom_matches[1][0], symptom_matches[0][0])])

    # Creating the treatment data from the treatment FAISS index
    treatment_texts = "\n".join([f"Suggested Treatment Plan: {treatment_dataset[idx]}" 
                                 for idx in treatment_matches[1][0]])

    prompt = (
        f"User Input: {user_input}\n\n"
        f"Matched Symptoms and Similarity:\n{symptom_texts}\n\n"
        f"Treatment Suggestions:\n{treatment_texts}\n\n"
        "Based on the above information, provide a finalized lung cancer treatment plan.i dont want to any other irrelevant information. "
        "Please be concise and clear in your response. "
    )
    
    # Get treatment suggestion from Gemini AI
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("Lung Cancer Risk Prediction and Treatment Suggestion")
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

    # --- Gemini AI Treatment Suggestion ---
    st.subheader("🤖 Gemini AI Finalized Treatment Plan:")
    gemini_response = get_gemini_treatment_suggestion(user_input, (symptom_distances, symptom_indices), (treatment_distances, treatment_indices))
    st.write(gemini_response)
