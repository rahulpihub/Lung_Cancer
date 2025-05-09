import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import google.generativeai as genai

# Configure the Gemini API key
genai.configure(api_key="AIzaSyAjBwYfXyCJ0SPzrV0meK2dPvhbLNMrqIs")

# Load dataset and FAISS index
faiss_index = faiss.read_index("Faiss/lung_faiss.index")
with open("Faiss/lung_mapping.pkl", "rb") as f:
    dataset = pickle.load(f)

# Load embedding model for FAISS search
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

# Function to get the risk estimate from Gemini AI
def get_risk_estimate(user_input):
    prompt = f"Assess the lung cancer risk based on these symptoms: {user_input}. Provide a risk percentage."
    
    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-001")
    response = model.generate_content(prompt)
    return response.text

# Function to get prevention and suggestions based on risk level
def get_prevention_suggestions(user_input, risk_level):
    prompt = f"""Based on the symptoms: {user_input} and a risk level of {risk_level}%, provide:
    1. Detailed explanation of what these symptoms might indicate
    2. Preventive measures the person should take
    3. Lifestyle changes that could help reduce risk
    4. When they should seek medical attention
    5. Screening recommendations
    
    Format the response in clear sections with bullet points where appropriate."""
    
    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-001")
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI setup
st.title("Lung Cancer Risk Prediction")
st.write("Enter your symptoms to check the risk of lung cancer:")

# User input for symptoms or details
user_input = st.text_input("Enter symptoms or details here:")

# Check if the user has entered input
if user_input:
    # Step 1: Get the risk estimate from Gemini AI
    gemini_response = get_risk_estimate(user_input)
    
    # Step 2: Extract the risk percentage from Gemini's response
    gemini_risk_estimate = 0
    try:
        gemini_risk_estimate = float(gemini_response.split('%')[0].strip())  # Extract percentage
    except:
        gemini_risk_estimate = 50  # Default to 50% if parsing fails
    
    # Step 3: Perform the FAISS search
    distances, indices = search(user_input)
    
    # Step 4: Calculate the percentage match based on cosine similarity
    faiss_risk_percentage = (1 - np.mean(distances[0])) * 100  # Risk estimation based on similarity
    
    # Step 5: Combine the results (Gemini AI and FAISS match)
    final_risk_estimate = (gemini_risk_estimate + faiss_risk_percentage) / 2  # Average the estimates
    
    # Step 6: Display the results
    st.write(f"⚠️ **Risk Estimate** from Gemini AI: {gemini_risk_estimate:.2f}%")
    st.write(f"📊 **Cosine Similarity-based Risk Estimate**: {faiss_risk_percentage:.2f}%")
    st.write(f"🟠 **Final Combined Risk Estimate**: {final_risk_estimate:.2f}%")
    
    # Step 7: Display matched data from the dataset
    st.subheader("Matched Dataset Segments:")
    for i, idx in enumerate(indices[0]):
        st.write(f"**Match {i + 1}:**")
        st.write(f"**Text:** {dataset[idx]}")
        st.write(f"**Cosine Similarity:** {distances[0][i]:.4f}")
        st.write("\n")
    
    # Provide a final recommendation based on the risk estimate
    if final_risk_estimate > 70:
        st.write("✅ **High risk detected based on symptoms. Consider consulting a healthcare professional immediately.**")
        risk_level = "High"
    elif final_risk_estimate > 50:
        st.write("⚠️ **Moderate risk detected. You should seek medical advice soon.**")
        risk_level = "Moderate"
    else:
        st.write("🟢 **Low risk detected. However, if symptoms persist, consider seeing a doctor.**")
        risk_level = "Low"
    
    # Get prevention and suggestion information
    prevention_suggestions = get_prevention_suggestions(user_input, final_risk_estimate)
    
    # Display prevention and suggestions in an expandable section
    with st.expander("📋 **Detailed Analysis, Prevention & Suggestions**", expanded=True):
        st.markdown(prevention_suggestions)

else:
    st.warning("Please enter some symptoms to check the risk.")
