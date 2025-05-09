import streamlit as st
import google.generativeai as genai

# --- CONFIGURE YOUR GEMINI API KEY ---
genai.configure(api_key="AIzaSyA-mtMmIKMx14s8bISw2u2EXApTKcEbz1Y")  # Be sure to secure it in production

# --- INIT GEMINI MODEL ---
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-001")

# --- STREAMLIT UI ---
st.title("🔬 Lung Cancer Assistant (Gemini Flash 2.0)")

user_input = st.text_input("Ask about lung cancer:", placeholder="e.g., What are early symptoms?")

if user_input:
    with st.spinner("Thinking with Gemini..."):
        try:
            response = model.generate_content(user_input)
            st.success("Gemini says:")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"Error from Gemini: {e}")
