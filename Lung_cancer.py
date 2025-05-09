import google.generativeai as genai

# Load your API key
genai.configure(api_key="AIzaSyA-mtMmIKMx14s8bISw2u2EXApTKcEbz1Y")  # Replace securely later

# Load Gemini 1.5 Flash model
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-001")

# Test prompt
prompt = "What are the early symptoms of lung cancer?"

# Generate response
response = model.generate_content(prompt)
print("Gemini Response:\n", response.text)
