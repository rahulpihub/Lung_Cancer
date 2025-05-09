from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load dataset
with open("lung_dataset_clean.txt", "r") as f:
    dataset = [line.strip() for line in f if line.strip()]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed dataset
embeddings = model.encode(dataset, convert_to_numpy=True)

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # IP = inner product = cosine if normalized
index.add(embeddings)

# Save index and dataset mapping
faiss.write_index(index, "Faiss/lung_faiss.index")
with open("Faiss/lung_mapping.pkl", "wb") as f:
    pickle.dump(dataset, f)

print("✅ FAISS index built and saved.")
