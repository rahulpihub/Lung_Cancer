import pickle

# Load the dataset
with open("Faiss/lung_mapping.pkl", "rb") as f:
    dataset = pickle.load(f)

# Print dataset length
print(f"Dataset length: {len(dataset)}")

# Print sample entries
print("\nSample entries:")
for i in range(min(5, len(dataset))):
    print(f"\nEntry {i}:")
    print(dataset[i])

# Look for treatment and prevention related entries
keywords = ["treatment", "prevention", "suggestion", "therapy", "medication", "lifestyle"]
print("\nEntries related to treatment and prevention:")
for i, entry in enumerate(dataset):
    if any(keyword in entry.lower() for keyword in keywords):
        print(f"\nEntry {i}:")
        print(entry)
        if i > 10:  # Limit to 10 matches
            break
