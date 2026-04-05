from datasets import load_dataset
import json
from tqdm import tqdm

print("Downloading Wikipedia...")

# Use the new dataset format
dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train[:1000]"
)

print(f"Downloaded {len(dataset)} articles")

# Convert to simple format
documents = []
for i, doc in enumerate(tqdm(dataset, desc="Processing")):
    text = doc['text'].strip()
    
    if len(text) < 100:
        continue
    
    documents.append({
        "id": i,
        "title": doc.get('title', f'Doc_{i}'),
        "text": text[:1000]
    })

# Save
with open("data/raw/documents.json", 'w', encoding='utf-8') as f:
    json.dump(documents, f, indent=2, ensure_ascii=False)

print(f"✓ Saved {len(documents)} documents to data/raw/documents.json")
print(f"Sample title: {documents[0]['title']}")