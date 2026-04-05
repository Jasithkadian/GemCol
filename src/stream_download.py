from datasets import load_dataset
import json

print("Downloading Wikipedia (streaming mode)...")

# Use streaming to avoid downloading massive parquet files
dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True
)

documents = []
for i, doc in enumerate(dataset):
    text = doc['text'].strip()
    
    if len(text) < 100:
        continue
    
    documents.append({
        "title": doc.get('title', f'Doc_{i}'),
        "content": text[:1000]
    })
    
    if len(documents) >= 100:
        break

with open("data/raw/documents.json", 'w', encoding='utf-8') as f:
    json.dump(documents, f, indent=2, ensure_ascii=False)

print(f"✓ Saved {len(documents)} documents to data/raw/documents.json")
