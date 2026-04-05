import os
import json
from datasets import load_dataset
from tqdm import tqdm

print("Downloading Wikipedia...")

# Use the new dataset format
dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train[:1500]"
)

print(f"Downloaded {len(dataset)} articles")

documents = []
for i, doc in enumerate(tqdm(dataset, desc="Processing"), start=1):
    text = doc['text'].strip()
    
    if len(text) < 100:
        continue
    
    documents.append({
        "id": i,
        "title": doc.get('title', f'Doc_{i}'),
        "text": text[:2000]
    })

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(base_dir, "data", "raw", "documents.json")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(documents, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(documents)} documents")
print("First 3 document titles as preview:")
for idx, d in enumerate(documents[:3]):
    print(f"  {idx + 1}. {d['title']}")
print(f"Save path: {save_path}")