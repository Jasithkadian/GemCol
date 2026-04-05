import os
import json
import wikipediaapi

wiki = wikipediaapi.Wikipedia('GemCol/1.0', 'en')

CORE_ARTICLES = [
    "France", "Paris", "French Revolution", "French language",
    "William Shakespeare", "Romeo and Juliet", "Hamlet", "Macbeth",
    "Photosynthesis", "Chlorophyll", "Plant cell", "Carbon dioxide",
    "Artificial intelligence", "Machine learning", "Deep learning",
    "History of artificial intelligence", "Neural network",
    "World War II", "Adolf Hitler", "Nazi Germany",
    "Causes of World War II", "Winston Churchill", "D-Day"
]

PADDING_ARTICLES = [
    "Albert Einstein", "Isaac Newton", "Charles Darwin",
    "Quantum mechanics", "Theory of relativity", "DNA",
    "Ancient Rome", "Ancient Greece", "Roman Empire",
    "United States", "United Kingdom", "China", "India",
    "Mathematics", "Physics", "Chemistry", "Biology",
    "Internet", "Computer science", "Programming language",
    "Democracy", "Capitalism", "Philosophy", "Psychology"
]

all_articles_to_fetch = CORE_ARTICLES + PADDING_ARTICLES
total_to_fetch = len(all_articles_to_fetch)

documents = []
skipped_count = 0
downloaded_titles = []

for idx, title in enumerate(all_articles_to_fetch, 1):
    page = wiki.page(title)
    
    if not page.exists():
        print(f"Skipped: {title} (not found)")
        skipped_count += 1
        continue
        
    text = page.text.strip()
    
    if len(text) < 100:
        print(f"Skipped: {title} (too short)")
        skipped_count += 1
        continue
        
    text_cut = text[:2000]
    doc_id = len(documents) + 1
    
    documents.append({
        "id": doc_id,
        "title": page.title,
        "text": text_cut
    })
    downloaded_titles.append(page.title)
    
    print(f"Downloaded {idx}/{total_to_fetch}: {page.title} ({len(text_cut)} chars)")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(base_dir, "data", "raw", "documents.json")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(documents, f, indent=2, ensure_ascii=False)

queries = [
    {"id": 1, "query": "What is the capital of France?"},
    {"id": 2, "query": "Who wrote Romeo and Juliet?"},
    {"id": 3, "query": "How does photosynthesis work?"},
    {"id": 4, "query": "History of artificial intelligence"},
    {"id": 5, "query": "What were the main causes of World War II?"}
]

queries_path = os.path.join(base_dir, "data", "raw", "queries.json")
with open(queries_path, 'w', encoding='utf-8') as f:
    json.dump(queries, f, indent=2, ensure_ascii=False)

print(f"\nTotal articles downloaded: {len(documents)}")
print(f"Total articles skipped: {skipped_count}")
print("Downloaded titles:")
for t in downloaded_titles:
    print(f"- {t}")

print(f"\nDocuments save path confirmed: {save_path}")
print(f"Queries save path confirmed: {queries_path}")
