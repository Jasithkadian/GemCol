import json
import os
from rank_bm25 import BM25Okapi

MIN_SCORE_THRESHOLD = 0.01

class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        # Preprocess: Simple lowercase split tokenization
        tokenized_corpus = [doc.get('content', doc.get('text', '')).lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=3):
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices (sorted descending by score)
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        # Prepare results
        results = []
        for i in top_indices:
            score = doc_scores[i]
            if score < MIN_SCORE_THRESHOLD:
                continue
                
            # Safely grab title if it exists, otherwise use ID
            doc_id = self.documents[i].get("id", i)
            title = self.documents[i].get("title", f"Doc {doc_id}")
            results.append({
                "id": doc_id,
                "title": title,
                "score": score
            })
        return results

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_path = os.path.join(base_dir, "data", "raw", "documents.json")
    queries_path = os.path.join(base_dir, "data", "raw", "queries.json")

    # Load documents
    try:
        with open(docs_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
    except FileNotFoundError:
        print(f"Error: Documents not found at {docs_path}")
        print("Please make sure you have generated/downloaded documents.json!")
        return

    # Load queries
    try:
        with open(queries_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
    except FileNotFoundError:
        print(f"Error: Queries not found at {queries_path}")
        return

    # Build the BM25 index
    print(f"Building BM25 Index for {len(documents)} documents...")
    retriever = BM25Retriever(documents)
    print("Index built successfully!")
    print("=" * 50)

    # Run searches for each query
    for q in queries:
        # Handles both shapes: {"query": "text"} or {"text": "text"}
        query_text = q.get("query", q.get("text", ""))
        if not query_text:
            continue
            
        print(f"Query: '{query_text}'")
        
        top_results = retriever.search(query_text, top_k=3)
        for rank, res in enumerate(top_results, 1):
            print(f"  {rank}. [Score: {res['score']:.4f}] {res['title']} (ID: {res['id']})")
            
        print("-" * 50)

if __name__ == "__main__":
    main()
