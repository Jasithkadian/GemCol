import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import DENSE_MODEL_NAME

class DenseRetriever:
    def __init__(self, documents, model_name=DENSE_MODEL_NAME, index_path=None, embeddings_path=None):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        
        # Dimensions for all-MiniLM-L6-v2 is 384
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.index_path = index_path
        self.embeddings_path = embeddings_path
        self.index = None

    def build_index(self):
        print(f"Loading model and encoding {len(self.documents)} documents... This may take a moment.")
        # Extract text; ST handles truncation automatically if text is long
        texts = [doc["text"] for doc in self.documents]
        
        # Encode documents (generate embeddings)
        # show_progress_bar gives visual feedback for large texts
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Normalize all document embeddings before adding to FAISS index
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Build FAISS IndexFlatIP
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        print(f"Added {self.index.ntotal} embeddings to the index.")

        # Save embeddings and index
        if self.embeddings_path:
            os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
            np.save(self.embeddings_path, embeddings)
            print(f"Saved embeddings to {self.embeddings_path}")
            
        if self.index_path:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            print(f"Saved FAISS index to {self.index_path}")

    def load_index(self):
        if not self.index_path or not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
            
        print(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(self.index_path)

    def search(self, query, top_k=3):
        if self.index is None:
            raise ValueError("Index is not loaded or built. Cannot search.")
            
        # Encode the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Normalize the query embedding the same way
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search the index
        # search returns cosine similarities and the indices of the neighbors
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # If indexing failed or less docs than k exist
            if idx == -1: 
                continue
                
            doc = self.documents[idx]
            results.append({
                "id": doc.get("id", idx), # Safeguard added locally for removed ids
                "title": doc.get("title", f"Doc {doc.get('id', idx)}"),
                "score": float(dist) # cosine similarity, so higher is actually better finding!
            })
            
        return results

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_path = os.path.join(base_dir, "data", "raw", "documents.json")
    queries_path = os.path.join(base_dir, "data", "raw", "queries.json")
    
    index_path = os.path.join(base_dir, "data", "indices", "faiss_index.bin")
    embeddings_path = os.path.join(base_dir, "data", "processed", "embeddings.npy")

    # Load documents
    try:
        with open(docs_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
    except FileNotFoundError:
        print(f"Error: {docs_path} not found. Please ensure you have generated documents.")
        return

    # Load queries
    try:
        with open(queries_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
    except FileNotFoundError:
        print(f"Error: {queries_path} not found.")
        return

    # Initialize Retriever
    retriever = DenseRetriever(documents, index_path=index_path, embeddings_path=embeddings_path)
    
    # Check if index already exists to avoid re-encoding on repeated runs
    if os.path.exists(index_path):
        retriever.load_index()
    else:
        retriever.build_index()

    print("=" * 50)

    # Run query searches
    for q in queries:
        query_text = q.get("query", q.get("text", ""))
        if not query_text:
            continue
            
        print(f"Query: '{query_text}'")
        top_results = retriever.search(query_text, top_k=3)
        
        for rank, res in enumerate(top_results, 1):
            # For Cosine Similarity, higher score is more similar
            print(f"  {rank}. [Score: {res['score']:.4f}] {res['title']} (ID: {res['id']})")
        print("-" * 50)

if __name__ == "__main__":
    main()
