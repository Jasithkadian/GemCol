from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL

def relevance_label(score):
    if score > 3:
        return "HIGH"
    elif score > 0:
        return "MEDIUM"
    else:
        return "LOW"

class ReRanker:
    def __init__(self, model_name=RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k=5):
        if not candidates:
            return []
            
        # Extract text/content dynamically for the cross-encoder
        pairs = [(query, doc.get('text', doc.get('content', ''))) for doc in candidates]
        
        scores = self.model.predict(pairs)
        
        results = []
        for doc, score in zip(candidates, scores):
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "rerank_score": float(score)
            })
            
        results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]
