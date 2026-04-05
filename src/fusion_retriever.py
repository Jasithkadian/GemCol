import json
import os

# Import the existing retriever classes
# Make sure no name resolution conflicts exist based on your directory structure
from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever
from config import FUSION_METHOD, ALPHA, TOP_K_RESULTS, FETCH_K, RRF_K

class FusionRetriever:
    def __init__(self, bm25_retriever, dense_retriever, rrf_k=RRF_K):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.k = rrf_k

    def hybrid_search(self, query, top_k=3, fetch_k=20):
        """
        Runs BM25 and Dense searches, then fuses results using RRF or alpha fusion.
        fetch_k: how many candidates to pull from each system before fusing.
        top_k: how many final results to return.
        """
        # Fetch initial results from both systems
        bm25_results = self.bm25.search(query, top_k=fetch_k)
        dense_results = self.dense.search(query, top_k=fetch_k)

        title_map = {}
        for res in bm25_results + dense_results:
            title_map[res['id']] = res['title']

        if FUSION_METHOD == "alpha":
            all_ids = set([res['id'] for res in bm25_results] + [res['id'] for res in dense_results])
            
            bm25_scores = [res['score'] for res in bm25_results]
            min_bm25 = min(bm25_scores) if bm25_scores else 0.0
            max_bm25 = max(bm25_scores) if bm25_scores else 1.0
            if max_bm25 == min_bm25:
                max_bm25 = min_bm25 + 1e-9
                
            bm25_dict = {res['id']: res['score'] for res in bm25_results}
            dense_dict = {res['id']: res['score'] for res in dense_results}
            
            hybrid_scores = {}
            for doc_id in all_ids:
                raw_bm25 = bm25_dict.get(doc_id, 0.0)
                norm_bm25 = (raw_bm25 - min_bm25) / (max_bm25 - min_bm25)
                norm_bm25 = max(0.0, min(1.0, norm_bm25))
                
                dense_score = dense_dict.get(doc_id, 0.0)
                
                hybrid_score = (ALPHA * dense_score) + ((1 - ALPHA) * norm_bm25)
                hybrid_scores[doc_id] = hybrid_score
                
            sorted_fusion = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            
        else:
            # RRF mapping
            rrf_scores = {}

            # 1. Process BM25 Ranks
            for rank, res in enumerate(bm25_results, 1):
                doc_id = res['id']
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (self.k + rank)

            # 2. Process Dense Ranks
            for rank, res in enumerate(dense_results, 1):
                doc_id = res['id']
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (self.k + rank)

            # 3. Sort Combined Results Descending (Higher RRF = Better Match)
            sorted_fusion = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        hybrid_results = []
        for doc_id, score in sorted_fusion[:top_k]:
            hybrid_results.append({
                "id": doc_id,
                "title": title_map[doc_id],
                "fusion_score": score
            })

        return bm25_results[:top_k], dense_results[:top_k], hybrid_results

def relevance_label(score):
    if score >= 0.6:
        return "HIGH    "
    elif score >= 0.3:
        return "MEDIUM  "
    else:
        return "LOW     "

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_path = os.path.join(base_dir, "data", "raw", "documents.json")
    queries_path = os.path.join(base_dir, "data", "raw", "queries.json")

    index_path = os.path.join(base_dir, "data", "indices", "faiss_index.bin")
    embeddings_path = os.path.join(base_dir, "data", "processed", "embeddings.npy")

    # Load data
    try:
        with open(docs_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
        with open(queries_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading JSON data: {e}")
        print("Make sure you have generated/downloaded documents.json and queries.json.")
        return

    # Initialize Retrievers
    print("Initializing BM25 Retriever...")
    bm25_r = BM25Retriever(documents)
    
    print("Initializing Dense Retriever...")
    dense_r = DenseRetriever(documents, index_path=index_path, embeddings_path=embeddings_path)
    
    if os.path.exists(index_path):
        dense_r.load_index()
    else:
        dense_r.build_index()

    # Initialize Fusion
    fusion_r = FusionRetriever(bm25_r, dense_r, rrf_k=RRF_K)

    print("=" * 60)

    # Run queries and compare
    for q in queries:
        query_text = q.get("query", q.get("text", ""))
        if not query_text:
            continue
            
        print(f"Query: {query_text}\n")
        
        # We fetch FETCH_K results from each to compute RRF, but only show TOP_K_RESULTS
        bm25_res, dense_res, hybrid_res = fusion_r.hybrid_search(
            query_text, 
            top_k=TOP_K_RESULTS, 
            fetch_k=FETCH_K
        )

        print("BM25 Results:")
        for res in bm25_res:
            print(f"-> {res['title']} (Score: {res['score']:.4f})")
            
        print("\nDense Results:")
        for res in dense_res:
            print(f"-> {res['title']:<14} (Similarity: {res['score']*100:>5.2f}%)")
            
        print("\nHybrid Results:")
        for rank, res in enumerate(hybrid_res, 1):
            print(f"Rank {rank} | {res['title']:<19} | Score: {res['fusion_score']:.4f} | Relevance: {relevance_label(res['fusion_score'])}")
            
        if hybrid_res:
            top_res = hybrid_res[0]
            top_label = relevance_label(top_res['fusion_score']).strip()
            print(f"\nTop Match: {top_res['title']} | Confidence: {top_res['fusion_score']*100:.2f}% | Relevance: {top_label}")
            
        print("-" * 60)

if __name__ == "__main__":
    main()
