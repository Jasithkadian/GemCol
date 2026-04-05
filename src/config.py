# Data Paths
DATA_DIR = "data"
RAW_DIR = "data/raw"
DOCUMENTS_PATH = "data/raw/documents.json"
QUERIES_PATH = "data/raw/queries.json"
INDEX_PATH = "data/indices/faiss_index.bin"
EMBEDDINGS_PATH = "data/processed/embeddings.npy"

# BM25 Settings
MIN_BM25_SCORE = 0.01

# Dense Retriever Settings
DENSE_MODEL_NAME = "all-MiniLM-L6-v2"

# Fusion Settings
FUSION_METHOD = "alpha"   # "rrf" or "alpha"
ALPHA = 0.7
TOP_K_RESULTS = 3
FETCH_K = 10
RRF_K = 60

# Reranker Settings (for Section 4 later)
ENABLE_RERANKER = False
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 5
