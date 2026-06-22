"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM provider (Gemini via OpenAI-compatible endpoint) ---
# Gemini expose một endpoint tương thích OpenAI, nên ta giữ nguyên `openai` SDK
# trong scaffold mà chỉ cần đổi base_url + api_key + model.
#   https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")


def _clean(v: str) -> str:
    """Bỏ qua các giá trị placeholder (sk-..., rỗng)."""
    v = (v or "").strip()
    return "" if v in ("", "sk-...", "your-key", "...") else v


GEMINI_API_KEY = _clean(GEMINI_API_KEY)
# langchain-google-genai đọc GOOGLE_API_KEY từ env → set lại cho chắc.
if GEMINI_API_KEY:
    os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY)
_REAL_OPENAI_KEY = _clean(os.getenv("OPENAI_API_KEY", ""))
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
# gemini-embedding-001: model embedding hiện hành trên OpenAI-compat endpoint
# (text-embedding-004 KHÔNG khả dụng ở endpoint này). Hỗ trợ output 768/1536/3072-dim.
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_EMBED_DIM = int(os.getenv("GEMINI_EMBED_DIM", "768"))

# OPENAI_API_KEY: dùng key thật của OpenAI nếu có; nếu không, fallback sang Gemini key
# để các đoạn code `OpenAI()` (đã trỏ base_url Gemini) vẫn chạy.
OPENAI_API_KEY = _REAL_OPENAI_KEY or GEMINI_API_KEY

# Cờ tiện dụng: có LLM để gọi không, và đang dùng Gemini hay OpenAI.
USE_GEMINI = bool(GEMINI_API_KEY) and not _REAL_OPENAI_KEY
LLM_API_KEY = OPENAI_API_KEY
LLM_BASE_URL = GEMINI_BASE_URL if USE_GEMINI else os.getenv("OPENAI_BASE_URL", "")
LLM_CHAT_MODEL = GEMINI_CHAT_MODEL if USE_GEMINI else os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
LLM_EMBED_MODEL = GEMINI_EMBED_MODEL if USE_GEMINI else os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def get_llm_client():
    """Trả về một OpenAI client đã cấu hình (Gemini hoặc OpenAI). None nếu thiếu key."""
    if not LLM_API_KEY:
        return None
    from openai import OpenAI

    kwargs = {"api_key": LLM_API_KEY}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return OpenAI(**kwargs)


# --- Qdrant (local Docker hoặc Qdrant Cloud) ---
# Nếu QDRANT_URL được set (Qdrant Cloud) → ưu tiên dùng url + api_key.
QDRANT_URL = os.getenv("QDRANT_URL", "") or os.getenv("QDRANT_CLUSTER_ENDPOINT", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"


def get_qdrant_client():
    """Trả về QdrantClient: Qdrant Cloud (nếu có QDRANT_URL) hoặc local."""
    from qdrant_client import QdrantClient

    if QDRANT_URL:
        # timeout cao để chịu được độ trễ mạng tới Qdrant Cloud (eu-west-2).
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=120)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)

# --- Embedding ---
# Hai backend:
#   - "api"   : gọi Gemini text-embedding-004 (768-dim) → KHÔNG cần tải model nặng.
#   - "local" : bge-m3 (1024-dim) qua sentence-transformers (cần tải ~2.2GB).
# Mặc định dùng "api" khi có Gemini key để tránh tải model.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "api" if USE_GEMINI else "local")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")   # dùng khi backend=local
EMBEDDING_DIM = GEMINI_EMBED_DIM if EMBEDDING_BACKEND == "api" else 1024

# Reranker backend: "api" = LLM-as-reranker (Gemini), "local" = CrossEncoder bge-reranker.
RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "api" if USE_GEMINI else "local")


def embed_texts(texts, batch_size: int = 100):
    """Embed danh sách text → list[list[float]] qua Gemini API (text-embedding-004)."""
    client = get_llm_client()
    if client is None:
        raise RuntimeError("Cần LLM/Gemini key để dùng embedding API.")
    out = []
    for start in range(0, len(texts), batch_size):
        batch = [t if t.strip() else " " for t in texts[start:start + batch_size]]
        resp = client.embeddings.create(model=LLM_EMBED_MODEL, input=batch,
                                        dimensions=GEMINI_EMBED_DIM)
        out.extend([d.embedding for d in resp.data])
    return out


def embed_query(text: str):
    """Embed 1 query → list[float]."""
    return embed_texts([text])[0]

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
