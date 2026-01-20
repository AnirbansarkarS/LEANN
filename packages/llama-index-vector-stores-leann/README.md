# LlamaIndex LEANN Integration

This package provides a LlamaIndex integration for the **LEANN** (Low-storage Embedding-based Approximate Nearest Neighbors) vector store.

LEANN achieves up to **97% storage savings** compared to traditional vector databases (like ChromaDB, FAISS, or PGVector) by using a novel graph-based selective recomputation strategy.

## Installation

```bash
pip install llama-index-vector-stores-leann
```

You also need the core LEANN library and at least one backend (e.g., HNSW):

```bash
pip install leann
```

## Quick Start

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.leann import LeannVectorStore

# 1. Load documents
documents = SimpleDirectoryReader("./data").load_data()

# 2. Initialize LEANN Vector Store
vector_store = LeannVectorStore(
    index_path="./leann_index",
    embedding_model="facebook/contriever"
)

# 3. Create Index and Storage Context
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# 4. Query
query_engine = index.as_query_engine()
response = query_engine.query("What are the benefits of LEANN?")
print(response)
```

## Key Features

- **Massive Storage Savings**: Store only the graph structure and a fraction of embeddings; recompute others on-the-fly.
- **Hybrid Search Suppport**: Seamlessly integrates with LlamaIndex's `QueryFusionRetriever`.
- **Metadata Filtering**: Supports LlamaIndex metadata filters which are translated to LEANN's `MetadataFilterEngine`.
- **Private & Local**: Run entirely on your hardware with local embedding models.

## Hybrid Search Example

```python
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

# Combine LEANN (Vector) with BM25 (Keyword)
retriever = QueryFusionRetriever(
    [
        index.as_retriever(similarity_top_k=5),
        BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=5),
    ],
    num_queries=1,
)

nodes = retriever.retrieve("privacy oriented vector search")
```

## Configuration

The `LeannVectorStore` constructor accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `index_path` | `str` | *Required* | Directory where index files will be stored. |
| `backend_name` | `str` | `"hnsw"` | Underlying vector index backend (`hnsw` or `diskann`). |
| `embedding_model` | `str` | `"facebook/contriever"` | Model name for recomputation. |
| `embedding_mode` | `str` | `"sentence-transformers"` | Backend for embeddings (`sentence-transformers`, `openai`, etc.). |
| `search_complexity` | `int` | `32` | `efSearch` or equivalent for search accuracy/speed tradeoff. |
| `recompute_embeddings` | `bool` | `True` | Whether to use LEANN's recomputation to save space. |

## License

MIT
