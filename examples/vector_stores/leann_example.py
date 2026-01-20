"""Example of using LEANN vector store with LlamaIndex.

This example demonstrates basic usage, hybrid search, and migration from ChromaDB.
"""

import os
import shutil
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.leann import LeannVectorStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

def run_example():
    # 1. Setup paths
    index_path = "./leann_index_example"
    data_dir = "./data" # Assuming some documents exist here
    
    # Cleanup previous runs
    if os.path.exists(index_path):
        if os.path.isdir(index_path):
            shutil.rmtree(index_path)
        else:
            os.remove(index_path)

    # 2. Load documents
    print(f"Loading documents from {data_dir}...")
    # Create dummy data if directory doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        with open(os.path.join(data_dir, "test.txt"), "w") as f:
            f.write("LEANN is a low-storage vector index that achieves 97% reduction.")
    
    documents = SimpleDirectoryReader(data_dir).load_data()

    # 3. Initialize LEANN Vector Store
    print("Initializing LEANN Vector Store...")
    vector_store = LeannVectorStore(
        index_path=index_path,
        backend_name="hnsw",
        embedding_model="facebook/contriever"
    )

    # 4. Create Index and Storage Context
    print("Building index (this will automatically call persist())...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    # 5. Basic Query
    print("\n--- Basic Semantic Query ---")
    query_engine = index.as_query_engine()
    response = query_engine.query("What is the storage reduction of LEANN?")
    print(f"Response: {response}")

    # 6. Hybrid Search (BM25 + LEANN)
    print("\n--- Hybrid Search (BM25 + LEANN) ---")
    retriever = QueryFusionRetriever(
        [
            index.as_retriever(similarity_top_k=2),
            BM25Retriever.from_defaults(
                docstore=index.docstore, 
                similarity_top_k=2
            ),
        ],
        num_queries=1,
        use_async=False, # Use False for simple demonstration
    )

    nodes = retriever.retrieve("storage reduction")
    print(f"Retrieved {len(nodes)} nodes via hybrid search.")
    for i, node in enumerate(nodes):
        print(f"Node {i+1}: {node.get_content()[:100]}... (Score: {node.get_score()})")

    # Cleanup
    print("\nExample finished. Cleaning up...")
    if os.path.exists(index_path):
        shutil.rmtree(index_path)

if __name__ == "__main__":
    run_example()
