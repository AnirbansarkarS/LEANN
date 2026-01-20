import os
import shutil
import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.leann import LeannVectorStore

@pytest.fixture
def leann_index_path() -> str:
    path = "./test_leann_index"
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    # Also clean up potential meta/passages files
    for ext in [".meta.json", ".passages.jsonl", ".passages.idx", ".ids.txt"]:
        p = f"{path}{ext}"
        if os.path.exists(p):
            os.remove(p)

    yield path

    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    for ext in [".meta.json", ".passages.jsonl", ".passages.idx", ".ids.txt"]:
        p = f"{path}{ext}"
        if os.path.exists(p):
            os.remove(p)

def test_leann_vector_store_basic(leann_index_path: str) -> None:
    """Test basic functionality of LeannVectorStore."""
    # Initialize
    vector_store = LeannVectorStore(
        index_path=leann_index_path,
        embedding_model="facebook/contriever", # Small model for testing
        embedding_mode="sentence-transformers"
    )
    
    # Add nodes
    nodes = [
        TextNode(text="LEANN is a low-storage vector index.", node_id="1"),
        TextNode(text="LlamaIndex is a data framework for LLM applications.", node_id="2"),
        TextNode(text="The quick brown fox jumps over the lazy dog.", node_id="3"),
    ]
    vector_store.add(nodes)
    
    # Persist (Build index)
    vector_store.persist(leann_index_path)
    
    # Query
    query = VectorStoreQuery(query_str="What is LEANN?", similarity_top_k=1)
    result = vector_store.query(query)
    
    assert len(result.nodes) == 1
    assert result.nodes[0].get_content() == "LEANN is a low-storage vector index."
    assert result.ids[0] == "1"

def test_leann_vector_store_delete(leann_index_path: str) -> None:
    """Test delete functionality (requires rebuild in LEANN)."""
    vector_store = LeannVectorStore(
        index_path=leann_index_path,
        embedding_model="facebook/contriever",
        embedding_mode="sentence-transformers"
    )
    
    nodes = [
        TextNode(text="Apple", node_id="n1", metadata={"ref_doc_id": "doc1"}),
        TextNode(text="Banana", node_id="n2", metadata={"ref_doc_id": "doc2"}),
    ]
    vector_store.add(nodes)
    vector_store.persist(leann_index_path)
    
    # Delete doc1
    vector_store.delete("doc1")
    
    # Query
    query = VectorStoreQuery(query_str="Apple", similarity_top_k=2)
    result = vector_store.query(query)
    
    # Since we deleted doc1, and delete calls persist(), the index should only have doc2
    assert len(result.nodes) == 1
    assert result.nodes[0].get_content() == "Banana"

if __name__ == "__main__":
    import shutil
    path = "./test_leann_index"
    try:
        test_leann_vector_store_basic(path)
        print("Basic test passed!")
        test_leann_vector_store_delete(path)
        print("Delete test passed!")
    finally:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        for ext in [".meta.json", ".passages.jsonl", ".passages.idx", ".ids.txt"]:
            p = f"{path}{ext}"
            if os.path.exists(p):
                os.remove(p)

