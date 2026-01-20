import logging
import os
from typing import Any, List, Optional, Union

from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
)

from leann import LeannBuilder, LeannSearcher

logger = logging.getLogger(__name__)

class LeannVectorStore(BasePydanticVectorStore):
    """LEANN vector store for LlamaIndex.
    
    Achieves up to 97% storage reduction by using graph-based selective 
    recomputation of embeddings.
    """

    stores_text: bool = True
    flat_metadata: bool = False
    
    index_path: str
    backend_name: str = "hnsw"
    embedding_model: str = "facebook/contriever"
    embedding_mode: str = "sentence-transformers"
    graph_degree: int = 32
    build_complexity: int = 64
    search_complexity: int = 32
    recompute_embeddings: bool = True
    
    _builder: Optional[LeannBuilder] = None
    _searcher: Optional[LeannSearcher] = None

    def __init__(
        self,
        index_path: str,
        backend_name: str = "hnsw",
        embedding_model: str = "facebook/contriever",
        embedding_mode: str = "sentence-transformers",
        graph_degree: int = 32,
        build_complexity: int = 64,
        search_complexity: int = 32,
        recompute_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize LEANN vector store."""
        super().__init__(
            index_path=index_path,
            backend_name=backend_name,
            embedding_model=embedding_model,
            embedding_mode=embedding_mode,
            graph_degree=graph_degree,
            build_complexity=build_complexity,
            search_complexity=search_complexity,
            recompute_embeddings=recompute_embeddings,
            **kwargs,
        )
        self._builder = LeannBuilder(
            backend_name=self.backend_name,
            embedding_model=self.embedding_model,
            embedding_mode=self.embedding_mode,
            graph_degree=self.graph_degree,
            build_complexity=self.build_complexity,
            **kwargs,
        )
        # If index exists, load searcher
        if os.path.exists(self.index_path) or os.path.exists(f"{self.index_path}.meta.json"):
            self._searcher = LeannSearcher(self.index_path)

    @property
    def client(self) -> Any:
        """Get client."""
        return self._searcher or self._builder

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes to index."""
        ids = []
        for node in nodes:
            metadata = node_to_metadata_dict(node, remove_text=True, flat_metadata=self.flat_metadata)
            metadata["id"] = node.node_id
            
            self._builder.add_text(
                text=node.get_content(metadata_mode=MetadataMode.NONE),
                metadata=metadata
            )
            ids.append(node.node_id)
        
        # In LlamaIndex, we often expect the index to be ready after add() 
        # for simple cases, but LEANN requires a separate build step.
        # We'll rely on the user or the storage context to call persist().
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes by ref_doc_id."""
        if not self._builder:
            raise ValueError("Builder not initialized")
            
        # LEANN doesn't support incremental delete easily.
        # The recommended approach for now is to filter out the deleted doc 
        # and rebuild if we have the original chunks, or mark it as deleted 
        # in metadata if we want to avoid rebuild.
        # Here we'll just log that it's not fully supported without a rebuild.
        logger.warning("Delete operation in LEANN currently requires a full index rebuild to be effective.")
        
        # Filter existing chunks in builder
        self._builder.chunks = [
            chunk for chunk in self._builder.chunks 
            if chunk["metadata"].get("ref_doc_id") != ref_doc_id
        ]
        
        # Rebuild if index already existed
        if os.path.exists(self.index_path):
            self.persist(self.index_path)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index."""
        if not self._searcher:
            if os.path.exists(self.index_path) or os.path.exists(f"{self.index_path}.meta.json"):
                self._searcher = LeannSearcher(self.index_path)
            else:
                raise ValueError(f"Index not found at {self.index_path}. Did you call persist()?")

        # LlamaIndex filters to LEANN filters mapping
        metadata_filters = None
        if query.filters:
            metadata_filters = {}
            for f in query.filters.filters:
                # Basic mapping: LlamaIndex operator to LEANN operator
                # LEANN supports: ==, !=, <, <=, >, >=, in, not_in, contains, etc.
                op = f.operator
                if op == "==":
                    leann_op = "=="
                elif op == "!=":
                    leann_op = "!="
                else:
                    leann_op = "==" # Fallback
                
                metadata_filters[f.key] = {leann_op: f.value}

        results = self._searcher.search(
            query.query_str if query.query_str else "", # LEANN can embed the string
            top_k=query.similarity_top_k,
            complexity=self.search_complexity,
            recompute_embeddings=self.recompute_embeddings,
            metadata_filters=metadata_filters,
            **kwargs,
        )

        nodes = []
        similarities = []
        ids = []
        for res in results:
            node = metadata_dict_to_node(res.metadata)
            node.set_content(res.text)
            nodes.append(node)
            similarities.append(res.score)
            ids.append(res.id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def persist(self, persist_path: str, fs: Any = None) -> None:
        """Persist index."""
        if not self._builder:
            raise ValueError("Builder not initialized")
        
        self._builder.build_index(persist_path)
        # Reload searcher after build
        self._searcher = LeannSearcher(persist_path)
