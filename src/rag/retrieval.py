"""RAG retrieval pipeline"""
from typing import List, Dict
from .vector_store import VectorStore
from transformers import AutoTokenizer, AutoModel
import torch

class RAGPipeline:
    def __init__(self, vector_store: VectorStore, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize RAG pipeline"""
        self.vector_store = vector_store
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.embedding_model.eval()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        results = self.vector_store.query(query, n_results=top_k)
        
        retrieved_docs = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    'document': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        return retrieved_docs
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents as context"""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[{i}] {doc['document']}")
        return "\n\n".join(context_parts)
    
    def generate_prompt(self, query: str, context: str) -> str:
        """Generate prompt with retrieved context"""
        return f"""### Context:
{context}

### Question:
{query}

### Answer:
"""

