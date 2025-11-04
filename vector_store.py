"""
Professional Vector Store Management for MedGPT
Handles FAISS vectorstore with caching and error handling
"""

from pathlib import Path
from typing import List, Optional
import streamlit as st

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.schema import Document


class VectorStoreManager:
    """Manages FAISS vectorstore with professional error handling"""
    
    def __init__(
        self,
        vectorstore_path: str = "vectorstore",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.vectorstore_path = Path(vectorstore_path)
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
    
    @st.cache_resource(show_spinner=False)
    def load(_self) -> bool:
        """
        Load vectorstore with caching
        Returns: True if successful, False otherwise
        """
        try:
            if not _self.vectorstore_path.exists():
                raise FileNotFoundError(
                    f"Vectorstore not found at {_self.vectorstore_path}"
                )
            
            # Initialize embeddings
            _self.embeddings = HuggingFaceEmbeddings(
                model_name=_self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Load FAISS vectorstore
            _self.vectorstore = FAISS.load_local(
                str(_self.vectorstore_path),
                _self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            print(f"✅ Vectorstore loaded: {_self.vectorstore.index.ntotal:,} chunks")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load vectorstore: {e}")
            return False
    
    def get_retriever(self, k: int = 3):
        """
        Get retriever with specified number of results
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Configured retriever
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not loaded. Call load() first.")
        
        if not self.retriever or self.retriever.search_kwargs.get('k') != k:
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            )
        
        return self.retriever
    
    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search vectorstore for relevant documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        try:
            retriever = self.get_retriever(k=k)
            docs = retriever.invoke(query)
            return docs
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def get_stats(self) -> dict:
        """Get vectorstore statistics"""
        if not self.vectorstore:
            return {"loaded": False, "total_chunks": 0}
        
        return {
            "loaded": True,
            "total_chunks": self.vectorstore.index.ntotal,
            "embedding_model": self.embedding_model,
            "dimension": self.vectorstore.index.d
        }
