"""
Vector Store implementation for document retrieval.

This module provides vector storage and retrieval functionality
using ChromaDB with support for embeddings and similarity search.
"""

import os
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from uuid import UUID
from datetime import datetime
import json

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from loguru import logger

from ..models.document import Document, DocumentChunk, SearchQuery, SearchResult, DocumentStats
from config.settings import VectorDBConfig


class DocumentProcessor:
    """Document processing utilities."""
    
    @staticmethod
    def chunk_document(
        document: Document,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[DocumentChunk]:
        """
        Split document into chunks.
        
        Args:
            document: Document to chunk
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of document chunks
        """
        content = document.content
        chunks = []
        
        if len(content) <= chunk_size:
            # Document is small enough to be a single chunk
            chunk_metadata = {
                "chunk_index": 0,
                "start_char": 0,
                "end_char": len(content),
                "page_number": 1,
                "overlap_with_previous": 0,
                "overlap_with_next": 0
            }
            
            chunk = DocumentChunk(
                document_id=document.id,
                content=content,
                chunk_metadata=chunk_metadata,
                document_metadata=document.metadata
            )
            chunks.append(chunk)
            return chunks
        
        # Split into overlapping chunks
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            
            # Try to break at word boundaries
            if end < len(content):
                # Look for the last space within the chunk
                last_space = content.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                overlap_prev = chunk_overlap if chunk_index > 0 else 0
                overlap_next = chunk_overlap if end < len(content) else 0
                
                chunk_metadata = {
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    "page_number": None,  # Would be calculated for PDFs
                    "overlap_with_previous": overlap_prev,
                    "overlap_with_next": overlap_next
                }
                
                chunk = DocumentChunk(
                    document_id=document.id,
                    content=chunk_content,
                    chunk_metadata=chunk_metadata,
                    document_metadata=document.metadata
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = end - chunk_overlap
            if start >= len(content):
                break
        
        return chunks
    
    @staticmethod
    def extract_text_features(text: str) -> Dict[str, Any]:
        """
        Extract features from text for enhanced retrieval.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted features
        """
        words = text.split()
        sentences = text.split('.')
        
        features = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "has_numbers": any(char.isdigit() for char in text),
            "has_urls": "http" in text.lower() or "www" in text.lower(),
            "has_emails": "@" in text and "." in text,
            "language_indicators": {
                "english": sum(1 for word in ["the", "and", "or", "but", "in", "on", "at"] if word in text.lower()),
                "technical": sum(1 for word in ["algorithm", "function", "method", "system"] if word in text.lower())
            }
        }
        
        return features


class VectorStore:
    """Vector store for document retrieval using ChromaDB."""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.processor = DocumentProcessor()
        
        # Initialize the store
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize the vector store."""
        try:
            # Create persistence directory
            os.makedirs(self.config.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name
                )
                logger.info(f"Loaded existing collection: {self.config.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": self.config.distance_metric}
                )
                logger.info(f"Created new collection: {self.config.collection_name}")
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            logger.info(f"Initialized embedding model: {self.config.embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Addition results
        """
        if not self.collection or not self.embedding_model:
            await self._initialize()
        
        try:
            total_chunks = 0
            failed_documents = []
            
            for document in documents:
                try:
                    # Process document into chunks
                    chunks = self.processor.chunk_document(
                        document,
                        chunk_size=500,  # Could be configurable
                        chunk_overlap=50
                    )
                    
                    # Generate embeddings for chunks
                    chunk_texts = [chunk.content for chunk in chunks]
                    embeddings = self.embedding_model.encode(chunk_texts).tolist()
                    
                    # Prepare data for ChromaDB
                    ids = [str(chunk.id) for chunk in chunks]
                    metadatas = []
                    
                    for chunk in chunks:
                        metadata = {
                            "document_id": str(chunk.document_id),
                            "chunk_index": chunk.chunk_metadata.chunk_index,
                            "start_char": chunk.chunk_metadata.start_char,
                            "end_char": chunk.chunk_metadata.end_char,
                            "word_count": chunk.get_word_count(),
                            "char_count": chunk.get_char_count(),
                            "document_title": chunk.document_metadata.title or "",
                            "document_author": chunk.document_metadata.author or "",
                            "document_type": str(document.document_type),
                            "created_at": chunk.created_at.isoformat(),
                            "tags": json.dumps(chunk.document_metadata.tags)
                        }
                        
                        # Add text features
                        features = self.processor.extract_text_features(chunk.content)
                        metadata.update({f"feature_{k}": v for k, v in features.items() if isinstance(v, (int, float, bool, str))})
                        
                        metadatas.append(metadata)
                    
                    # Add to collection
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=chunk_texts,
                        metadatas=metadatas
                    )
                    
                    total_chunks += len(chunks)
                    logger.info(f"Added document {document.id} with {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"Failed to add document {document.id}: {e}")
                    failed_documents.append(str(document.id))
            
            return {
                "added_documents": len(documents) - len(failed_documents),
                "failed_documents": failed_documents,
                "total_chunks": total_chunks
            }
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {
                "added_documents": 0,
                "failed_documents": [str(doc.id) for doc in documents],
                "total_chunks": 0,
                "error": str(e)
            }
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Search for relevant document chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Optional metadata filters
            
        Returns:
            List of relevant document chunks
        """
        if not self.collection or not self.embedding_model:
            await self._initialize()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert results to DocumentChunk objects
            chunks = []
            
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    if similarity >= threshold:
                        metadata = results["metadatas"][0][i]
                        content = results["documents"][0][i]
                        
                        # Reconstruct chunk metadata
                        chunk_metadata = {
                            "chunk_index": metadata.get("chunk_index", 0),
                            "start_char": metadata.get("start_char", 0),
                            "end_char": metadata.get("end_char", 0),
                            "page_number": metadata.get("page_number"),
                            "overlap_with_previous": 0,
                            "overlap_with_next": 0
                        }
                        
                        # Reconstruct document metadata
                        document_metadata = {
                            "title": metadata.get("document_title"),
                            "author": metadata.get("document_author"),
                            "tags": json.loads(metadata.get("tags", "[]"))
                        }
                        
                        chunk = DocumentChunk(
                            id=UUID(chunk_id),
                            document_id=UUID(metadata["document_id"]),
                            content=content,
                            chunk_metadata=chunk_metadata,
                            document_metadata=document_metadata,
                            similarity_score=similarity
                        )
                        
                        chunks.append(chunk)
            
            logger.info(f"Found {len(chunks)} relevant chunks for query: {query[:50]}...")
            return chunks
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def delete_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            Deletion results
        """
        if not self.collection:
            await self._initialize()
        
        try:
            deleted_count = 0
            
            for doc_id in document_ids:
                # Find all chunks for this document
                results = self.collection.get(
                    where={"document_id": doc_id},
                    include=["metadatas"]
                )
                
                if results["ids"]:
                    # Delete all chunks for this document
                    self.collection.delete(ids=results["ids"])
                    deleted_count += len(results["ids"])
                    logger.info(f"Deleted {len(results['ids'])} chunks for document {doc_id}")
            
            return {
                "deleted_documents": len(document_ids),
                "deleted_chunks": deleted_count
            }
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return {
                "deleted_documents": 0,
                "deleted_chunks": 0,
                "error": str(e)
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Store statistics
        """
        if not self.collection:
            await self._initialize()
        
        try:
            # Get collection info
            collection_count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_results = self.collection.get(
                limit=min(1000, collection_count),
                include=["metadatas"]
            )
            
            stats = {
                "total_chunks": collection_count,
                "total_documents": 0,
                "document_types": {},
                "average_chunk_size": 0,
                "embedding_model": self.config.embedding_model,
                "collection_name": self.config.collection_name
            }
            
            if sample_results["metadatas"]:
                # Analyze metadata
                document_ids = set()
                doc_types = {}
                total_chars = 0
                
                for metadata in sample_results["metadatas"]:
                    document_ids.add(metadata.get("document_id"))
                    
                    doc_type = metadata.get("document_type", "unknown")
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    
                    total_chars += metadata.get("char_count", 0)
                
                stats["total_documents"] = len(document_ids)
                stats["document_types"] = doc_types
                stats["average_chunk_size"] = total_chars / len(sample_results["metadatas"]) if sample_results["metadatas"] else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    async def similarity_search_with_score(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search with similarity scores.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Optional filters
            
        Returns:
            List of (chunk, score) tuples
        """
        chunks = await self.search(query, top_k, threshold=0.0, filters=filters)
        return [(chunk, chunk.similarity_score or 0.0) for chunk in chunks]
    
    async def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of document chunks
        """
        if not self.collection:
            await self._initialize()
        
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"]
            )
            
            chunks = []
            if results["ids"]:
                for i, chunk_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    content = results["documents"][i]
                    
                    chunk_metadata = {
                        "chunk_index": metadata.get("chunk_index", 0),
                        "start_char": metadata.get("start_char", 0),
                        "end_char": metadata.get("end_char", 0),
                        "page_number": metadata.get("page_number"),
                        "overlap_with_previous": 0,
                        "overlap_with_next": 0
                    }
                    
                    document_metadata = {
                        "title": metadata.get("document_title"),
                        "author": metadata.get("document_author"),
                        "tags": json.loads(metadata.get("tags", "[]"))
                    }
                    
                    chunk = DocumentChunk(
                        id=UUID(chunk_id),
                        document_id=UUID(document_id),
                        content=content,
                        chunk_metadata=chunk_metadata,
                        document_metadata=document_metadata
                    )
                    
                    chunks.append(chunk)
            
            # Sort by chunk index
            chunks.sort(key=lambda x: x.chunk_metadata.chunk_index)
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            return []
    
    async def reset_collection(self) -> bool:
        """
        Reset the collection (delete all data).
        
        Returns:
            True if successful
        """
        try:
            if self.collection:
                self.client.delete_collection(self.config.collection_name)
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": self.config.distance_metric}
                )
                logger.info(f"Reset collection: {self.config.collection_name}")
                return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False

