"""
Vector store client for finding similar notes using vector embeddings.
Uses ChromaDB for efficient storage and retrieval.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from datetime import datetime

from .embedding import EmbeddingService


class VectorStoreClient:
    """
    Vector store client for finding similar notes using vector embeddings.
    Uses ChromaDB for efficient storage and retrieval.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        collection_name: str = "embeddings",
        persist_directory: Optional[Path] = None
    ):
        """
        Initialize the vector store client.

        Args:
            embedding_service: Instance of EmbeddingService for generating embeddings
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.embedding_service = embedding_service
        self.collection_name = collection_name
        self.persist_directory = persist_directory or Path.home() / ".cache" / "obsidian-agent" / "embeddings"
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"model": self.embedding_service.model_name}
        )

    def add_note(
        self,
        note_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a note to the vector store.

        Args:
            note_id: Unique identifier for the note
            content: Text content of the note
            metadata: Additional metadata about the note
        """
        # Generate embedding
        try:
            print(f"\n[VectorStore] 🔄 Generating embedding for note: {note_id}")
            embedding = self.embedding_service.get_embedding(content)
            
            
            # Ensure embedding is in the correct format
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                raise ValueError(f"Unexpected embedding type: {type(embedding)}")
                
            # Validate embedding
            if not embedding or len(embedding) == 0:
                raise ValueError("Generated embedding is empty")
                
            print(f"[VectorStore] ✅ Generated embedding of length {len(embedding)}")
            
        except Exception as e:
            print(f"[VectorStore] ❌ Error generating embedding: {str(e)}")
            raise
        
        # Prepare metadata
        note_metadata = {
            "text": content,
            "file_path": note_id,
            "last_updated": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        # Ensure all metadata values are of supported types
        for key, value in note_metadata.items():
            if isinstance(value, (list, set, tuple)):
                note_metadata[key] = ", ".join(str(v) for v in value)
            elif not isinstance(value, (str, int, float, bool, type(None))):
                note_metadata[key] = str(value)
        
        # Add to collection
        try:
            print(f"\n[VectorStore] 🔄 Adding note to collection: {note_id}")
            
            # First try to delete if exists
            try:
                self.collection.delete(ids=[note_id])
                print("[VectorStore] ℹ️ Deleted existing note if present")
            except Exception as e:
                print(f"[VectorStore] ℹ️ No existing note to delete: {str(e)}")
                
            # Add the note with embedding
            print("[VectorStore] 📥 Adding note with embedding...")
            
            self.collection.add(
                ids=[note_id],
                embeddings=[embedding],
                metadatas=[note_metadata],
                documents=[content]
            )
            
            # Verify the note was added correctly
            print("[VectorStore] 🔍 Verifying note addition...")
            result = self.collection.get(
                ids=[note_id],
                include=['embeddings', 'metadatas', 'documents']
            )
            
            print(f"[VectorStore] 📊 Verification results:")
            print(f"  IDs: {result['ids']}")
            print(f"  Has embeddings: {'Yes' if result['embeddings'] is not None else 'No'}")
            print(f"  Embedding length: {len(result['embeddings'][0]) if result['embeddings'] is not None else 0}")
            
            # Check if embeddings exist and are not empty
            if len(result["ids"]) == 0:
                raise ValueError(f"No embeddings found for note {note_id}")
                
            if len(result["embeddings"]) == 0:
                raise ValueError(f"Empty embeddings list for note {note_id}")
                
            stored_embedding = result["embeddings"][0]
            if stored_embedding is None:
                raise ValueError(f"Stored embedding is None for note {note_id}")
                
            # Verify embedding length matches
            if len(stored_embedding) != len(embedding):
                raise ValueError(f"Stored embedding length ({len(stored_embedding)}) does not match generated embedding length ({len(embedding)})")
                
            print(f"[VectorStore] ✅ Successfully added note {note_id} with embedding")
            
        except Exception as e:
            print(f"[VectorStore] ❌ Error adding note to collection: {str(e)}")
            raise

    def update_note(
        self,
        note_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update an existing note in the vector store.

        Args:
            note_id: Unique identifier for the note
            content: Updated text content
            metadata: Updated metadata
        """
        # Remove existing note
        self.collection.delete(ids=[note_id])
        
        # Add updated note
        self.add_note(note_id, content, metadata)

    def find_similar_notes(
        self,
        note_id: str,
        top_n: int = 5,
        exclude_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find notes similar to the given note.

        Args:
            note_id: ID of the note to find similar notes for
            top_n: Number of similar notes to return
            exclude_ids: List of note IDs to exclude from results

        Returns:
            List of dictionaries containing similar notes and their metadata
        """
        # Get the note's embedding
        try:
            result = self.collection.get(ids=[note_id], include=['embeddings'])
            
            if len(result["ids"]) == 0:
                print(f"Note {note_id} not found in collection")
                return []
                
            if len(result["embeddings"]) == 0:
                print(f"No embeddings found for note {note_id}")
                return []
                
            # Query similar notes
            query_result = self.collection.query(
                query_embeddings=[result["embeddings"][0]],
                n_results=top_n + 1,  # +1 because the note itself will be in results
                where={"file_path": {"$ne": note_id}} if exclude_ids is None else {
                    "file_path": {"$nin": exclude_ids + [note_id]}
                }
            )
            
            # Format results
            similar_notes = []
            for i, (id_, distance, metadata, content) in enumerate(zip(
                query_result["ids"][0],
                query_result["distances"][0],
                query_result["metadatas"][0],
                query_result["documents"][0]
            )):
                similar_notes.append({
                    "note_id": id_,
                    "similarity_score": 1 - distance,  # Convert distance to similarity
                    "metadata": metadata,
                    "content": content
                })
            
            return similar_notes
            
        except Exception as e:
            print(f"Error in find_similar_notes: {str(e)}")
            return []

    def get_note_metadata(self, note_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific note.

        Args:
            note_id: ID of the note

        Returns:
            Dictionary containing note metadata, or None if note not found
        """
        result = self.collection.get(ids=[note_id])
        if not result["ids"]:
            return None
        return result["metadatas"][0]

    def delete_note(self, note_id: str) -> None:
        """
        Delete a note from the vector store.

        Args:
            note_id: ID of the note to delete
        """
        self.collection.delete(ids=[note_id])

    def clear_embeddings(self) -> None:
        """Clear all stored embeddings."""
        try:
            # Get all IDs first
            result = self.collection.get()
            if result["ids"]:
                # Delete using IDs
                self.collection.delete(ids=result["ids"])
            print("Successfully cleared all embeddings")
        except Exception as e:
            print(f"Error clearing embeddings: {str(e)}")
            raise

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embeddings
        """
        return self.embedding_service.get_embeddings_batch(texts)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary containing model information
        """
        return self.embedding_service.get_model_info()


if __name__ == "__main__":
    # Example usage
    embedding_service = EmbeddingService()
    vector_store_client = VectorStoreClient(embedding_service)
    
    # Add some test notes
    test_notes = [
        ("note1", "This is a test note about machine learning."),
        ("note2", "Machine learning is a subset of artificial intelligence."),
        ("note3", "Python is a popular programming language."),
        ("note4", "Data science involves statistics and programming."),
        ("note5", "Deep learning is a type of machine learning.")
    ]
    
    for note_id, content in test_notes:
        vector_store_client.add_note(note_id, content)
    
    # Find similar notes
    similar = vector_store_client.find_similar_notes("note1", top_n=2)
    print("\nSimilar notes to note1:")
    for note in similar:
        print(f"ID: {note['note_id']}")
        print(f"Content: {note['content']}")
        print(f"Similarity: {note['similarity_score']:.3f}")
        print() 