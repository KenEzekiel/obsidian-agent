"""
Embedding service for generating and managing text embeddings.
Uses ChromaDB for persistent storage of embeddings.
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from openai import RateLimitError, APIError
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import tiktoken

# Load environment variables
load_dotenv()

class EmbeddingService:
    """
    Service for generating text embeddings using various models.
    Supports both local models and OpenAI's embedding models.
    Uses ChromaDB for persistent storage of embeddings.
    """

    def __init__(
        self,
        model_type: str = "api",
        model_name: str = "text-embedding-3-small",
        persist_dir: str = "data/embeddings",
        batch_size: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_tokens: int = 8000  # Leave some buffer for safety
    ):
        """
        Initialize the embedding service.

        Args:
            model_type: Type of model to use ('local' or 'api')
            model_name: Name of the model to use
            persist_dir: Directory to persist embeddings
            batch_size: Number of texts to process in each batch
            max_retries: Maximum number of retries for API calls
            retry_delay: Base delay between retries in seconds
            max_tokens: Maximum number of tokens per text
        """
        self.model_type = model_type
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_tokens = max_tokens
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="embeddings",
            metadata={"model": model_name}
        )
        
        # Initialize model
        self._initialize_model()
        
        # Initialize tokenizer for API models
        if self.model_type == "api":
            self.tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

    def _initialize_model(self) -> None:
        """Initialize the embedding model based on type."""
        if self.model_type == "local":
            try:
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=str(self.persist_dir / "model_cache")
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to initialize local model '{self.model_name}'. "
                    f"Please ensure it's a valid Sentence Transformers model. Error: {str(e)}"
                )
        elif self.model_type == "api":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Please set it in your .env file or environment variables."
                )
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _handle_api_error(self, error: Exception, attempt: int) -> Tuple[bool, float]:
        """
        Handle API errors and determine if retry is needed.
        
        Returns:
            Tuple of (should_retry, delay_before_retry)
        """
        if isinstance(error, RateLimitError):
            # Exponential backoff for rate limits
            delay = self.retry_delay * (2 ** (attempt - 1))
            return True, delay
        elif isinstance(error, APIError):
            # Retry other API errors with linear backoff
            return attempt < self.max_retries, self.retry_delay
        return False, 0

    def _get_text_hash(self, file_path: str, text: str) -> str:
        """
        Generate a unique hash for the text using both file path and content.
        
        Args:
            file_path: Path of the file containing the text
            text: Text content to hash
            
        Returns:
            Unique hash string
        """
        # Combine file path and text for unique hashing
        combined = f"{file_path}:{text}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks that fit within the model's context window.
        Splits based on newlines to preserve document structure.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        if self.model_type != "api":
            return [text]  # No chunking needed for local models
            
        # Split text into paragraphs based on newlines
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            # Get token count for the paragraph
            paragraph_tokens = len(self.tokenizer.encode(paragraph))
            
            if current_length + paragraph_tokens > self.max_tokens:
                # Current chunk is full, start a new one
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = paragraph_tokens
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_tokens
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks

    def _average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """
        Average multiple embeddings into a single embedding.
        
        Args:
            embeddings: List of embeddings to average
            
        Returns:
            Averaged embedding
        """
        if not embeddings:
            return []
        return list(np.mean(embeddings, axis=0))

    def get_embedding(self, file_path: str, text: str) -> List[float]:
        """
        Get embedding for a single text.

        Args:
            file_path: Path of the file containing the text
            text: Text to generate embedding for

        Returns:
            List of floats representing the embedding
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")
            
        # Check if embedding exists in ChromaDB
        text_hash = self._get_text_hash(file_path, text)
        results = self.collection.get(
            ids=[text_hash],
            include=["embeddings", "metadatas"]
        )
        
        # Check if results exist and have embeddings
        if results and results["ids"] and len(results["embeddings"]) > 0:
            return results["embeddings"][0]
        
        # Split text into chunks if needed
        chunks = self._chunk_text(text)
        chunk_embeddings = []
        
        # Generate embeddings for each chunk
        for chunk in chunks:
            if self.model_type == "local":
                embedding = self.model.encode(chunk)
                chunk_embeddings.append(embedding.tolist())
            else:
                for attempt in range(1, self.max_retries + 1):
                    try:
                        response = self.client.embeddings.create(
                            model=self.model_name,
                            input=chunk
                        )
                        chunk_embeddings.append(response.data[0].embedding)
                        break
                    except Exception as e:
                        should_retry, delay = self._handle_api_error(e, attempt)
                        if not should_retry:
                            raise
                        if attempt < self.max_retries:
                            time.sleep(delay)
                        else:
                            raise Exception(f"Failed after {self.max_retries} attempts: {str(e)}")
        
        # Average the chunk embeddings
        result = self._average_embeddings(chunk_embeddings)
        
        # Store in ChromaDB
        self.collection.add(
            ids=[text_hash],
            embeddings=[result],
            metadatas=[{"text": text, "file_path": file_path, "num_chunks": len(chunks)}],
            documents=[text]
        )
        
        return result

    def get_embeddings_batch(self, file_paths: List[str], texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts.

        Args:
            file_paths: List of file paths
            texts: List of texts to generate embeddings for

        Returns:
            List of embeddings
        """
        if not texts:
            return []
            
        if len(file_paths) != len(texts):
            raise ValueError("Number of file paths must match number of texts")
            
        # Check which texts need new embeddings
        text_hashes = [self._get_text_hash(fp, text) for fp, text in zip(file_paths, texts)]
        results = self.collection.get(
            ids=text_hashes,
            include=["embeddings", "metadatas"]
        )
        
        # Create mapping of hash to index
        hash_to_idx = {h: i for i, h in enumerate(text_hashes)}
        
        # Initialize results array
        embeddings = [None] * len(texts)
        
        # Fill in existing embeddings
        if results and results["ids"]:
            for i, hash_id in enumerate(results["ids"]):
                if hash_id in hash_to_idx:
                    embeddings[hash_to_idx[hash_id]] = results["embeddings"][i]
        
        # Find texts that need new embeddings
        texts_to_embed = []
        indices_to_embed = []
        file_paths_to_embed = []
        for i, emb in enumerate(embeddings):
            if emb is None:
                texts_to_embed.append(texts[i])
                indices_to_embed.append(i)
                file_paths_to_embed.append(file_paths[i])
        
        if texts_to_embed:
            # Process each text individually to handle chunking
            new_embeddings = []
            for text, file_path in zip(texts_to_embed, file_paths_to_embed):
                embedding = self.get_embedding(file_path, text)
                new_embeddings.append(embedding)
            
            # Update results
            for i, emb in zip(indices_to_embed, new_embeddings):
                embeddings[i] = emb
        
        return embeddings

    def clear_embeddings(self) -> None:
        """Clear all stored embeddings."""
        self.collection.delete(where={})

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary containing model information
        """
        if self.model_type == "local":
            return {
                "type": "local",
                "name": self.model_name,
                "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                "max_sequence_length": self.model.max_seq_length
            }
        else:
            # Get model info from OpenAI
            models = self.client.models.list()
            model_info = next((m for m in models.data if m.id == self.model_name), None)
            
            if not model_info:
                raise ValueError(f"Model {self.model_name} not found in OpenAI's available models")
                
            return {
                "type": "api",
                "name": self.model_name,
                "provider": "openai",
                "embedding_dimension": 1536 if "3-small" in self.model_name else 3072,  # OpenAI's embedding dimensions
                "max_sequence_length": 8191,  # OpenAI's max context length
                "max_tokens_per_text": self.max_tokens
            }

def main():
    """Test the embedding service with a sample graph."""
    # Initialize service
    service = EmbeddingService(
        model_type="api",
        model_name="text-embedding-3-small",
        persist_dir="data/embeddings"
    )
    
    # Load sample graph
    with open("data/graph.json", "r") as f:
        sample_graph = json.load(f)
    
    # Test single embedding
    embedding = service.get_embedding("test.md", sample_graph["content"])
    print(f"Generated embedding of dimension: {len(embedding)}")
    
    # Test batch embedding
    file_paths = ["test1.md", "test2.md", "test3.md"]
    texts = [sample_graph["content"]] * 3
    embeddings = service.get_embeddings_batch(file_paths, texts)
    print(f"Generated {len(embeddings)} embeddings in batch")
    
    # Print model info
    print("\nModel Information:")
    for key, value in service.get_model_info().items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()