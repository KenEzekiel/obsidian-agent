"""
Main entry point for the Obsidian Agent application.
Handles vault analysis, graph building, and embedding generation.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from rich.console import Console
from rich.progress import Progress

from core.vault import VaultReader
from core.graph import GraphBuilder
from services.embedding import EmbeddingService
from services.vectorstore import VectorStoreClient

console = Console()

class ObsidianAnalyzer:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.vault_reader = VaultReader()
        self.graph_builder = GraphBuilder()
        
    def get_markdown_files(self) -> List[Path]:
        """Recursively get all markdown files in the vault"""
        markdown_files = []
        for file in self.vault_path.rglob("*.md"):
            markdown_files.append(file)
        return markdown_files
    
    def extract_content_and_links(self) -> Dict[str, Dict[str, str | List[str]]]:
        """
        Extract content and wikilinks from all markdown files.
        Returns a dictionary with note names as keys and their content and links as values.
        """
        vault_data = {}
        for file in self.get_markdown_files():
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract links
            target_links = self.vault_reader.extract_wikilinks(content)
            
            # Store content and links
            vault_data[str(file)] = {
                "content": content,
                "links": list(target_links)  # Convert set to list
            }
            
        return vault_data
    
    def analyze(self) -> Dict[str, Dict[str, str | List[str]]]:
        """Main analysis pipeline"""
        # Extract content and links from vault
        vault_data = self.extract_content_and_links()
        
        # Build graph structure with content and links
        self.graph_builder.build_graph(vault_data)
        self.graph_builder.write_graph()
        
        return vault_data

def process_with_embeddings(
    vault_data: Dict[str, Dict[str, str | List[str]]],
    output_path: str,
    model_type: str = "api",
    model_name: str = "text-embedding-3-small",
    batch_size: int = 32
):
    """
    Process vault data and generate embeddings for all notes.
    
    Args:
        vault_data: Dictionary containing note content and links
        output_path: Path to save the embeddings
        model_type: Type of embedding model to use ('local' or 'api')
        model_name: Name of the model to use
        batch_size: Number of texts to process in each batch
    """
    console.print("[bold blue]Initializing services...[/]")
    embedding_service = EmbeddingService(
        model_type=model_type,
        model_name=model_name,
        batch_size=batch_size
    )
    
    similarity_service = SimilaritySearchService(
        embedding_service=embedding_service,
        collection_name="embeddings",  # Use same collection name as before
        persist_directory=Path(output_path).parent / "embeddings"  # Use same directory structure
    )
    
    # Get all notes
    notes = list(vault_data.keys())
    total_notes = len(notes)
    
    console.print(f"[bold green]Processing {total_notes} notes...[/]")
    
    # Process notes in batches
    with Progress() as progress:
        task = progress.add_task("[cyan]Generating embeddings...", total=total_notes)
        
        for i in range(0, total_notes, batch_size):
            batch = notes[i:i + batch_size]
            # Create text for embedding by combining content and links
            batch_texts = [
                f"{vault_data[note]['content']}\nLinks: {', '.join(vault_data[note]['links'])}"
                for note in batch
            ]
            
            # Add notes to similarity service
            for note, text in zip(batch, batch_texts):
                similarity_service.add_note(
                    note_id=note,
                    content=text,
                    metadata={
                        "text": text,
                        "file_path": note,
                        "links": ", ".join(vault_data[note]["links"]),  # Convert list to comma-separated string
                        "model": model_name
                    }
                )
            
            progress.update(task, advance=len(batch))
    
    # Save embeddings info
    console.print("[bold blue]Saving embeddings info...[/]")
    embeddings_info = {
        "model_type": model_type,
        "model_name": model_name,
        "total_notes": total_notes,
        "model_info": embedding_service.get_model_info()
    }
    
    with open(output_path, 'w') as f:
        json.dump(embeddings_info, f, indent=2)
    console.print(f"[bold green]Embeddings info saved to {output_path}[/]")
    
    # Print model info
    console.print("\n[bold]Model Information:[/]")
    for key, value in embeddings_info["model_info"].items():
        console.print(f"{key}: {value}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Obsidian Agent - Vault Analysis and Embedding Generation")
    parser.add_argument(
        "vault_path",
        help="Path to Obsidian vault directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/embeddings.json",
        help="Path to save the embeddings info"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["local", "api"],
        default="api",
        help="Type of embedding model to use"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="text-embedding-3-small",
        help="Name of the model to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of texts to process in each batch"
    )
    
    args = parser.parse_args()
    
    try:
        # Analyze vault and get content with links
        console.print("[bold blue]Analyzing vault...[/]")
        analyzer = ObsidianAnalyzer(args.vault_path)
        vault_data = analyzer.analyze()
        
        # Generate embeddings
        process_with_embeddings(
            vault_data,
            args.output,
            args.model_type,
            args.model_name,
            args.batch_size
        )
        
        analyzer.graph_builder.write_graph()
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        raise

if __name__ == "__main__":
    main()
