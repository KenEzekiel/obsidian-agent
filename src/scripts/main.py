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
from datetime import datetime

from src.core.vault import VaultReader
from src.core.graph import GraphBuilder
from src.services.embedding import EmbeddingService
from src.services.vectorstore import VectorStoreClient
from src.agents.implementations.langgraph_agent import LangGraphAgentSystem

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
    
    vector_store_client = VectorStoreClient(
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
            
            # Add notes to vector store
            for note, text in zip(batch, batch_texts):
                vector_store_client.add_note(
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
        
        # Clear existing embeddings before processing
        console.print("[bold blue]Clearing existing embeddings...[/]")
        embedding_service = EmbeddingService(
            model_type=args.model_type,
            model_name=args.model_name,
            batch_size=args.batch_size
        )
        vector_store = VectorStoreClient(
            embedding_service=embedding_service,
            collection_name="embeddings",
            persist_directory=Path("./data/embeddings")
        )
        vector_store.clear_embeddings()
        console.print("[bold green]Embeddings cleared successfully[/]")
        
        # Generate embeddings
        process_with_embeddings(
            vault_data,
            args.output,
            args.model_type,
            args.model_name,
            args.batch_size
        )
        
        analyzer.graph_builder.write_graph()
        
        # Test embeddings with a sample note
        console.print("\n[bold blue]Testing embeddings with sample note...[/]")
        
        # Initialize embedding service and vector store
        embedding_service = EmbeddingService(
            model_type=args.model_type,
            model_name=args.model_name,
            batch_size=args.batch_size
        )
        vector_store = VectorStoreClient(
            embedding_service=embedding_service,
            collection_name="embeddings",
            persist_directory=Path("./data/embeddings")
        )
        
        # Get first note from vault data for testing
        test_note_path = "/Users/kenneth.ezekiel.suprantoni/Documents/Github/obsidian-agent/Mock Vault/05 - Ilmu/08.2 - Kriptografi/01.1 - Pengenalan Kriptografi.md"
        test_note_content = vault_data[test_note_path]["content"]
        
        # Get embeddings for test note
        test_note_embeddings = vector_store.collection.get(
            ids=[test_note_path],
            include=['embeddings']
        )
        if len(test_note_embeddings["embeddings"]) > 0:
            console.print(f"[green]Found embeddings of length: {len(test_note_embeddings['embeddings'][0])}[/] for note: {test_note_path}")
        else:
            console.print("[red]No embeddings found for test note[/]")
        
        console.print(f"[cyan]Testing with note:[/] {test_note_path}")
        console.print("[cyan]Finding similar notes...[/]")
        
        similar_notes = vector_store.find_similar_notes(
            note_id=test_note_path,
            top_n=3
        )
        
        if similar_notes:
            console.print("\n[green]Similar notes found:[/]")
            for note in similar_notes:
                console.print(f"\n[yellow]Note ID:[/] {note['note_id']}")
                console.print(f"[yellow]Similarity Score:[/] {note['similarity_score']:.3f}")
                console.print(f"[yellow]Preview:[/]\n{note['content'][:200]}...")
        else:
            console.print("\n[red]No similar notes found[/]")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/]")
        raise

def run_langgraph_agent():
    # Initialize services
    agent = LangGraphAgentSystem()

    # Process vault
    vault_path = "./Mock Vault"  # Update this path as needed
    results = agent.process_vault(vault_path)

    # Convert results to JSON-serializable format
    output = []
    for state in results:
        state_dict = state.model_dump()
        
        # Convert datetime objects to strings
        for link in state_dict["suggested_links"]:
            link["created_at"] = link["created_at"].isoformat()
            
        output.append(state_dict)

    # Create output directory if it doesn't exist
    output_dir = Path("./data")
    output_dir.mkdir(exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"agent_results_{timestamp}.json"

    # Write results to file
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results written to {output_file}")

if __name__ == "__main__":
    main()
