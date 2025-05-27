from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agents.base import AgentSystem, AgentState, NoteLink, VectorSimilarityTool, console
from src.services.vectorstore import VectorStoreClient
from src.services.embedding import EmbeddingService
from src.core.vault import VaultReader, VaultWriter
from src.config.agent_config import AgentConfig, get_default_config



class AutoGenAgentSystem(AgentSystem):
    """AutoGen implementation of the agent system using multiple specialized agents."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the AutoGen agent system.
        
        Args:
            config: Optional configuration. If not provided, uses default configuration.
        """
        self.config = config or get_default_config()
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService(
            model_type=self.config.embedding_config.model_type,
            model_name=self.config.embedding_config.model_name,
            batch_size=self.config.embedding_config.batch_size
        )
        
        # Initialize vector store
        self.vectorstore = VectorStoreClient(
            embedding_service=self.embedding_service,
            collection_name=self.config.vectorstore_config.collection_name,
            persist_directory=self.config.vectorstore_config.persist_directory
        )
        
        self.similarity_tool = VectorSimilarityTool(self.vectorstore)

    def _escape_path(self, path: str) -> str:
        """Escape a file path for Rich markup."""
        return str(path).replace("[", "\\[").replace("]", "\\]")

    def process_note(self, note_id: str, content: str, vault_path: str) -> AgentState:
        """Process a single note through the workflow."""
        console.print("\n[bold blue]AutoGen[/] 📝 Processing note:", note_id)
        
        # Initialize vault writer
        vault_writer = VaultWriter(Path(vault_path))
        
        # Check if note exists in vector store before adding
        try:
            console.print("\n[bold]🔍 Checking vector store...[/]")
            existing_note = self.vectorstore.collection.get(
                ids=[note_id],
                include=['embeddings', 'metadatas', 'documents']
            )
            
            if existing_note["embeddings"] is None or len(existing_note["embeddings"]) == 0:
                console.print("\n[bold]📥 Adding to vector store...[/]")
                self.vectorstore.add_note(note_id, content)
                console.print("[green]✓[/] Added to vector store")
            else:
                console.print("[yellow]ℹ️[/] Already in vector store")
                
        except Exception as e:
            console.print(Panel(
                f"[red]Vector store error[/]\n"
                f"Error: {str(e)}",
                title="Error",
                border_style="red"
            ))
            return AgentState(
                current_note_id=note_id,
                error=f"Vector store error: {str(e)}"
            )
        
        # Initialize state
        state = AgentState(
            current_note_id=note_id,
            context={"current_note_content": content}
        )
        
        try:
            # Step 1: Categorize note
            console.print("\n[bold]📑 Categorizing note...[/]")
            # TODO: Implement categorization logic
            
            # Step 2: Analyze content
            console.print("\n[bold]📝 Analyzing content...[/]")
            # TODO: Implement content analysis logic
            
            # Step 3: Find similar notes
            similar_notes = self.similarity_tool.func(
                note_id=note_id,
                top_n=5,
                exclude_ids=[note_id]
            )
            state.context["similar_notes"] = similar_notes
            
            # Step 4: Generate link suggestions
            if similar_notes:
                console.print("\n[bold]🔗 Suggesting links...[/]")
                # TODO: Implement link suggestion logic
            
            # Step 5: Update clusters
            # TODO: Implement cluster updates
            
            # Step 6: Write suggestions and tags
            if state.suggested_links or state.context.get("topics") or state.context.get("note_categories"):
                console.print("\n[bold]📝 Writing suggestions and tags...[/]")
                
                # Format links for VaultWriter
                links = [
                    {
                        "target": link.target_note,
                        "score": link.similarity_score
                    }
                    for link in state.suggested_links
                ]
                
                # Get topics and categories from context
                topics = state.context.get("topics", [])
                categories = state.context.get("note_categories", [])
                
                # Write suggestions and tags
                try:
                    vault_writer.write_suggestions(
                        note_path=Path(note_id),
                        links=links,
                        topics=topics,
                        categories=categories
                    )
                    console.print("[green]✓[/] Successfully wrote suggestions and tags")
                    
                except Exception as e:
                    console.print(Panel(
                        f"[red]Error writing suggestions[/]\n"
                        f"Error: {str(e)}",
                        title="Error",
                        border_style="red"
                    ))
            
            # Mark as processed
            self.mark_processed(state, note_id)
            console.print("[green]✓[/] Workflow completed")
            
        except Exception as e:
            console.print(Panel(
                f"[red]Error in workflow[/]\n"
                f"Error: {str(e)}",
                title="Error",
                border_style="red"
            ))
            state.error = str(e)
        
        return state

    def process_vault(self, vault_path: str) -> List[AgentState]:
        """Process all notes in a vault."""
        vault = VaultReader(Path(vault_path))
        notes = vault.get_all_notes()
        states = []
        
        console.print(Panel(
            f"Path: {self._escape_path(vault_path)}\n"
            f"Total notes: {len(notes)}",
            title="📚 Processing Vault",
            border_style="blue"
        ))
        
        for note in notes[:1]:
            try:
                content = vault.read_note(note)
                console.print(Panel(
                    f"Note: {self._escape_path(note)}\n"
                    f"Preview: {content[:100]}...",
                    title="[bold blue]📝 Processing Note[/]",
                    border_style="blue"
                ))
                state = self.process_note(str(note), content, vault_path)
                states.append(state)
                
                # Print summary
                table = Table(title="Note Summary", show_header=False)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="white")
                
                table.add_row("Links", str(len(state.suggested_links)))
                table.add_row("Clusters", str(len(state.topic_clusters)))
                
                console.print(table)
                
            except Exception as e:
                error_msg = f"Error processing note: {self._escape_path(str(note))}\nError: {str(e)}"
                console.print(Panel(
                    error_msg,
                    title="Error",
                    border_style="red"
                ))
                continue
                
        return states

def run_agent():
    """Run the AutoGen agent on a vault."""
    # Initialize services
    agent = AutoGenAgentSystem()

    # Process vault
    vault_path = "/Users/kenneth.ezekiel.suprantoni/Documents/Github/obsidian-agent/Mock Vault" 
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
    run_agent()
