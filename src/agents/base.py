from typing import List, Dict, Optional, Protocol, TypedDict, Callable, Any
from pydantic import BaseModel, Field
from datetime import datetime
import traceback
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()

class NoteLink(BaseModel):
    source_note: str
    target_note: str
    similarity_score: float
    created_at: datetime = Field(default_factory=datetime.now)

class SimilarNote(TypedDict):
    note_id: str
    similarity_score: float
    content: str

class AgentState(BaseModel):
    """Base state model that all implementations should extend."""
    current_note_id: str
    processed_notes: List[str] = Field(default_factory=list)
    suggested_links: List[NoteLink] = Field(default_factory=list)
    topic_clusters: Dict[str, List[str]] = Field(default_factory=dict)
    context: Dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None

class AgentTool(BaseModel):
    """Base class for agent tools."""
    name: str
    description: str
    func: Callable[..., Any]

class VectorSimilarityTool(AgentTool):
    """Tool for finding similar notes using vector similarity search."""
    vectorstore: Any = Field(description="Vector store client for similarity search")
    
    def __init__(self, vectorstore_client: Any):
        super().__init__(
            name="find_similar_notes",
            description="Find similar notes using vector similarity search",
            func=self._find_similar_notes,
            vectorstore=vectorstore_client
        )

    def _find_similar_notes(self, note_id: str, top_n: int = 5, exclude_ids: Optional[List[str]] = None) -> List[SimilarNote]:
        """
        Find similar notes using vector similarity search.
        
        Args:
            note_id: The ID of the note to find similar notes for
            top_n: Number of similar notes to return
            exclude_ids: List of note IDs to exclude from results
            
        Returns:
            List of similar notes with their similarity scores and content
        """
        try:
            console.print("\n[bold blue]VectorSimilarityTool[/] 🔍 Starting search for note:", note_id)
            console.print("[dim]Using vectorstore:[/]", self.vectorstore)
            
            # First check if the note exists in the vector store
            try:
                note = self.vectorstore.collection.get(ids=[note_id], include=['embeddings'])
                console.print("[green]✓[/] Found note in vectorstore")
            except Exception as e:
                console.print("[red]✗[/] Error getting note:", str(e))
                return []
            
            console.print(f"[bold]🔎 Finding {top_n} similar notes...[/]")
            similar_notes = self.vectorstore.find_similar_notes(
                note_id=note_id,
                top_n=top_n,
                exclude_ids=exclude_ids or []
            )
            
            if not similar_notes:
                console.print("[yellow]ℹ️[/] No similar notes found for", note_id)
                return []
                
            # Convert to SimilarNote format
            formatted_notes = []
            
            # Create a table for similar notes
            table = Table(title="Similar Notes Found", show_header=True, header_style="bold magenta")
            table.add_column("Note ID", style="cyan")
            table.add_column("Score", justify="right", style="green")
            table.add_column("Preview", style="white")
            
            for note in similar_notes:
                try:
                    formatted_note = {
                        "note_id": note["note_id"],
                        "similarity_score": note["similarity_score"],
                        "content": note.get("content", "")
                    }
                    formatted_notes.append(formatted_note)
                    
                    # Add to table
                    table.add_row(
                        formatted_note['note_id'],
                        f"{formatted_note['similarity_score']:.3f}",
                        formatted_note['content'][:100] + "..."
                    )
                except KeyError as e:
                    console.print("[red]⚠️[/] Error formatting note:", str(e))
                    console.print("[dim]Note structure:[/]", note)
                    continue
            
            console.print(table)
            console.print(f"[green]✓[/] Successfully formatted {len(formatted_notes)} notes")
            return formatted_notes
            
        except Exception as e:
            console.print(Panel(
                f"[red]Error in find_similar_notes[/]\n"
                f"Error: {str(e)}\n"
                f"Type: {type(e)}\n"
                f"Traceback:\n{traceback.format_exc()}",
                title="Error",
                border_style="red"
            ))
            return []

class AgentSystem(Protocol):
    """Protocol defining the interface for agent systems."""
    
    def process_note(self, note_id: str, content: str) -> AgentState:
        """Process a single note through the workflow."""
        ...
    
    def process_vault(self, vault_path: str) -> List[AgentState]:
        """Process all notes in a vault."""
        ...

    def add_link(self, state: AgentState, source: str, target: str, score: float, reason: str):
        """Add a new link to the state."""
        state.suggested_links.append(
            NoteLink(
                source_note=source,
                target_note=target,
                similarity_score=score,
                reason=reason
            )
        )
    
    def add_to_cluster(self, state: AgentState, cluster_name: str, note: str):
        """Add a note to a topic cluster."""
        if cluster_name not in state.topic_clusters:
            state.topic_clusters[cluster_name] = []
        if note not in state.topic_clusters[cluster_name]:
            state.topic_clusters[cluster_name].append(note)
    
    def mark_processed(self, state: AgentState, note: str):
        """Mark a note as processed."""
        if note not in state.processed_notes:
            state.processed_notes.append(note) 