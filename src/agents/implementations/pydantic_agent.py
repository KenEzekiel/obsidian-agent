from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from src.agents.base import AgentSystem, AgentState, NoteLink, SimilarNote, VectorSimilarityTool, console
from src.services.vectorstore import VectorStoreClient
from src.services.embedding import EmbeddingService
from src.core.vault import VaultReader, VaultWriter
from src.config.agent_config import AgentConfig, get_default_config

class PydanticAgentSystem(AgentSystem):
    """Pydantic implementation of the agent system."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Pydantic agent system.
        
        Args:
            config: Optional configuration. If not provided, uses default configuration.
        """
        self.config = config or get_default_config()
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_config.model,
            temperature=self.config.llm_config.temperature
        )
        
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

    def _categorize_note(self, content: str) -> List[str]:
        """Categorize the note based on its utility and purpose."""
        console.print("\n[bold blue]Pydantic[/] 📑 Categorizing note")
        
        try:
            system_message = """You are an expert at analyzing notes and categorizing them based on their utility and purpose.
            Categorize the note into one or more of these categories:
            - Knowledge: Educational content, concepts, theories, explanations
            - Reference: Lists, resources, links, tools
            - Project: Project plans, requirements, documentation
            - Reminder: Tasks, todos, deadlines, important dates
            - Journal: Personal reflections, thoughts, experiences
            - Decision: Decision records, reasoning, trade-offs
            - Template: Reusable structures or formats
            - Archive: Historical information, completed items
            
            Return the categories as a comma-separated list, maximum 3 categories."""

            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=content)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse categories
            categories = [cat.strip() for cat in response.content.split(",")]
            
            # Display results
            console.print(Panel(
                "\n".join(f"• {cat}" for cat in categories),
                title="Note Categories",
                border_style="green"
            ))
            
            console.print("[green]✓[/] Successfully categorized note")
            return categories
            
        except Exception as e:
            console.print(Panel(
                f"[red]Error in categorize_note[/]\n"
                f"Error: {str(e)}",
                title="Error",
                border_style="red"
            ))
            return []

    def _analyze_content(self, content: str) -> List[str]:
        """Analyze note content and identify key topics."""
        console.print("\n[bold blue]Pydantic[/] 📝 Analyzing content")
        
        try:
            system_message = "You are an expert at analyzing note content and identifying key topics. Return the topics as a comma-separated list. Every topic should be like a title that has clear context on the domain of the note, e.g. Definition of AI, Goals of AI, etc."
            
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=content)
            ]
            
            response = self.llm.invoke(messages)
            
            # Convert comma-separated string to array
            topics = [topic.strip() for topic in response.content.split(",")]
            
            # Print identified topics
            console.print(Panel(
                "\n".join(f"• {topic}" for topic in topics),
                title="Identified Topics",
                border_style="blue"
            ))
            
            console.print("[green]✓[/] Successfully analyzed content and identified topics")
            return topics
            
        except Exception as e:
            console.print(Panel(
                f"[red]Error in analyze_content[/]\n"
                f"Error: {str(e)}",
                title="Error",
                border_style="red"
            ))
            return []

    def _summarize_content(self, content: str, max_length: int = 300) -> str:
        """Summarize content using LLM."""
        try:
            system_message = f"""You are an expert at summarizing content. Create a concise summary that captures the key points and main ideas. Make the summary should be comprehensive and coherent with each other.
            The summary should be no longer than {max_length} characters.
            Focus on the most important information that would be relevant for making connections with other notes."""
            
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=content)
            ]
            
            response = self.llm.invoke(messages)
            summary = response.content.strip()
            
            # Display the summary result
            console.print(Panel(
                self._escape_path(summary),
                title="Content Summary",
                border_style="blue"
            ))
            return summary
            
        except Exception as e:
            console.print(Panel(
                f"[red]Error in summarize_content[/]\n"
                f"Error: {str(e)}",
                title="Error",
                border_style="red"
            ))
            # Fallback to simple truncation if summarization fails
            return content[:max_length] + "..." if len(content) > max_length else content

    def _suggest_links(self, current_note_id: str, current_topics: List[str], current_content: str, similar_notes: List[Dict]) -> List[Dict]:
        """Suggest meaningful connections between notes using LLM."""
        console.print("\n[bold blue]Pydantic[/] 🔗 Suggesting links for note:", self._escape_path(current_note_id))
        
        try:
            system_message = """You are an expert at suggesting meaningful connections between notes. You answer in this JSON FORMAT: 
            {
                "links": [
                    {
                        "note_id": "note_id",
                        "confidence_score": "confidence_score"
                    }
                ]
            }
            
            sorted by confidence_score in descending order
            """
            
            human_message = f"""Given the current note and similar notes, suggest meaningful links.

            Current note:
            {current_note_id}
            Topics: {current_topics}
            Summary: {current_content}

            Similar notes:
            {"\n\n".join([
                f"Note: {note['note_id']}\n"
                f"Similarity: {note['similarity_score']:.3f}\n"
                f"Summary: {note['content_summary']}"
                for note in similar_notes
            ])}
            """
            
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=human_message)
            ]
            
            response = self.llm.invoke(messages)
            
            console.print(Panel(
                response.content,
                title="🤖 LLM Response",
                border_style="blue"
            ))
            
            # Parse LLM response
            content = response.content.strip()
            if content.startswith("```"):
                content = "\n".join(content.split("\n")[1:-1])
            
            response_data = json.loads(content)
            suggested_links = response_data.get("links", [])
            
            if suggested_links:
                for link in suggested_links:
                    try:
                        console.print("\n[bold]📎 Adding suggested link:[/]")
                        console.print(f"  From: [cyan]{self._escape_path(current_note_id)}[/]")
                        console.print(f"  To: [cyan]{self._escape_path(link['note_id'])}[/]") 
                        console.print(f"  Confidence: [green]{link['confidence_score']}[/]")
                        
                        if float(link["confidence_score"]) >= 0.5:
                            console.print("[yellow]ℹ️[/] Link confidence meets threshold (>= 0.5), proceeding...")
                        else:
                            console.print("[yellow]ℹ️[/] Link confidence below threshold (< 0.5), skipping...")
                            continue
                            
                        return suggested_links
                    except (KeyError, ValueError) as e:
                        console.print(Panel(
                            f"[red]Error adding individual link[/]\nError: {str(e)}",
                            title="Error",
                            border_style="red"
                        ))
            else:
                console.print("[yellow]ℹ️[/] No links suggested by LLM")
                return []
                
        except json.JSONDecodeError as e:
            console.print(Panel(
                f"[red]Error parsing LLM response[/]\nError: {str(e)}\nContent: {content}",
                title="Error", 
                border_style="red"
            ))
            return []
        except Exception as e:
            console.print(Panel(
                f"[red]Error in suggest_links[/]\nError: {str(e)}",
                title="Error",
                border_style="red"
            ))
            return []

    def _escape_path(self, path: str) -> str:
        """Escape a file path for Rich markup."""
        return str(path).replace("[", "\\[").replace("]", "\\]")

    def process_note(self, note_id: str, content: str, vault_path: str) -> AgentState:
        """Process a single note through the workflow."""
        console.print("\n[bold blue]Pydantic[/] 📝 Processing note:", note_id)
        
        # Initialize vault writer
        vault_writer = VaultWriter(Path(vault_path))
        
        # Check if note exists in vector store before adding
        try:
            console.print("\n[bold]🔍 Checking vector store...[/]")
            existing_note = self.vectorstore.collection.get(
                ids=[note_id],
                include=['embeddings', 'metadatas', 'documents']
            )
            
            table = Table(title="Vector Store Status", show_header=False)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("IDs", str(existing_note['ids']))
            table.add_row("Has embeddings", "Yes" if existing_note['embeddings'] is not None else "No")
            if existing_note['embeddings'] is not None:
                table.add_row("Embedding length", str(len(existing_note['embeddings'][0])))
            
            console.print(table)
            
            if existing_note["embeddings"] is None or len(existing_note["embeddings"]) == 0:
                console.print("\n[bold]📥 Adding to vector store...[/]")
                self.vectorstore.add_note(note_id, content)
                console.print("[green]✓[/] Added to vector store")
                
                # Verify the note was added correctly
                console.print("\n[bold]🔍 Verifying note addition...[/]")
                check_note = self.vectorstore.collection.get(
                    ids=[note_id],
                    include=['embeddings', 'metadatas', 'documents']
                )
                
                verify_table = Table(title="Verification Results", show_header=False)
                verify_table.add_column("Property", style="cyan")
                verify_table.add_column("Value", style="white")
                
                verify_table.add_row("IDs", str(check_note['ids']))
                verify_table.add_row("Has embeddings", "Yes" if check_note['embeddings'] is not None else "No")
                if check_note['embeddings'] is not None:
                    verify_table.add_row("Embedding length", str(len(check_note['embeddings'][0])))
                
                console.print(verify_table)
                
                if not check_note['embeddings'] or len(check_note['embeddings']) == 0:
                    console.print("[red]❌[/] Warning: Note was added but embeddings are still missing")
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
        
        # Debug initial state
        console.print("\n[bold]🔍 Initial State:[/]")
        state_table = Table(title="Initial State", show_header=False)
        state_table.add_column("Property", style="cyan")
        state_table.add_column("Value", style="white")
        
        state_table.add_row("Current Note ID", state.current_note_id)
        state_table.add_row("Context Keys", ", ".join(state.context.keys()))
        state_table.add_row("Content Length", str(len(state.context.get("current_note_content", ""))))
        
        console.print(state_table)
        
        # Run workflow steps
        try:
            # Step 1: Categorize note
            categories = self._categorize_note(content)
            state.context["note_categories"] = categories
            
            # Step 2: Analyze content
            topics = self._analyze_content(content)
            state.context["topics"] = topics
            
            # Step 3: Find similar notes
            similar_notes = self.similarity_tool.func(
                note_id=note_id,
                top_n=5,
                exclude_ids=[note_id]
            )
            state.context["similar_notes"] = similar_notes
            
            # Step 4: Generate links
            if similar_notes:
                current_summary = self._summarize_content(content, max_length=500)
                similar_summaries = []
                
                for note in similar_notes[:3]:
                    content = note.get("content", "")
                    summary = self._summarize_content(content, max_length=300)
                    similar_summaries.append({
                        "note_id": note["note_id"],
                        "similarity_score": note["similarity_score"],
                        "content_summary": summary
                    })
                
                # Get LLM suggestions for links
                suggested_links = self._suggest_links(
                    note_id,
                    state.context.get("topics", []),
                    current_summary,
                    similar_summaries
                )
                
                # Add suggested links to state
                for link in suggested_links:
                    if float(link["confidence_score"]) >= 0.5:
                        self.add_link(
                            state,
                            note_id,
                            link["note_id"],
                            float(link["confidence_score"]),
                            "LLM suggested link"
                        )
            
            # Step 5: Update clusters
            for topic in topics:
                self.add_to_cluster(state, topic, note_id)
            
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
                    
                    # Display summary
                    table = Table(title="Written Content Summary", show_header=False)
                    table.add_column("Section", style="cyan")
                    table.add_column("Count", style="white")
                    table.add_row("Links added", str(len(links)))
                    table.add_row("Topics added", str(len(topics)))
                    table.add_row("Categories added", str(len(categories)))
                    console.print(table)
                    
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
        
        for note in notes:
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
                # Create error message with escaped paths
                error_msg = f"Error processing note: {self._escape_path(str(note))}\nError: {str(e)}"
                console.print(Panel(
                    error_msg,
                    title="Error",
                    border_style="red"
                ))
                continue
                
        return states

def run_agent():
    """Run the Pydantic agent on a vault."""
    # Initialize services
    agent = PydanticAgentSystem()

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
