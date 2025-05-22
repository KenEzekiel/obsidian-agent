import json
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, TypedDict, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from src.agents.base import AgentSystem, AgentState, NoteLink, SimilarNote, VectorSimilarityTool, console
from src.services.vectorstore import VectorStoreClient
from src.services.embedding import EmbeddingService
from src.core.vault import VaultReader, VaultWriter
from src.config.agent_config import AgentConfig, get_default_config

class LangGraphState(TypedDict):
    state: AgentState
    messages: Sequence[HumanMessage | AIMessage]

class LangGraphAgentSystem(AgentSystem):
    """LangGraph implementation of the agent system."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the LangGraph agent system.
        
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
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(LangGraphState)
        
        # Add nodes
        workflow.add_node("categorize_note", self._categorize_note_node)
        workflow.add_node("analyze_content", self._analyze_content_node)
        workflow.add_node("find_similar", self._find_similar_node)
        workflow.add_node("generate_links", self._generate_links_node)
        workflow.add_node("update_clusters", self._update_clusters_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Define edges
        workflow.add_edge("categorize_note", "analyze_content")
        workflow.add_edge("analyze_content", "find_similar")
        workflow.add_edge("find_similar", "generate_links")
        workflow.add_edge("generate_links", "update_clusters")
        workflow.add_edge("update_clusters", "finalize")
        workflow.add_edge("finalize", END)
        
        workflow.set_entry_point("categorize_note")
        return workflow.compile()
    
    def _categorize_note_node(self, state: LangGraphState) -> LangGraphState:
        """Node for categorizing the note based on its utility and purpose."""
        console.print("\n[bold blue]LangGraph[/] 📑 Categorizing note:", state['state'].current_note_id)
        
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

            note_content = state["state"].context.get("current_note_content", "")
            if not note_content:
                raise ValueError("No note content found in context")
                
            console.print("[bold]🔍 Analyzing note purpose...[/]")
            
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=note_content)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse categories and justifications
            categories_with_reasons = [cat.strip() for cat in response.content.split(",")]
            state["state"].context["note_categories"] = categories_with_reasons
            
            # Display results
            console.print(Panel(
                "\n".join(f"• {cat}" for cat in categories_with_reasons),
                title="Note Categories",
                border_style="green"
            ))
            
            console.print("[green]✓[/] Successfully categorized note")
            
        except Exception as e:
            console.print(Panel(
                f"[red]Error in categorize_note_node[/]\n"
                f"Error: {str(e)}",
                title="Error",
                border_style="red"
            ))
            raise
            
        return state
    
    def _analyze_content_node(self, state: LangGraphState) -> LangGraphState:
        """Node for analyzing note content."""
        console.print("\n[bold blue]LangGraph[/] 📝 Analyzing content for note:", state['state'].current_note_id)
        
        try:
            # Debug context
            console.print("\n[bold]🔍 Checking context...[/]")
            context_table = Table(title="Current Context", show_header=False)
            context_table.add_column("Key", style="cyan")
            context_table.add_column("Value", style="white")
            
            for key, value in state['state'].context.items():
                context_table.add_row(key, str(value)[:100] + "..." if len(str(value)) > 100 else str(value))
            console.print(context_table)
            
            system_message = "You are an expert at analyzing note content and identifying key topics. Return the topics as a comma-separated list. Every topic should be like a title that has clear context on the domain of the note, e.g. Definition of AI, Goals of AI, etc."
            console.print("[dim]Using system message:[/]", system_message)
            
            note_content = state["state"].context.get("current_note_content", "")
            if not note_content:
                raise ValueError("No note content found in context")
                
            console.print("[bold]🤔 Analyzing note content...[/]")
            console.print("[dim]Content length:[/]", len(note_content))
            
            # Create messages directly instead of using template
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=note_content)
            ]
            
            chain = self.llm
            response = chain.invoke(messages)
            
            # Convert comma-separated string to array
            topics_array = [topic.strip() for topic in response.content.split(",")]
            state["state"].context["topics"] = topics_array
            console.print("[green]✓[/] Successfully analyzed content and identified topics")
            
            # Print identified topics
            console.print(Panel(
                "\n".join(f"• {topic}" for topic in topics_array),
                title="Identified Topics",
                border_style="blue"
            ))
            
        except Exception as e:
            console.print(Panel(
                f"[red]Error in analyze_content_node[/]\n"
                f"Error: {str(e)}",
                title="Error",
                border_style="red"
            ))
            state["state"].error = f"Error analyzing content: {str(e)}"
            state["state"].context["topics"] = []
            
        return state
    
    def _find_similar_node(self, state: LangGraphState) -> LangGraphState:
        """Node for finding similar notes."""
        current_note_id: str = state["state"].current_note_id
        console.print("\n[bold blue]LangGraph[/] 🔍 Finding similar notes for:", current_note_id)
        
        try:
            # Use the vector similarity tool
            similar_notes = self.similarity_tool.func(
                note_id=current_note_id,
                top_n=5,
                exclude_ids=[current_note_id]
            )
            
            # Store results in state
            state["state"].context["similar_notes"] = similar_notes
            
            # Create a table for similar notes
            table = Table(title="Similar Notes Found", show_header=True, header_style="bold magenta")
            table.add_column("Note ID", style="cyan")
            table.add_column("Score", justify="right", style="green")
            table.add_column("Preview", style="white")
            
            for note in state['state'].context['similar_notes']:
                table.add_row(
                    note['note_id'],
                    f"{note['similarity_score']:.3f}",
                    note['content'][:100] + "..."
                )
            
            console.print(table)
            
        except Exception as e:
            console.print(Panel(
                f"[red]Error in find_similar_node[/]\n"
                f"Error: {str(e)}",
                title="Error",
                border_style="red"
            ))
            state["state"].error = f"Error finding similar notes: {str(e)}"
            state["state"].context["similar_notes"] = []
        
        return state
    
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
                self._escape_path(summary),  # Escape any file paths in the summary
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

    def _escape_path(self, path: str) -> str:
        """Escape a file path for Rich markup."""
        return str(path).replace("[", "\\[").replace("]", "\\]")

    def _generate_links_node(self, state: LangGraphState) -> LangGraphState:
        """Node for generating link suggestions."""
        console.print("\n[bold blue]LangGraph[/] 🔗 Generating links for note:", self._escape_path(state['state'].current_note_id))
        
        if state["state"].error:
            console.print("[yellow]⚠️[/] Skipping due to error:", state['state'].error)
            return state
            
        # Get current note content and create a summary
        current_content = state["state"].context.get("current_note_content", "")
        console.print("[bold]📝 Summarizing current note...[/]")
        current_summary = self._summarize_content(current_content, max_length=500)
        
        # Get similar notes and create summaries
        similar_notes = state["state"].context.get("similar_notes", [])
        similar_summaries = []
        
        console.print("[bold]📝 Summarizing similar notes...[/]")
        for note in similar_notes[:3]:
            content = note.get("content", "")
            summary = self._summarize_content(content, max_length=300)
            similar_summaries.append({
                "note_id": note["note_id"],
                "similarity_score": note["similarity_score"],
                "content_summary": summary
            })
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at suggesting meaningful connections between notes. You answer in this JSON FORMAT: 
            {{
                "links": [
                    {{
                        "note_id": "note_id",
                        "confidence_score": "confidence_score"
                    }}
                ]
            }}
            
            sorted by confidence_score in descending order
            """),
            ("human", """Given the current note and similar notes, suggest meaningful links.

            Current note:
            {current_note_id}
            Topics: {current_topics}
            Summary: {current_content}

            Similar notes:
            {similar_notes}
            """)
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "current_note_id": state["state"].current_note_id,
            "current_topics": state["state"].context.get("topics", ""),
            "current_content": current_summary,
            "similar_notes": "\n\n".join([
                f"Note: {note['note_id']}\n"
                f"Similarity: {note['similarity_score']:.3f}\n"
                f"Summary: {note['content_summary']}"
                for note in similar_summaries
            ])
        })
        
        console.print(Panel(
            response.content,
            title="🤖 LLM Response",
            border_style="blue"
        ))
        
        # Parse LLM response and add suggested links
        try:
            # Clean the response content by removing markdown code block formatting
            content = response.content.strip()
            if content.startswith("```"):
                # Remove the first line (```json) and last line (```)
                content = "\n".join(content.split("\n")[1:-1])
            
            response_data = json.loads(content)
            suggested_links = response_data.get("links", [])
            
            if suggested_links:
                for link in suggested_links:
                    try:
                        console.print("\n[bold]📎 Adding suggested link:[/]")
                        console.print(f"  From: [cyan]{self._escape_path(state['state'].current_note_id)}[/]")
                        console.print(f"  To: [cyan]{self._escape_path(link['note_id'])}[/]") 
                        console.print(f"  Confidence: [green]{link['confidence_score']}[/]")
                        
                        if float(link["confidence_score"]) >= 0.5:
                            console.print("[yellow]ℹ️[/] Link confidence meets threshold (>= 0.5), proceeding...")
                        else:
                            console.print("[yellow]ℹ️[/] Link confidence below threshold (< 0.5), skipping...")
                            continue
                        self.add_link(
                            state["state"],
                            state["state"].current_note_id,
                            link["note_id"],
                            float(link["confidence_score"]),
                            response.content
                        )
                        console.print("[green]✓[/] Link added successfully")
                    except (KeyError, ValueError) as e:
                        console.print(Panel(
                            f"[red]Error adding individual link[/]\nError: {str(e)}",
                            title="Error",
                            border_style="red"
                        ))
            else:
                console.print("[yellow]ℹ️[/] No links suggested by LLM")
                
        except json.JSONDecodeError as e:
            console.print(Panel(
                f"[red]Error parsing LLM response[/]\nError: {str(e)}\nContent: {content}",
                title="Error", 
                border_style="red"
            ))
            
        return state
    
    def _update_clusters_node(self, state: LangGraphState) -> LangGraphState:
        """Node for updating topic clusters."""
        if state["state"].error:
            return state
            
        topics = state["state"].context.get("topics", "")
        
        for topic in topics:
            topic = topic.strip()
            if topic:
                self.add_to_cluster(state["state"], topic, state["state"].current_note_id)
        
        return state
    
    def _finalize_node(self, state: LangGraphState) -> LangGraphState:
        """Node for finalizing note processing."""
        self.mark_processed(state["state"], state["state"].current_note_id)
        return state
    
    def _write_suggestions_to_note(self, state: AgentState, vault) -> None:
        """Write suggested links and topics to the note file."""
        console.print("\n[bold blue]LangGraph[/] 📝 Writing suggestions to note:", state.current_note_id)
        
        try:
            # Read existing content
            content = vault.read_note(state.current_note_id)
            
            # Build suggestions section with JSON-like format
            suggestions_section = "\n\n## Agent Suggestions\n```suggestions\n{\n"
            
            # Add links subsection
            suggestions_section += '  "suggested_links": [\n'
            for link in state.suggested_links:
                suggestions_section += f'    {{\n      "target": "{link.target_note}",\n      "score": {link.similarity_score:.2f}\n    }},\n'
            suggestions_section = suggestions_section.rstrip(',\n') + '\n  ],\n'
            
            # Add topics subsection
            suggestions_section += '  "topics": [\n'
            for topic, notes in state.topic_clusters.items():
                if state.current_note_id in notes:
                    suggestions_section += f'    "{topic}",\n'
            suggestions_section = suggestions_section.rstrip(',\n') + '\n  ]\n'
            
            suggestions_section += "}\n```\n"
            
            # Check if section already exists and replace if it does
            if "## Agent Suggestions" in content:
                start = content.find("## Agent Suggestions")
                end = content.find("```\n", start) + 4
                content = content[:start] + suggestions_section.lstrip() + content[end:]
            else:
                content += suggestions_section
                
            # Write back to file
            vault.write_note(state.current_note_id, content)
            
            console.print("[green]✓[/] Successfully wrote suggestions to note")
            
            # Display summary
            table = Table(title="Written Content Summary", show_header=False)
            table.add_column("Section", style="cyan")
            table.add_column("Count", style="white")
            table.add_row("Links added", str(len(state.suggested_links)))
            table.add_row("Topics added", str(len([t for t, notes in state.topic_clusters.items() 
                                                 if state.current_note_id in notes])))
            console.print(table)
            
        except Exception as e:
            console.print(Panel(
                f"[red]Error writing suggestions to note[/]\n"
                f"Error: {str(e)}\n"
                f"Note: {state.current_note_id}",
                title="Error",
                border_style="red"
            ))
            
    def process_note(self, note_id: str, content: str, vault_path: str) -> AgentState:
        """Process a single note through the workflow."""
        console.print("\n[bold blue]LangGraph[/] 📝 Processing note:", note_id)
        
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
        
        # Initialize state and run workflow
        console.print("\n[bold]📝 Initializing workflow state...[/]")
        initial_state = AgentState(
            current_note_id=note_id,
            context={"current_note_content": content}
        )
        
        # Debug initial state
        console.print("\n[bold]🔍 Initial State:[/]")
        state_table = Table(title="Initial State", show_header=False)
        state_table.add_column("Property", style="cyan")
        state_table.add_column("Value", style="white")
        
        state_table.add_row("Current Note ID", initial_state.current_note_id)
        state_table.add_row("Context Keys", ", ".join(initial_state.context.keys()))
        state_table.add_row("Content Length", str(len(initial_state.context.get("current_note_content", ""))))
        
        console.print(state_table)
        
        console.print("\n[bold]🚀 Running workflow...[/]")
        result = self.graph.invoke({
            "state": initial_state,
            "messages": []
        })
        
        # Write suggestions and tags using VaultWriter
        if result["state"].suggested_links or result["state"].context.get("topics") or result["state"].context.get("note_categories"):
            console.print("\n[bold]📝 Writing suggestions and tags...[/]")
            
            # Format links for VaultWriter
            links = [
                {
                    "target": link.target_note,
                    "score": link.similarity_score
                }
                for link in result["state"].suggested_links
            ]
            
            # Get topics and categories from context
            topics = result["state"].context.get("topics", [])
            categories = result["state"].context.get("note_categories", [])
            
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
        
        console.print("[green]✓[/] Workflow completed")
        return result["state"]
    
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
    """Run the LangGraph agent on a vault."""
    # Initialize services
    agent = LangGraphAgentSystem()

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
