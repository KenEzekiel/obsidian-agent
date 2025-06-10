# Goal-Orientation: The agent has a persistent goal (meaningfully link the current note).
# Perception: It can sense the state of the current note and the broader vault.
# Reasoning & Planning: It can decide what to do next based on its goal and perceptions, rather than following a fixed sequence. This might involve choosing between different tools or strategies.
# Action & Tool Use: It can execute actions (like searching, analyzing, suggesting) using a set of tools.
# (Optional but good) Learning/Adaptation: It might learn from user feedback or outcomes to improve its suggestions over time.

# Agentic system includes using different strategies to achieve the goal.
# Kalau single strategy kaya sekarang, not agentic, like the first system.

# Agentic system uses multiple strategies to achieve the goal.

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, TypedDict, Literal, Union, Any
import json

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from obsidian_agent.agents.base import (
    AgentSystem,
    AgentState,
    VectorSimilarityTool,
    console,
)
from obsidian_agent.services.vectorstore import VectorStoreClient
from obsidian_agent.services.embedding import EmbeddingService
from obsidian_agent.core.vault import VaultReader, VaultWriter
from obsidian_agent.config.agent_config import AgentConfig, get_default_config


class AgenticState(TypedDict):
    """State for the agentic system."""

    state: AgentState
    messages: List[Union[HumanMessage, AIMessage]]
    current_strategy: Optional[str]
    strategy_history: List[Dict]
    reasoning: Optional[str]
    confidence: float
    needs_fallback: bool
    observations: List[Dict[str, Any]]
    thoughts: List[str]


class Strategy(TypedDict):
    """Strategy definition."""

    name: str
    description: str
    confidence_threshold: float
    max_attempts: int
    attempts: int


class LangGraphAgenticSystem(AgentSystem):
    """Goal-oriented, multi-strategy agentic system using LangGraph with reactive capabilities."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the agentic system."""
        self.config = config or get_default_config()

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_config.model,
            temperature=self.config.llm_config.temperature,
        )

        # Initialize embedding service
        self.embedding_service = EmbeddingService(
            model_type=self.config.embedding_config.model_type,
            model_name=self.config.embedding_config.model_name,
            batch_size=self.config.embedding_config.batch_size,
        )

        # Initialize vector store
        self.vectorstore = VectorStoreClient(
            embedding_service=self.embedding_service,
            collection_name=self.config.vectorstore_config.collection_name,
            persist_directory=self.config.vectorstore_config.persist_directory,
        )

        self.similarity_tool = VectorSimilarityTool(self.vectorstore)

        # Define available strategies
        self.strategies = {
            "semantic_similarity": Strategy(
                name="semantic_similarity",
                description="Find links based on semantic similarity between notes",
                confidence_threshold=0.7,
                max_attempts=3,
                attempts=0,
            ),
            "topic_based": Strategy(
                name="topic_based",
                description="Find links based on shared topics and themes",
                confidence_threshold=0.6,
                max_attempts=2,
                attempts=0,
            ),
            "reference_based": Strategy(
                name="reference_based",
                description="Find links based on explicit references and citations",
                confidence_threshold=0.8,
                max_attempts=2,
                attempts=0,
            ),
            "category_based": Strategy(
                name="category_based",
                description="Find links based on note categories and purposes",
                confidence_threshold=0.75,
                max_attempts=2,
                attempts=0,
            ),
            "fallback": Strategy(
                name="fallback",
                description="Use basic similarity as fallback strategy",
                confidence_threshold=0.5,
                max_attempts=1,
                attempts=0,
            ),
        }

        # Initialize ReAct agent
        self._initialize_react_agent()

        # Create the workflow graph
        self.graph = self._create_graph()

    def _escape_path(self, path: str) -> str:
        """Escape a file path for Rich markup."""
        return str(path).replace("[", "\\[").replace("]", "\\]")

    def _initialize_react_agent(self):
        """Initialize the ReAct agent with tools and prompt."""
        # Define tools for the ReAct agent
        self.tools = [
            Tool(
                name="find_similar_notes",
                func=self.similarity_tool.func,
                description="Find similar notes using vector similarity search",
            ),
            Tool(
                name="analyze_topics",
                func=self._analyze_topics,
                description="Analyze note content to identify key topics",
            ),
            Tool(
                name="find_references",
                func=self._find_references,
                description="Find explicit references to other notes",
            ),
            Tool(
                name="analyze_category",
                func=self._analyze_category,
                description="Analyze note content to determine its category and purpose",
            ),
            Tool(
                name="find_category_matches",
                func=self._find_category_matches,
                description="Find notes with matching categories and purposes",
            ),
            Tool(
                name="evaluate_confidence",
                func=self._evaluate_confidence,
                description="Evaluate confidence in current strategy results",
            ),
        ]

        # Create ReAct prompt
        self.react_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at analyzing notes and finding meaningful connections.
            You have access to the following tools:
            {tools}
            
            Use the following format:
            
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
            ]
        )

        # Create ReAct agent
        self.react_agent = create_react_agent(
            llm=self.llm, tools=self.tools, prompt=self.react_prompt
        )

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.react_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    def _analyze_topics(self, content: str) -> List[str]:
        """Analyze note content to identify key topics."""
        system_message = """You are an expert at identifying key topics in notes.
        Return the topics as a comma-separated list."""

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=content),
        ]

        response = self.llm.invoke(messages)
        return [topic.strip() for topic in response.content.split(",")]

    def _find_references(self, content: str) -> List[str]:
        """Find explicit references to other notes."""
        system_message = """You are an expert at identifying references to other notes.
        Look for explicit mentions, citations, or references to other notes.
        Return the referenced note IDs as a comma-separated list."""

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=content),
        ]

        response = self.llm.invoke(messages)
        return [note_id.strip() for note_id in response.content.split(",")]

    def _analyze_category(self, content: str) -> Dict[str, Any]:
        """Analyze note content to determine its category and purpose."""
        system_message = """You are an expert at categorizing notes based on their purpose and content.
        Categorize the note into one or more of these categories:
        - Knowledge: Educational content, concepts, theories, explanations
        - Reference: Lists, resources, links, tools
        - Project: Project plans, requirements, documentation
        - Reminder: Tasks, todos, deadlines, important dates
        - Journal: Personal reflections, thoughts, experiences
        - Decision: Decision records, reasoning, trade-offs
        - Template: Reusable structures or formats
        - Archive: Historical information, completed items
        
        Return a JSON object with:
        {
            "primary_category": "main category",
            "secondary_categories": ["additional categories"],
            "purpose": "brief description of note's purpose",
            "confidence": confidence score between 0 and 1
        }"""

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=content),
        ]

        response = self.llm.invoke(messages)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "primary_category": "Unknown",
                "secondary_categories": [],
                "purpose": "Unable to determine purpose",
                "confidence": 0.0,
            }

    def _find_category_matches(
        self, category_info: Dict[str, Any], exclude_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Find notes with matching categories and purposes."""
        try:
            # Get all notes from vector store
            all_notes = self.vectorstore.get_all_notes()
            matches = []

            for note in all_notes:
                if exclude_ids and note["note_id"] in exclude_ids:
                    continue

                # Analyze category of potential match
                note_category = self._analyze_category(note["content"])

                # Check for category matches
                if note_category["primary_category"] == category_info[
                    "primary_category"
                ] or any(
                    cat in note_category["secondary_categories"]
                    for cat in category_info["secondary_categories"]
                ):
                    # Calculate match score based on category overlap
                    primary_match = (
                        1.0
                        if note_category["primary_category"]
                        == category_info["primary_category"]
                        else 0.0
                    )
                    secondary_matches = sum(
                        1
                        for cat in category_info["secondary_categories"]
                        if cat in note_category["secondary_categories"]
                    )
                    secondary_score = secondary_matches / max(
                        len(category_info["secondary_categories"]), 1
                    )

                    # Combine scores with confidence
                    match_score = (
                        primary_match * 0.7 + secondary_score * 0.3
                    ) * note_category["confidence"]

                    if (
                        match_score
                        >= self.strategies["category_based"]["confidence_threshold"]
                    ):
                        matches.append(
                            {
                                "note_id": note["note_id"],
                                "score": match_score,
                                "category_info": note_category,
                                "reason": f"Category match: {note_category['primary_category']}",
                            }
                        )

            return sorted(matches, key=lambda x: x["score"], reverse=True)

        except Exception as e:
            console.print(
                Panel(
                    f"[red]Error in find_category_matches[/]\nError: {str(e)}",
                    title="Error",
                    border_style="red",
                )
            )
            return []

    def _evaluate_confidence(self, results: Dict[str, Any]) -> float:
        """Evaluate confidence in current strategy results."""
        system_message = """You are an expert at evaluating the quality of note connections.
        Given the results of a strategy, evaluate the confidence in the connections found.
        Return a confidence score between 0 and 1."""

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=json.dumps(results, indent=2)),
        ]

        response = self.llm.invoke(messages)
        try:
            return float(response.content.strip())
        except ValueError:
            return 0.5  # Default confidence if parsing fails

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow with reactive capabilities."""
        workflow = StateGraph(AgenticState)

        # Add nodes
        workflow.add_node("react_agent", self._react_agent_node)
        workflow.add_node("evaluate_results", self._evaluate_results_node)
        workflow.add_node("finalize", self._finalize_node)

        # Define edges with conditional routing
        workflow.add_conditional_edges(
            "react_agent",
            self._should_continue,
            {True: "evaluate_results", False: "finalize"},
        )

        workflow.add_edge("evaluate_results", "react_agent")
        workflow.add_edge("finalize", END)

        # Set entry point
        workflow.set_entry_point("react_agent")

        return workflow.compile()

    def _should_continue(self, state: AgenticState) -> bool:
        """Determine if we should continue with ReAct agent or move to finalize."""
        return (
            state["confidence"] < 0.6
            and len(state["observations"]) < 3
            and not state["needs_fallback"]
        )

    def _react_agent_node(self, state: AgenticState) -> AgenticState:
        """Node for the ReAct agent."""
        console.print("\n[bold blue]Agentic[/] 🤖 ReAct agent thinking...")

        try:
            # Prepare input for ReAct agent
            current_note = state["state"].current_note_id
            content = state["state"].context.get("current_note_content", "")

            # Analyze category first
            category_info = self._analyze_category(content)
            state["state"].context["category_info"] = category_info

            # Create input for ReAct agent
            agent_input = f"""Analyze this note and find meaningful connections:
            
            Note ID: {current_note}
            Content: {content[:500]}...
            
            Category Analysis:
            Primary Category: {category_info["primary_category"]}
            Secondary Categories: {", ".join(category_info["secondary_categories"])}
            Purpose: {category_info["purpose"]}
            Confidence: {category_info["confidence"]}
            
            Previous observations:
            {json.dumps(state.get("observations", []), indent=2)}
            
            Previous thoughts:
            {json.dumps(state.get("thoughts", []), indent=2)}"""

            print(agent_input)

            # Run ReAct agent
            result = self.agent_executor.invoke(
                {"input": agent_input, "chat_history": state["messages"]}
            )

            print(result)

            # Update state with ReAct results
            state["messages"].extend(result.get("chat_history", []))
            state["observations"].append(result.get("output", {}))
            state["thoughts"].append(result.get("intermediate_steps", []))

            # Process any links found by ReAct agent
            if "links" in result.get("output", {}):
                for link in result["output"]["links"]:
                    self.add_link(
                        state["state"],
                        current_note,
                        link["target"],
                        link["score"],
                        link.get("reason", "ReAct agent suggestion"),
                    )

            # Add category-based links
            category_matches = self._find_category_matches(
                category_info, exclude_ids=[current_note]
            )

            for match in category_matches:
                self.add_link(
                    state["state"],
                    current_note,
                    match["note_id"],
                    match["score"],
                    match["reason"],
                )

            console.print("[green]✓[/] ReAct agent completed analysis")

        except Exception as e:
            console.print(
                Panel(
                    f"[red]Error in react_agent_node[/]\nError: {str(e)}",
                    title="Error",
                    border_style="red",
                )
            )
            state["needs_fallback"] = True

        return state

    def _evaluate_results_node(self, state: AgenticState) -> AgenticState:
        """Node for evaluating ReAct agent results."""
        console.print("\n[bold blue]Agentic[/] 📊 Evaluating results...")

        try:
            # Get latest observation
            latest_observation = (
                state["observations"][-1] if state["observations"] else {}
            )

            # Evaluate confidence
            confidence = self._evaluate_confidence(latest_observation)
            state["confidence"] = confidence

        except Exception as e:
            console.print(
                Panel(
                    f"[red]Error in evaluate_results_node[/]\nError: {str(e)}",
                    title="Error",
                    border_style="red",
                )
            )
            state["needs_fallback"] = True

        return state

    def _finalize_node(self, state: AgenticState) -> AgenticState:
        """Node for finalizing note processing."""
        console.print("\n[bold blue]Agentic[/] ✨ Finalizing...")

        try:
            # Mark note as processed
            self.mark_processed(state["state"], state["state"].current_note_id)

            # Print summary
            table = Table(title="Processing Summary", show_header=False)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Observations", str(len(state["observations"])))
            table.add_row("Links found", str(len(state["state"].suggested_links)))
            table.add_row("Final confidence", f"{state['confidence']:.2f}")

            console.print(table)

        except Exception as e:
            console.print(
                Panel(
                    f"[red]Error in finalize_node[/]\nError: {str(e)}",
                    title="Error",
                    border_style="red",
                )
            )

        return state

    def process_note(self, note_id: str, content: str, vault_path: str) -> AgentState:
        """Process a single note through the reactive workflow."""
        console.print("\n[bold blue]Agentic[/] 📝 Processing note:", note_id)

        # Initialize vault writer
        vault_writer = VaultWriter(Path(vault_path))

        # Initialize state
        initial_state = AgenticState(
            state=AgentState(
                current_note_id=note_id, context={"current_note_content": content}
            ),
            messages=[],
            current_strategy="react_agent",
            strategy_history=[],
            reasoning=None,
            confidence=0.0,
            needs_fallback=False,
            observations=[],
            thoughts=[],
        )

        # Run the workflow
        result = self.graph.invoke(initial_state)

        # Write suggestions and tags
        if result["state"].suggested_links:
            try:
                # Format links for VaultWriter
                links = [
                    {"target": link.target_note, "score": link.similarity_score}
                    for link in result["state"].suggested_links
                ]

                # Write suggestions
                vault_writer.write_suggestions(
                    note_path=Path(note_id),
                    links=links,
                    topics=[],  # Topics are handled by ReAct agent
                    categories=[],  # Categories are handled by ReAct agent
                )

            except Exception as e:
                console.print(
                    Panel(
                        f"[red]Error writing suggestions[/]\nError: {str(e)}",
                        title="Error",
                        border_style="red",
                    )
                )

        return result["state"]

    def process_vault(self, vault_path: str) -> List[AgentState]:
        """Process all notes in a vault."""
        vault = VaultReader(Path(vault_path))
        notes = vault.get_all_notes()
        states = []

        console.print(
            Panel(
                f"Path: {self._escape_path(vault_path)}\nTotal notes: {len(notes)}",
                title="📚 Processing Vault",
                border_style="blue",
            )
        )

        for note in notes:
            try:
                content = vault.read_note(note)
                console.print(
                    Panel(
                        f"Note: {self._escape_path(note)}\nPreview: {content[:100]}...",
                        title="[bold blue]📝 Processing Note[/]",
                        border_style="blue",
                    )
                )
                state = self.process_note(str(note), content, vault_path)
                states.append(state)

            except Exception as e:
                error_msg = f"Error processing note: {self._escape_path(str(note))}\nError: {str(e)}"
                console.print(Panel(error_msg, title="Error", border_style="red"))
                continue

        return states


def run_agent():
    """Run the agentic system on a vault."""
    # Initialize services
    agent = LangGraphAgenticSystem()

    # Process vault
    vault_path = (
        "/Users/kenneth.ezekiel.suprantoni/Documents/Github/obsidian-agent/Mock Vault"
    )
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
    output_file = output_dir / f"agentic_results_{timestamp}.json"

    # Write results to file
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results written to {output_file}")


if __name__ == "__main__":
    run_agent()
