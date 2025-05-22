from enum import Enum
from typing import Type
from .base import AgentSystem
from .implementations.langgraph_agent import LangGraphAgentSystem

class AgentFramework(Enum):
    LANGGRAPH = "langgraph"
    CREWAI = "crewai"
    AUTOGEN = "autogen"
    PYDANTIC = "pydantic"

class AgentFactory:
    """Factory for creating agent system instances."""
    
    _implementations: dict[AgentFramework, Type[AgentSystem]] = {
        AgentFramework.LANGGRAPH: LangGraphAgentSystem,
        # Add other implementations as they are created
        # AgentFramework.CREWAI: CrewAIAgentSystem,
        # AgentFramework.AUTOGEN: AutogenAgentSystem,
        # AgentFramework.PYDANTIC: PydanticAgentSystem,
    }
    
    @classmethod
    def create(cls, framework: AgentFramework) -> AgentSystem:
        """Create an agent system instance for the specified framework."""
        if framework not in cls._implementations:
            raise ValueError(f"Unsupported framework: {framework}")
        
        implementation = cls._implementations[framework]
        return implementation()
    
    @classmethod
    def register_implementation(cls, framework: AgentFramework, implementation: Type[AgentSystem]):
        """Register a new agent system implementation."""
        cls._implementations[framework] = implementation 