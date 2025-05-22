"""
Configuration system for agent selection and settings.
"""

from typing import Dict, Any, Optional, Literal
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    """Configuration for LLM settings."""
    model: str = "gemini-2.0-flash"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    system_message: Optional[str] = None

class VectorStoreConfig(BaseModel):
    """Configuration for vector store settings."""
    collection_name: str = "embeddings"
    embedding_model: str = "text-embedding-3-small"
    similarity_threshold: float = 0.7
    max_results: int = 5
    persist_directory: Optional[Path] = None

class EmbeddingConfig(BaseModel):
    """Configuration for embedding settings."""
    model_type: str = "api"
    model_name: str = "text-embedding-3-small"
    batch_size: int = 32
    max_retries: int = 3
    retry_delay: float = 1.0
    max_tokens: int = 8000

class FrameworkConfig(BaseModel):
    """Base configuration for framework-specific settings."""
    framework: str
    config: Dict[str, Any] = Field(default_factory=dict)

class LangGraphConfig(FrameworkConfig):
    """Configuration for LangGraph framework."""
    def __init__(self, **data):
        super().__init__(framework="langgraph", **data)

class CrewAIConfig(FrameworkConfig):
    """Configuration for CrewAI framework."""
    def __init__(self, **data):
        super().__init__(framework="crewai", **data)

class AutogenConfig(FrameworkConfig):
    """Configuration for AutoGen framework."""
    def __init__(self, **data):
        super().__init__(framework="autogen", **data)

class PydanticConfig(FrameworkConfig):
    """Configuration for Pydantic framework."""
    def __init__(self, **data):
        super().__init__(framework="pydantic", **data)

class AgentConfig(BaseModel):
    """Main configuration for the agent system."""
    framework_config: FrameworkConfig
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    vectorstore_config: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding_config: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    
    @classmethod
    def create_langgraph_config(cls, **kwargs) -> 'AgentConfig':
        """Create a configuration for LangGraph framework."""
        return cls(
            framework_config=LangGraphConfig(**kwargs),
            llm_config=LLMConfig(**kwargs.get("llm_config", {})),
            vectorstore_config=VectorStoreConfig(**kwargs.get("vectorstore_config", {})),
            embedding_config=EmbeddingConfig(**kwargs.get("embedding_config", {}))
        )
    
    @classmethod
    def create_crewai_config(cls, **kwargs) -> 'AgentConfig':
        """Create a configuration for CrewAI framework."""
        return cls(
            framework_config=CrewAIConfig(**kwargs),
            llm_config=LLMConfig(**kwargs.get("llm_config", {})),
            vectorstore_config=VectorStoreConfig(**kwargs.get("vectorstore_config", {})),
            embedding_config=EmbeddingConfig(**kwargs.get("embedding_config", {}))
        )
    
    @classmethod
    def create_autogen_config(cls, **kwargs) -> 'AgentConfig':
        """Create a configuration for AutoGen framework."""
        return cls(
            framework_config=AutogenConfig(**kwargs),
            llm_config=LLMConfig(**kwargs.get("llm_config", {})),
            vectorstore_config=VectorStoreConfig(**kwargs.get("vectorstore_config", {})),
            embedding_config=EmbeddingConfig(**kwargs.get("embedding_config", {}))
        )
    
    @classmethod
    def create_pydantic_config(cls, **kwargs) -> 'AgentConfig':
        """Create a configuration for Pydantic framework."""
        return cls(
            framework_config=PydanticConfig(**kwargs),
            llm_config=LLMConfig(**kwargs.get("llm_config", {})),
            vectorstore_config=VectorStoreConfig(**kwargs.get("vectorstore_config", {})),
            embedding_config=EmbeddingConfig(**kwargs.get("embedding_config", {}))
        )

    @classmethod
    def from_yaml(cls, config_path: Path) -> "AgentConfig":
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    def to_yaml(self, config_path: Path) -> None:
        """Save configuration to a YAML file."""
        with open(config_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

def get_default_config() -> AgentConfig:
    """Get default configuration."""
    return AgentConfig(
        framework_config=LangGraphConfig(
            config={
                "temperature": 0.0,
                "max_tokens": 2000,
                "safety_settings": {
                    "HARASSMENT": "block_none",
                    "HATE_SPEECH": "block_none",
                    "SEXUALLY_EXPLICIT": "block_none",
                    "DANGEROUS_CONTENT": "block_none"
                }
            }
        ),
        llm_config=LLMConfig(
            model="gemini-2.0-flash",
            temperature=0.0,
            system_message="You are an expert at analyzing note content and identifying key topics."
        ),
        vectorstore_config=VectorStoreConfig(
            collection_name="embeddings",
            embedding_model="text-embedding-3-small",
            similarity_threshold=0.7,
            max_results=5,
            persist_directory=Path("./data/embeddings")
        ),
        embedding_config=EmbeddingConfig(
            model_type="api",
            model_name="text-embedding-3-small",
            batch_size=32,
            max_retries=3,
            retry_delay=1.0,
            max_tokens=8000
        )
    )

def create_agent_config(
    framework: str = "langgraph",
    model: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    similarity_threshold: float = 0.7,
    max_results: int = 5,
    cache_dir: Optional[Path] = None,
    provider: Literal["openai", "google"] = "google",
    system_message: Optional[str] = None
) -> AgentConfig:
    """
    Create a configuration with the specified settings.

    Args:
        framework: Agent framework to use
        model: LLM model to use
        temperature: Temperature for generation
        similarity_threshold: Minimum similarity score for results
        max_results: Maximum number of results to return
        cache_dir: Directory for caching
        provider: LLM provider (openai or google)
        system_message: System message to use for the model

    Returns:
        AgentConfig instance
    """
    return AgentConfig(
        framework_config=LangGraphConfig(
            config={
                "temperature": temperature,
                "max_tokens": 2000,
                "safety_settings": {
                    "HARASSMENT": "block_none",
                    "HATE_SPEECH": "block_none",
                    "SEXUALLY_EXPLICIT": "block_none",
                    "DANGEROUS_CONTENT": "block_none"
                }
            }
        ),
        llm_config=LLMConfig(
            model=model,
            temperature=temperature,
            system_message=system_message
        ),
        vectorstore_config=VectorStoreConfig(
            collection_name="embeddings",
            embedding_model="text-embedding-3-small",
            similarity_threshold=similarity_threshold,
            max_results=max_results,
            persist_directory=cache_dir or Path("./data/embeddings")
        ),
        embedding_config=EmbeddingConfig(
            model_type="api",
            model_name="text-embedding-3-small",
            batch_size=32,
            max_retries=3,
            retry_delay=1.0,
            max_tokens=8000
        )
    ) 