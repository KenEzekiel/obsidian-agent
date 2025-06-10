"""
GraphBuilder module for constructing and managing graph structures from Obsidian vault links.

This module provides functionality to build and serialize graph structures
representing the connections between notes in an Obsidian vault.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any


class GraphBuilder:
    """
    A class for building and managing graph structures from Obsidian vault links.
    
    This class provides methods to:
    - Build directed graphs from note links
    - Add and manage connections between notes
    - Store note content and metadata
    - Serialize and save graph structures
    """

    def __init__(self):
        """Initialize an empty graph structure."""
        self.graph: Dict[str, Dict[str, Any]] = {}
        self._tag_pattern = re.compile(r'#tag-([^\s]+)')
        self._category_pattern = re.compile(r'#cat-([^\s]+)')

    def add_note(self, file_path: str, content: str, links: List[str]) -> None:
        """
        Add a note with its content and links to the graph.

        Args:
            file_path: Path of the note
            content: Content of the note
            links: List of links in the note
        """
        # Extract tags and categories from content
        tags = self._tag_pattern.findall(content)
        categories = self._category_pattern.findall(content)
        
        self.graph[file_path] = {
            "content": content,
            "links": list(links),
            "tags": list(tags),
            "categories": list(categories)
        }

    def build_graph(self, vault_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Build the graph from a dictionary of notes with their content and links.

        Args:
            vault_data: Dictionary mapping file paths to dictionaries containing content and links

        Returns:
            The constructed graph as a dictionary mapping file paths to their data
        """
        for file_path, data in vault_data.items():
            self.add_note(file_path, data["content"], data["links"])
        
        return self.graph

    def get_graph(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current graph structure.

        Returns:
            The current graph as a dictionary mapping file paths to their data
        """
        return self.graph
    
    def get_lightweight_graph(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a lightweight version of the graph without content.

        Returns:
            Dictionary mapping file paths to their metadata (links, tags, categories)
        """
        return {
            file_path: {
                "links": data["links"],
                "tags": data["tags"],
                "categories": data["categories"]
            }
            for file_path, data in self.graph.items()
        }
    
    def write_graph(self, output_dir: Optional[Path] = None) -> None:
        """
        Write the graph structure to a JSON file.

        Args:
            output_dir: Directory to write the graph file to. If None, uses 'data' directory
                       in the project root.

        Raises:
            OSError: If there are issues creating directories or writing the file
        """
        # Determine output directory
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / 'data'
        output_dir.mkdir(exist_ok=True, parents=True)

        # Write full graph to JSON file
        graph_file = output_dir / 'graph.json'
        with open(graph_file, 'w', encoding='utf-8') as f:
            json.dump(self.graph, f, indent=2, ensure_ascii=False)
            
        # Write lightweight graph to JSON file
        lightweight_file = output_dir / 'graph_lightweight.json'
        with open(lightweight_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_lightweight_graph(), f, indent=2, ensure_ascii=False)
    
    def get_graph_as_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the graph in a dictionary format suitable for external use.
        This method provides a clean interface for other modules to access the graph data.

        Returns:
            Dictionary containing both full and lightweight graph data:
            {
                "full": {file_path: {content, links, tags, categories}},
                "lightweight": {file_path: {links, tags, categories}}
            }
        """
        return {
            "full": self.graph,
            "lightweight": self.get_lightweight_graph()
        }
