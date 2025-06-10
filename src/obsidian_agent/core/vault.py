"""
VaultReader and VaultWriter modules for parsing and analyzing Obsidian vault content.

This module provides functionality to read, write and parse Obsidian vault files,
extract wikilinks, and handle both regular links and attachments.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set


class VaultReader:
    """
    A class for reading and parsing Obsidian vault files.
    
    This class provides methods to:
    - Read markdown files from an Obsidian vault
    - Extract wikilinks and attachments
    - Process and normalize link formats
    """

    def __init__(self, vault_path: Optional[Path] = None):
        """
        Initialize a VaultReader instance.

        Args:
            vault_path: Path to the vault directory. If None, defaults to "../Mock Vault"
        """
        if vault_path is None:
            vault_path = Path(__file__).parent.parent / "Mock Vault"
        self.vault_path = vault_path
        
        # Improved regex patterns to handle all valid wikilink characters
        # Matches: [[link]], [[link|display]], [[09.1 - ABC]], etc.
        self._wikilink_pattern = re.compile(r'(?<!!)\[\[([^\]\[]+?)(?:\|([^\]\[]+?))?\]\]')
        self._attachment_pattern = re.compile(r'!\[\[([^\]\[]+?)(?:\|([^\]\[]+?))?\]\]')

    def read_note(self, note_path: Path) -> str:
        """
        Read the contents of a specific note.

        Args:
            note_path: Path to the note file

        Returns:
            The contents of the note as a string

        Raises:
            FileNotFoundError: If the note doesn't exist
            UnicodeDecodeError: If the file encoding is not UTF-8
        """
        try:
            with open(note_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(f"Failed to decode {note_path}: {str(e)}")

    def get_all_notes(self) -> List[Path]:
        """
        Get paths to all Markdown notes in the vault.

        Returns:
            List of Path objects for each .md file

        Raises:
            FileNotFoundError: If vault_path doesn't exist
        """
        if not self.vault_path.exists():
            raise FileNotFoundError(f"Vault path does not exist: {self.vault_path}")
        return sorted(self.vault_path.rglob("*.md"))

    def _process_wikilink(self, match: str) -> str:
        """
        Process a wikilink match to extract and normalize the link.

        Args:
            match: The string inside [[brackets]]

        Returns:
            Normalized link text (without file extension)
        """
        # The regex returns a tuple of (link, display) or (link, None)
        # We only want the link part (first element)
        link = match[0].strip()
        
        # Only remove .md extension if it exists, preserve other extensions
        if link.lower().endswith('.md'):
            return link[:-3]
        return link

    def extract_wikilinks(self, content: str, include_attachments: bool = False) -> Set[str]:
        """
        Extract wikilinks from note content.

        Args:
            content: String content of a note
            include_attachments: Whether to include ![[wikilinks]] for images/attachments

        Returns:
            Set of unique wikilink texts (without the brackets)
        """
        pattern = self._attachment_pattern if include_attachments else self._wikilink_pattern
        matches = pattern.findall(content)
        return {self._process_wikilink(match) for match in matches}

    def get_note_wikilinks(self, note_path: Path, include_attachments: bool = False) -> Set[str]:
        """
        Get all wikilinks from a specific note.

        Args:
            note_path: Path to the note file
            include_attachments: Whether to include ![[wikilinks]] for images/attachments

        Returns:
            Set of unique wikilink texts found in the note

        Raises:
            FileNotFoundError: If the note doesn't exist
        """
        content = self.read_note(note_path)
        return self.extract_wikilinks(content, include_attachments)

    def get_all_wikilinks(self, include_attachments: bool = False) -> Dict[Path, Set[str]]:
        """
        Get all wikilinks from all notes in the vault.

        Args:
            include_attachments: Whether to include ![[wikilinks]] for images/attachments

        Returns:
            Dictionary mapping note paths to sets of unique wikilinks found in each note

        Raises:
            FileNotFoundError: If vault_path doesn't exist
        """
        return {
            note_path: self.get_note_wikilinks(note_path, include_attachments)
            for note_path in self.get_all_notes()
        }


class VaultWriter:
    """
    A class for writing to Obsidian vault files.
    
    This class provides methods to:
    - Append content to existing notes
    - Create new notes
    - Modify note content
    - Write wikilinks and tags in specific formats
    """

    def __init__(self, vault_path: Optional[Path] = None):
        """
        Initialize a VaultWriter instance.

        Args:
            vault_path: Path to the vault directory. If None, defaults to "../Mock Vault"
        """
        if vault_path is None:
            vault_path = Path(__file__).parent.parent / "Mock Vault"
        self.vault_path = vault_path

    def write_suggestions(self, note_path: Path, links: List[Dict[str, float]], topics: List[str], categories: List[str]) -> None:
        """
        Write suggested links, topics, and categories to a note in a structured format.

        Args:
            note_path: Path to the note file
            links: List of dictionaries with 'target' and 'score' keys
            topics: List of topic strings to write as #tag-topic
            categories: List of category strings to write as #cat-category
        """
        if not note_path.exists():
            raise FileNotFoundError(f"Note does not exist: {note_path}")

        try:
            content = self.read_note(note_path)
            
            # Build suggestions section
            suggestions_section = "\n\n## Agent Suggestions\n{\n"
            
            # Add links subsection with wikilinks
            suggestions_section += '  "suggested_links": [\n'
            for link in links:
                target = link["target"]
                # Convert absolute path to relative path
                if target.startswith(str(self.vault_path)):
                    target = target[len(str(self.vault_path)) + 1:]  # +1 to remove the leading slash
                # Remove .md extension
                if target.endswith('.md'):
                    target = target[:-3]
                suggestions_section += f'    {{\n      "link": "[[{target}]]",\n      "score": {link["score"]:.2f}\n    }},\n'
            suggestions_section = suggestions_section.rstrip(',\n') + '\n  ],\n'
            
            # Add topics subsection with #tag- format
            suggestions_section += '  "topics": [\n'
            for topic in topics:
                # Convert to lowercase and replace spaces with hyphens
                formatted_topic = topic.lower().replace(' ', '-')
                suggestions_section += f'    " #tag-{formatted_topic} ",\n'
            suggestions_section = suggestions_section.rstrip(',\n') + '\n  ],\n'
            
            # Add categories subsection with #cat- format
            suggestions_section += '  "categories": [\n'
            for category in categories:
                # Convert to lowercase and replace spaces with hyphens
                formatted_category = category.lower().replace(' ', '-')
                suggestions_section += f'    " #cat-{formatted_category} ",\n'
            suggestions_section = suggestions_section.rstrip(',\n') + '\n  ]\n'
            
            suggestions_section += "}\n"
            
            # Check if section already exists and replace if it does
            if "## Agent Suggestions" in content:
                start = content.find("## Agent Suggestions")
                end = content.find("}\n", start) + 2  # Changed to find the end of the JSON object
                content = content[:start] + suggestions_section.lstrip() + content[end:]
            else:
                content += suggestions_section
                
            # Write back to file
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except Exception as e:
            raise Exception(f"Failed to write suggestions to {note_path}: {str(e)}")

    def read_note(self, note_path: Path) -> str:
        """Read the contents of a note file."""
        with open(note_path, 'r', encoding='utf-8') as f:
            return f.read()

    def append_to_note(self, note_path: Path, content: str, add_newline: bool = True) -> None:
        """
        Append content to an existing note.

        Args:
            note_path: Path to the note file
            content: Content to append
            add_newline: Whether to add a newline before the content

        Raises:
            FileNotFoundError: If the note doesn't exist
            UnicodeDecodeError: If the file encoding is not UTF-8
        """
        if not note_path.exists():
            raise FileNotFoundError(f"Note does not exist: {note_path}")

        try:
            with open(note_path, 'a', encoding='utf-8') as f:
                if add_newline:
                    f.write('\n')
                f.write(content)
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(f"Failed to write to {note_path}: {str(e)}")

    def append_wikilinks(self, note_path: Path, links: List[Dict[str, str]], add_newline_between: bool = True) -> None:
        """
        Append multiple wikilinks to an existing note.

        Args:
            note_path: Path to the note file
            links: List of dictionaries, each containing:
                  - 'link': Text of the link (what the link points to)
                  - 'display': Optional display text for the link (can be None)
            add_newline_between: Whether to add newlines between multiple links

        Raises:
            FileNotFoundError: If the note doesn't exist
            UnicodeDecodeError: If the file encoding is not UTF-8
        """
        formatted_links = []
        for link_info in links:
            link_text = link_info['link']
            display_text = link_info.get('display')
            
            if display_text:
                wikilink = f"[[{link_text}|{display_text}]]"
            else:
                wikilink = f"[[{link_text}]]"
            formatted_links.append(wikilink)
            
        separator = '\n' if add_newline_between else ' '
        combined_links = separator.join(formatted_links)
        
        self.append_to_note(note_path, combined_links)

if __name__ == "__main__":
    # Example usage
    reader = VaultReader()
    
    try:
        # Print all notes found
        print("\nAll notes in vault:")
        for note_path in reader.get_all_notes():
            print(f"- {note_path.relative_to(reader.vault_path)}")
        
        # Print wikilinks found in each note
        print("\nWikilinks in each note:")
        for path, links in reader.get_all_wikilinks().items():
            if links:  # Only print if note contains wikilinks
                print(f"\n=== {path.relative_to(reader.vault_path)} ===")
                for link in sorted(links):
                    print(f"- [[{link}]]")
                    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
