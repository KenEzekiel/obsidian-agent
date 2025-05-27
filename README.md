# Obsidian Agent

A tool for analyzing your Obsidian vault structure and generating graph representations of note connections.

## Features

- **Vault Analysis**: Extract wikilinks and connections from your Obsidian vault
- **Graph Building**: Generate a graph representation of note connections
- **Attachment Support**: Optionally include attachment links in the analysis
- **Configurable**: Flexible options for analysis and output

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KenEzekiel/obsidian-agent.git
cd obsidian-agent
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the analyzer with your Obsidian vault:

```bash
python src/main.py /path/to/your/vault
```

## Project Structure

```
obsidian-agent/
├── src/
│   ├── core/
│   │   ├── vault_reader.py    # Vault reading and link extraction
│   │   └── graph_builder.py   # Graph building and analysis
│   ├── services/
│   │   └── embedding_service.py    # Embedding conversion and storage
│   └── main.py               # Main entry point
├── requirements.txt          # Project dependencies
└── README.md                
```


Works by having a tool for similarity search, so it doesn't have to check pair-by-pair for links. Cons are it can miss some links, but much more efficient this way.

Improvement in langgraph --> pair analysis, and then critique then decide, tool call juga di langgraph

Improvement --> use note categories too

Improvement --> Analyzer for evaluator

Essentially the vault is like a simple MCP