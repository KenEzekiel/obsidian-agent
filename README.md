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
git clone https://github.com/yourusername/obsidian-agent.git
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

### Command Line Options

- `--include-attachments`: Include attachment links in the analysis
- `--output-dir`: Directory to save the graph file (default: data/)
- `--verbose`, `-v`: Enable verbose output

Example:
```bash
python src/main.py /path/to/your/vault \
    --include-attachments \
    --output-dir data/ \
    --verbose
```

## Project Structure

```
obsidian-agent/
├── src/
│   ├── core/
│   │   ├── vault_reader.py    # Vault reading and link extraction
│   │   └── graph_builder.py   # Graph building and analysis
│   ├── utils/
│   │   └── logger.py         # Logging utilities
│   └── main.py               # Main entry point
├── requirements.txt          # Project dependencies
└── README.md                # This file
```

## Development

### Adding New Features

1. Create a new branch for your feature
2. Implement your changes
3. Add tests if applicable
4. Submit a pull request

### Running Tests

```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.