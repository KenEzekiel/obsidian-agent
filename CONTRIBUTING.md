# Contributing to Obsidian Agent

Thank you for your interest in contributing! Here are some guidelines to help you get started.

## Getting Started

1. **Fork the repository** and clone your fork.
2. **Install Poetry** (if not already):
   ```bash
   pip install poetry
   ```
3. **Install dependencies**:
   ```bash
   poetry install
   ```
4. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b my-feature
   ```

## Code Style

- Use [black](https://black.readthedocs.io/en/stable/) for formatting.
- Use [flake8](https://flake8.pycqa.org/en/latest/) for linting.
- Type annotations are required for all public functions.
- Write clear docstrings for all public classes and methods.

## Running Tests

- Add tests for new features or bugfixes.
- Run tests with:
  ```bash
  poetry run pytest
  ```

## Pull Requests

- Ensure your branch is up to date with `main`.
- Open a pull request with a clear description of your changes.
- Reference related issues in your PR description.
- One feature/fix per PR.

## Community

- Be respectful and constructive in discussions.
- For questions, open a GitHub Discussion or Issue.

Thank you for helping make Obsidian Agent better! 