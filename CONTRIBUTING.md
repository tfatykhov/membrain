# Contributing to Membrain

Thank you for your interest in contributing to Membrain! This document provides guidelines for contributing.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/membrain.git
   cd membrain
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Ensure all tests pass: `pytest tests/ -v`
   - Run linting: `ruff check src/`
   - Run type checking: `mypy src/`

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

4. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a PR on GitHub.

## Code Style

- We use **Ruff** for linting and formatting
- Line length: 100 characters
- Type hints are required for public functions
- Docstrings follow Google style

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Adding tests
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

## Testing

- All new features must have tests
- Aim for >80% code coverage
- Run tests with: `pytest tests/ -v --cov=src/membrain`

## Questions?

Open an issue or reach out to the maintainers.
