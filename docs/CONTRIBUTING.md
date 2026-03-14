# Contributing to FinCrime-LLM

Thank you for your interest in contributing! This guide will help you get started.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/FinCrime-LLM.git
   cd FinCrime-LLM
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements_dev.txt
   ```
4. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. Make your changes
2. Run the test suite:
   ```bash
   pytest tests/ -v
   ```
3. Format your code:
   ```bash
   black . --line-length 100
   ```
4. Run linting:
   ```bash
   flake8 --max-line-length 100
   mypy training/ inference/ api/
   ```
5. Commit your changes with a clear message
6. Push to your fork and open a Pull Request

## Code Standards

- **Python**: 3.11+
- **Type hints**: Required for all function signatures
- **Docstrings**: Google-style
- **Formatting**: Black (line length 100)
- **Linting**: flake8, mypy

## Areas for Contribution

- Improving training scripts and data pipelines
- Adding support for more African countries and regulatory frameworks
- Enhancing evaluation metrics and benchmarks
- Writing tests and documentation
- Bug fixes and performance improvements

## Reporting Issues

Please use [GitHub Issues](https://github.com/PatrickAttankurugu/FinCrime-LLM/issues) to report bugs or request features. Include:
- A clear description of the issue
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Your environment (OS, Python version, GPU)
