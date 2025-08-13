#!/bin/bash
# Development setup script for RAG Topic Modeling

echo "Setting up development environment with UV..."

# Install UV if not present
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
fi

# Sync dependencies
echo "Installing dependencies..."
uv sync

echo "Development environment ready!"
echo "To activate the virtual environment, run: source .venv/bin/activate"
echo "Or run commands with: uv run <command>"