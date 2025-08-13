# UV Commands Reference

## Basic Commands

- `uv sync` - Install all dependencies from pyproject.toml
- `uv run <command>` - Run command in UV virtual environment
- `uv add <package>` - Add a new dependency
- `uv remove <package>` - Remove a dependency
- `uv lock` - Update the lock file
- `uv python install <version>` - Install a specific Python version
- `uv venv` - Create virtual environment manually

## Running the Application

```bash
# Run Streamlit app
uv run streamlit run app.py

# Run Python script
uv run python run.py

# Install dependencies
uv sync
```

## Development Workflow

1. Add dependencies: `uv add package-name`
2. Install/sync: `uv sync`
3. Run commands: `uv run command`
4. Activate environment (optional): `source .venv/bin/activate`

## Migration Notes

- Replaced `requirements.txt` with `pyproject.toml`
- Virtual environment automatically managed in `.venv/`
- Faster dependency resolution and installation
- Better lock file management with `uv.lock`