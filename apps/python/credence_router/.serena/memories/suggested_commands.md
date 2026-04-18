# Suggested Commands

## Setup
```bash
uv sync --all-groups          # Install all dependencies including optional and dev
```

## Testing
```bash
pytest tests/                 # Run all tests
pytest tests/test_router.py   # Run specific test file
pytest -k "test_name"         # Run specific test by name
```

## Linting & Formatting
```bash
ruff check src/ tests/        # Lint check
ruff check src/ tests/ --fix  # Lint with auto-fix
ruff format src/ tests/       # Format code
```

## Running the Application
```bash
# Simulated benchmark (no API keys needed)
credence-router bench --run --simulate

# Full benchmark (needs ANTHROPIC_API_KEY, PERPLEXITY_API_KEY)
credence-router bench --run

# Interactive routing
credence-router route "What is 2+2?" -o "3" "4" "5" "6"
```

## System Utilities
```bash
git status / git log / git diff    # Standard git commands
ls / cd / find / grep              # Standard Linux filesystem commands
```
