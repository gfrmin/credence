# Task Completion Checklist

When a coding task is completed, run the following in order:

1. **Format**: `ruff format src/ tests/`
2. **Lint**: `ruff check src/ tests/` (fix any issues with `--fix` or manually)
3. **Test**: `pytest tests/`

All three must pass before considering the task done.
