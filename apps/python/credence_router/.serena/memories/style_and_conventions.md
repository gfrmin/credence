# Code Style and Conventions

## Formatting
- **Linter/formatter**: ruff
- **Line length**: 99 characters
- **Imports**: `from __future__ import annotations` at top of every module

## Naming
- snake_case for functions, methods, variables
- PascalCase for classes
- UPPER_SNAKE_CASE for module-level constants
- Private helpers prefixed with `_` (e.g., `_safe_ast_eval`, `_walk_node`, `_extract_expression`)

## Type Hints
- Full type annotations on function signatures
- Uses `Protocol` for interfaces (e.g., `Tool` is a `@runtime_checkable` Protocol)
- Uses `NDArray[np.float64]` for numpy arrays
- Uses `tuple[str, ...]` (modern style) rather than `Tuple[str, ...]`

## Docstrings
- Module-level docstrings explaining purpose and key design notes
- Class and method docstrings where non-obvious
- Concise style, not verbose

## Patterns
- Frozen dataclasses for value objects (`Answer`)
- Protocol classes for tool interface
- Functional helpers as module-level private functions
- Tests organised in classes (`TestRouterInit`, `TestRouterSolve`, etc.)
- Fixtures use `tmp_path` from pytest
- Test helpers prefixed with `_` (e.g., `_make_simple_tools`)

## Paradigm
- Prefers functional programming where possible
- No raw eval() — uses safe AST parsing for calculator
- Lazy imports for optional dependencies (e.g., compat layer)
