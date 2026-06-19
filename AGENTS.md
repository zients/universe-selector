# Repository Guidelines

## Minimal Fallback

Use these baseline rules:

- Install dependencies with `uv sync`.
- Run quality gates with:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
```

- Keep tests offline; pytest runs with sockets disabled.
- Do not commit local state such as `config.yaml`, `.universe-selector/`, `.venv/`, or `.worktrees/`.
- Preserve the command boundary: `batch` computes and persists ranking runs, `report` and `inspect` only read persisted successful runs, and `value` is live and ephemeral.
- Do not add profile-specific DuckDB columns; ranking profiles should persist profile-specific values through declared metric keys and `metrics_json`.
