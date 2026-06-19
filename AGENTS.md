# Repository Guidelines

## Primary Agent Instructions

Before working in this repository, read the repo skill first:

- `.agents/skills/universe-selector/SKILL.md`

This is the single source of truth for project structure, CLI workflows, ranking profiles, providers, valuation models, persistence, fixtures, and quality gates.

## Minimal Fallback

If your agent environment does not load skills, use these baseline rules:

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
