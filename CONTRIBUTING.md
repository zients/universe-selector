# Contributing

Thanks for your interest in contributing to Universe Selector. This is an
alpha-stage research tool; contributions, bug reports, and ideas are welcome.

## Development Setup

Requirements:

- Python `>=3.12,<3.15`
- [`uv`](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/zients/universe-selector.git
cd universe-selector
uv sync
cp config.example.yaml config.yaml
```

For a network-free smoke run, set `data_mode: fixture` in `config.yaml`. See the
Fixture Smoke Run section in the [README](README.md).

## Quality Gates

Run the same checks CI runs before opening a pull request:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

CI runs these on Python 3.12 and 3.14 and also builds the wheel and source
distribution.

## Tests

- Tests must run offline. The suite runs with sockets disabled
  (`--disable-socket`); use the fixtures under `tests/fixtures/` instead of live
  network calls.
- Add tests for new behavior. New ranking profiles and providers should cover
  validation, construction, persistence, report, and inspect behavior, as
  described in [docs/extending.md](docs/extending.md).

## Pull Requests

- Keep changes focused and accompanied by tests.
- Make sure all quality gates pass locally.
- Describe what changed and why in the pull request description.

## Extending

Universe Selector is built around two extension points — ranking profiles and
providers. See [docs/extending.md](docs/extending.md) for the contracts and the
required registration and test steps.
