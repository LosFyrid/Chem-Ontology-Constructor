# Repository Guidelines

## Project Structure & Module Organization
- `src/` — core Python package: `agents/`, `ontology/`, `parsers/`, `utils/`.
- `tests/` — pytest suites (unit and scenario folders such as `unit_test/`, `Supramolecular/`).
- `config/` — runtime/config templates; keep secrets in `.env` only locally.
- `data/` — input corpora and ontology artifacts; treat as read‑only in tests.
- `logs/`, `test_results/` — generated outputs and run artifacts.
- Top‑level scripts (e.g., `workflow_test.py`, `vanilla_llm.py`) demonstrate end‑to‑end flows.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && .\\.venv\\Scripts\\activate`
- Install dev deps: `pip install -r requirements-dev.txt`
- Run tests: `pytest -q` (subset: `pytest tests/unit_test -q`)
- Lint/format (recommended): `ruff check src tests` and `black src tests` if available.
- Run demos: `python workflow_test.py` or `python vanilla_llm.py`

## Coding Style & Naming Conventions
- Python 3.10+; PEP 8 with 4‑space indent; prefer type hints.
- Files/modules: `snake_case.py`; classes: `PascalCase`; functions/vars: `snake_case`.
- Docstrings: concise summary + args/returns where non‑trivial.
- Keep pure logic in `src/**`; scripts only orchestrate.

## Testing Guidelines
- Framework: pytest with `pytest-mock`, `freezegun`.
- Test files: `tests/**/test_*.py`; name tests for behavior (e.g., `test_parses_phosphates_when_ph_range_valid`).
- Use fixtures; avoid hitting network or mutating `data/`. Write temp files under `tmp_path`.
- Aim for meaningful coverage of critical paths in `src/agents` and `src/parsers`.

## Commit & Pull Request Guidelines
- Commits: imperative mood, short scope, clear intent.
  - Examples: `feat(agents): add team supervisor router`, `fix(parsers): handle empty section titles`.
- PRs: include purpose, key changes, test evidence (`pytest -q` output), and any config notes. Link issues where relevant.

## Agent-Specific Instructions
- Place reusable agent utilities in `src/agents/` (e.g., `helper_utilities.py`); domain teams under subfolders (e.g., `builder_team/`).
- Keep prompts/templates close to the agent code; put environment/config values in `config/` and load via `.env`.
- Required envs (example): set `OPENAI_API_KEY` in `.env`; never commit secrets.
