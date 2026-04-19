# Contributing to aiomtec2mqtt

Thank you for your interest in contributing! This guide covers the workflow
for code changes, reviews, and releases.

## Quick start

```bash
# 1. Fork and clone
git clone https://github.com/<your-user>/aiomtec2mqtt.git
cd aiomtec2mqtt

# 2. Create a virtualenv (Python 3.13+)
python -m venv .venv
source .venv/bin/activate

# 3. Install project + dev dependencies via uv (fast) or pip
pip install -U uv
uv pip install -e .
uv pip install -r requirements_test.txt

# 4. Install Git hooks via prek (reads .pre-commit-config.yaml)
prek install
```

## Development workflow

1. **Create a branch off `main`** — use descriptive names:
   `feature/short-topic`, `fix/short-topic`, `docs/short-topic`.
2. **Write/adjust tests first** when fixing a bug (test reproduces the bug
   before the fix). Aim to keep total coverage ≥ 85 %.
3. **Run checks locally** before pushing:

   ```bash
   prek run --all-files           # ruff, mypy, pylint, bandit, codespell, ...
   pytest --cov=aiomtec2mqtt      # tests with coverage gate
   ```

4. **Open a Pull Request** against `main`. Link issues if applicable.

## Coding standards

- Python 3.13+, full type hints, `from __future__ import annotations`.
- Line length **99**. See ROADMAP §2.2 for rationale.
- Docstrings: Google-style (enforced by ruff `D` rules).
- Public modules declare an explicit `__all__`.
- No new code without tests.

## Commit messages

Short, imperative summary on line 1 (≤ 72 chars), blank line, optional body
explaining _why_. Conventional Commits are welcome but not required.

## Pull Request checklist

- [ ] Tests added / updated and passing locally
- [ ] `prek run --all-files` clean
- [ ] No drop in coverage; threshold `fail_under = 85`
- [ ] CHANGELOG entry (if user-facing)
- [ ] Docs/README updated (if behavior changed)

## Reporting bugs

Open an issue with:

- Python version + OS
- aiomtec2mqtt version (`pip show aiomtec2mqtt`)
- M-TEC firmware version (if relevant)
- Reproduction steps + expected vs. actual behavior

## Security

Please do **not** open public issues for security findings — see
[SECURITY.md](./SECURITY.md).

## License

By contributing, you agree your work is released under the project's
LGPL license.
