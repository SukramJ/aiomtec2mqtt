# Release Process

## Versioning

`aiomtec2mqtt` follows [Semantic Versioning](https://semver.org/). The version
is declared in `pyproject.toml` under `[project].version`.

## Changelog automation

[Release Drafter](https://github.com/release-drafter/release-drafter) composes
the next release notes automatically from merged PRs. The configuration is in
`.github/release-drafter.yml`:

- PR titles following **Conventional Commits** (`feat:`, `fix:`, `chore:`,
  `docs:`, `refactor:`) get auto-labeled.
- Each label is assigned to a category (Features, Bug Fixes, Security,
  Maintenance, Documentation, Dependencies).
- Labels also drive the **version bump** (`major` / `minor` / `patch`).

## Steps

1. Merge all PRs destined for the release. Release Drafter keeps a draft up
   to date on the releases page.
2. Bump `version` in `pyproject.toml` and push to `main`.
3. Publish the drafted release on GitHub (tag is created automatically).
4. CI publishes the wheel/sdist to PyPI and the container image to GHCR
   (see `.github/workflows/`).

## Security releases

Coordinate with the disclosure window documented in `SECURITY.md`. Use the
`security` label on the PR to ensure the fix is called out in release notes.
