# Release Process

This document describes the automated release process for aiomtec2mqtt.

## Overview

The project uses GitHub Actions for automated releases and PyPI publishing. The process consists of two main workflows:

1. **release-on-tag.yml** - Creates GitHub Releases from tags
2. **python-publish.yml** - Publishes packages to PyPI

## Release Workflow

### 1. Prepare the Release

**Update CHANGELOG.md:**

Add a new version section following the format:

```markdown
## [1.0.0] - 2026-01-20

### Added

- New feature descriptions

### Changed

- Changed feature descriptions

### Fixed

- Bug fix descriptions
```

**Update Version:**

Update the version in `aiomtec2mqtt/__init__.py`:

```python
__version__ = "1.0.0"
```

**Commit Changes:**

```bash
git add CHANGELOG.md aiomtec2mqtt/__init__.py
git commit -m "Prepare release 1.0.0"
git push origin main
```

### 2. Create and Push Tag

Create a tag for the release:

```bash
# Using annotated tag (recommended)
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Or lightweight tag
git tag v1.0.0
git push origin v1.0.0
```

**Tag Format:**

- `v1.0.0` (with 'v' prefix) - recommended
- `1.0.0` (without 'v' prefix) - also supported

### 3. Automated Process

Once the tag is pushed:

1. **release-on-tag.yml triggers:**

   - Extracts release notes from CHANGELOG.md
   - Creates a GitHub Release with the extracted notes
   - Triggers python-publish workflow via repository_dispatch

2. **python-publish.yml triggers:**
   - Builds wheel and sdist distributions
   - Publishes to PyPI using Trusted Publishing

## Workflow Details

### release-on-tag.yml

**Trigger:** Push of tags matching `v*` or `[0-9]+.[0-9]+.[0-9]+`

**Steps:**

1. Checkout repository
2. Extract version from tag name (strips leading 'v')
3. Extract release notes from CHANGELOG.md
   - Searches for section: `## [VERSION] - DATE`
   - Extracts content until next version section
   - Falls back to default notes if section not found
4. Determine previous version for comparison link
5. Create GitHub Release
6. Trigger python-publish workflow

**Requirements:**

- CHANGELOG.md must exist at repository root
- Version sections must follow format: `## [1.0.0] - 2026-01-20`

### python-publish.yml

**Triggers:**

- GitHub Release published
- repository_dispatch event: `release-on-tag-succeeded`

**Jobs:**

1. **release-build:**

   - Sets up Python 3.13
   - Installs build dependencies
   - Builds wheel and sdist: `python -m build`
   - Uploads distributions as artifact

2. **pypi-publish:**
   - Downloads built distributions
   - Publishes to PyPI using Trusted Publishing (OIDC)
   - Requires `pypi` environment configured in GitHub

**Requirements:**

- PyPI Trusted Publishing configured for repository
- GitHub environment named `pypi` with appropriate settings

## PyPI Trusted Publishing Setup

Trusted Publishing uses OpenID Connect (OIDC) to securely publish packages without long-lived tokens.

### Configuration Steps:

1. **On PyPI (pypi.org):**

   - Go to project settings (or create new project)
   - Navigate to "Publishing" section
   - Add a new "Trusted Publisher"
   - Fill in:
     - Owner: `SukramJ`
     - Repository: `aiomtec2mqtt`
     - Workflow: `python-publish.yml`
     - Environment: `pypi`

2. **On GitHub:**

   - Go to repository Settings → Environments
   - Create environment named `pypi`
   - (Optional) Add protection rules like required reviewers

3. **First Release:**
   - If package doesn't exist on PyPI yet, create it manually:
     ```bash
     pip install build twine
     python -m build
     twine upload dist/*
     ```
   - Then configure Trusted Publishing for future releases

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 1.0.0)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Testing the Workflow

### Test Tag (without publishing):

You can test the release workflow without publishing to PyPI:

1. Create a test tag with suffix:

   ```bash
   git tag v1.0.0-test
   git push origin v1.0.0-test
   ```

2. The workflow will create a GitHub Release but won't publish to PyPI

### Dry Run:

For local testing of the build process:

```bash
# Build distributions locally
python -m pip install build
python -m build

# Check distributions
ls -lh dist/

# Validate with twine
pip install twine
twine check dist/*
```

## Rollback Procedure

If a release needs to be rolled back:

1. **Delete the GitHub Release:**

   - Go to repository Releases
   - Delete the problematic release

2. **Delete the Git Tag:**

   ```bash
   git tag -d v1.0.0
   git push origin :refs/tags/v1.0.0
   ```

3. **PyPI Package:**
   - PyPI doesn't allow deleting releases
   - Instead, yank the release (marks it as unavailable)
   - Release a new fixed version

## Troubleshooting

### Release Notes Not Found

**Error:** "No matching section found in CHANGELOG.md"

**Solution:**

- Ensure CHANGELOG.md contains section: `## [1.0.0] - 2026-01-20`
- Version must exactly match tag (without 'v' prefix)
- Check that section follows Keep a Changelog format

### PyPI Publishing Failed

**Error:** "Trusted publishing exchange failure"

**Solution:**

- Verify Trusted Publisher is configured on PyPI
- Check environment name matches: `pypi`
- Ensure workflow file path is correct: `python-publish.yml`
- Verify repository owner matches: `SukramJ`

### Build Failed

**Error:** Build process errors

**Solution:**

- Check pyproject.toml is valid
- Ensure version is set correctly in `__init__.py`
- Verify all dependencies are listed
- Test build locally first

## Manual Release (Emergency)

If automated workflows fail, you can release manually:

```bash
# 1. Build
python -m pip install build twine
python -m build

# 2. Check
twine check dist/*

# 3. Upload to PyPI
twine upload dist/*

# 4. Create GitHub Release manually via web interface
```

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
