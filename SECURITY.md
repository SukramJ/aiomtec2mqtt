# Security Policy

## Supported Versions

Security fixes are applied to the latest minor release. Older versions may
receive backports at the maintainers' discretion.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Please do **not** file a public GitHub issue for security-relevant findings.

Instead, report privately via GitHub Security Advisories:

<https://github.com/SukramJ/aiomtec2mqtt/security/advisories/new>

Alternatively, contact the maintainer directly via the email address listed
on the maintainer's GitHub profile.

Please include:

- A description of the issue and its impact
- Steps to reproduce (ideally a minimal PoC)
- Affected version(s) and environment details
- Your proposed fix, if any

## Response Expectations

- Acknowledgement within **5 business days**
- Triage and initial assessment within **10 business days**
- Fix timeline depends on severity; critical issues are prioritized

After a fix lands, we coordinate public disclosure (CVE assignment if
applicable) with the reporter.

## Release Integrity

Every release attaches:

- **CycloneDX SBOM** (`*.cdx.json` and `*.cdx.xml`) listing every dependency
  resolved into the wheel.
- **Sigstore bundles** (`*.sigstore.json`) — keyless OIDC signatures over
  every wheel and sdist, produced by GitHub Actions inside a hardened
  workflow.

Verify a downloaded artifact with the `sigstore-python` CLI:

```bash
pip install sigstore
sigstore verify github \
  --bundle aiomtec2mqtt-1.0.0-py3-none-any.whl.sigstore.json \
  --cert-identity "https://github.com/SukramJ/aiomtec2mqtt/.github/workflows/python-publish.yml@refs/tags/v1.0.0" \
  --cert-oidc-issuer "https://token.actions.githubusercontent.com" \
  aiomtec2mqtt-1.0.0-py3-none-any.whl
```

A successful verification proves the wheel was built and signed by the
official release workflow at the named tag.

## Scope

In-scope:

- `aiomtec2mqtt` source code (package, CLI, util tools)
- Default configuration and install scripts
- Released wheels / sdists on PyPI

Out-of-scope (report upstream):

- `PyModbus`, `aiomqtt`, `pydantic`, `prometheus-client`, `pyyaml`
- MQTT broker / Home Assistant / evcc vulnerabilities
