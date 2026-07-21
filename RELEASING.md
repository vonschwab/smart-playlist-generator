# Releasing MixArc

MixArc publishes to PyPI via GitHub Actions using **trusted publishing (OIDC)** — no API token is
stored anywhere. Publishing is wheel-only (the wheel bundles the built web UI).

## One-time setup (do this once, before the first release)

1. Create a free account at https://pypi.org (and https://test.pypi.org for rehearsals).
2. On PyPI → your account → **Publishing** → **Add a pending publisher**, enter exactly:
   - PyPI Project Name: `mixarc`
   - Owner: `vonschwab`
   - Repository name: `mixarc`
   - Workflow name: `publish.yml`
   - Environment name: *(leave blank)*
3. Repeat step 2 on TestPyPI (https://test.pypi.org) so the rehearsal can publish there.
   > Note: `mixarc` on PyPI is first-come. Confirm the name is still free before relying on it.

## Rehearse on TestPyPI (recommended before the first real publish)

1. GitHub → Actions → **Publish to PyPI** → **Run workflow** → target `testpypi`.
2. When it's green, in a clean venv: `pip install -i https://test.pypi.org/simple/ mixarc` then `mixarc --help`.

## Cut a real release

1. Bump `version` in `pyproject.toml` (first release: already `0.1.0`).
2. Add a one-line entry to `CHANGELOG.md`.
3. Commit + push to `master`.
4. GitHub → **Releases** → **Draft a new release** → tag `v0.1.0` → **Publish release**.
   The `publish.yml` workflow builds the wheel and publishes it to PyPI automatically.
5. Verify: in a clean environment, `pipx install mixarc` then run `mixarc` — the wizard opens.

## Warnings

- **PyPI versions are immutable** — you can never re-upload the same version number. Bump the version for any change.
- If a publish fails after upload, bump to the next patch version and re-release; don't try to overwrite.
