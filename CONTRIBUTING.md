# Contributing to GAAFpy

Thank you for your interest in contributing!

## Publishing a new release

New releases are published to [PyPI](https://pypi.org/project/gaafpy/) via the
**Publish GAAFpy to PyPI** GitHub Actions workflow, which is triggered manually.

### Prerequisites

| Requirement | Details |
|---|---|
| Repository secret `PYPI_API_TOKEN` | A PyPI API token scoped to the `gaafpy` project, stored in *Settings → Secrets and variables → Actions* |
| Write access to the repository | Required to create and push the release tag |

---

### Option 1 — GitHub web UI

1. Open the repository on GitHub and click the **Actions** tab.
2. In the left-hand sidebar select **Publish GAAFpy to PyPI**.
3. Click **Run workflow** (top-right of the run list).
4. Fill in the two inputs:
   | Input | Description |
   |---|---|
   | **version** | The new version number, e.g. `1.2.0` (must follow `MAJOR.MINOR.PATCH` format and be strictly greater than the current PyPI release) |
   | **dry_run** | Tick this box to validate and build only — the publish to PyPI and the release tag are skipped |
5. Choose the branch to run from (usually `main`) and click **Run workflow**.

---

### Option 2 — GitHub CLI

Install the [GitHub CLI](https://cli.github.com/) and authenticate with `gh auth login`, then:

**Dry run (safe to run at any time — no PyPI publish, no tag)**

```bash
gh workflow run publish.yml \
  --field version=1.2.0 \
  --field dry_run=true
```

**Real release**

```bash
gh workflow run publish.yml \
  --field version=1.2.0 \
  --field dry_run=false
```

Follow the run in your browser:

```bash
gh run watch
```

---

### What each step does

| Step | Dry run | Real run |
|---|---|---|
| Validate version format | ✅ | ✅ |
| Check version > latest PyPI release | ✅ | ✅ |
| Update version in `pyproject.toml` | ✅ | ✅ |
| Build wheel + sdist | ✅ | ✅ |
| Verify build artifacts | ✅ | ✅ |
| Publish to PyPI | ⏭ skipped | ✅ |
| Create & push `v<version>` tag | ⏭ skipped | ✅ |

---

### Version ordering rule

The workflow will refuse to publish if the supplied version is not **strictly greater than** the currently released version on PyPI. For example, if `1.0.0` is the latest release:

| Input | Result |
|---|---|
| `1.0.1` | ✅ Allowed |
| `2.0.0` | ✅ Allowed |
| `1.0.0` | ❌ Rejected — same version |
| `0.9.9` | ❌ Rejected — older version |
