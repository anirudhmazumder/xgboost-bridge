# Releasing

What a maintainer runs, in what order, and what each guard refuses and why.

This exists because the knowledge was in `docs/DECISIONS.md` entries and in the head
of whoever did the last release. The next release may be far enough out that neither
is available.

Both registries publish through **Trusted Publishing**. There is no npm token and no
PyPI token in this repository, and there never has been one — that is a property to
preserve, not a convenience.

---

## Before you start

Nothing here is reversible. A version number spent on PyPI or npm cannot be reused,
even after deletion — both registries refuse re-upload of a filename permanently. So
the order below front-loads everything that can fail for free.

| You need | Why |
|---|---|
| Write access to the repository | To push the version bump |
| Reviewer rights on the `release` environment | The publish jobs wait on it |
| Nothing else | No credentials are involved on your side |

---

## 1. Bump the version, in one place per package

```bash
# Python: the ONLY literal. pyproject.toml reads it (D057).
$EDITOR packages/python/src/xgboost_bridge/_version.py

# npm: semver spelling, so 1.0.0rc1 becomes 1.0.0-rc.1
$EDITOR packages/js/package.json
```

The two spellings differ deliberately — PyPI wants PEP 440 (`1.0.0rc1`), npm wants
semver (`1.0.0-rc.1`). `tools/check_dist_tags.mjs` holds one predicate for each and
a test asserts they agree about every version this project has shipped.

## 2. Regenerate the fixture corpus

```bash
uv sync
cd fixtures && uv run python -m generate.corpus && uv run python -m generate.adversarial && cd ..
```

Every fixture embeds `provenance.exporter_version`, so **all 23 change on every
bump**. Two tests fail until you do this, which is deliberate: a release cannot ship
with fixtures claiming the previous version (D057).

## 3. Prove that only provenance moved

Not optional, and not a formality. The bump touches 23 files that carry 299 margin
bit patterns, 299 output bit patterns and 3258 node values. Capture before and after,
then diff with `provenance.exporter_version` removed: the remainder must be
**byte-identical**. Every release so far has been proven this way (D057, D061, D062).

If anything else moved, stop. A regenerated corpus that differs numerically means the
exporter changed behaviour, and that is a finding rather than a release.

## 4. Run the full gate locally

```bash
npm --prefix packages/js run build      # FIRST: the parity harness refuses a stale bundle
uv run pytest                            # 1058 tests
npm --prefix packages/js run typecheck   # separate from the build, per D011
npm --prefix packages/js test            # 174 tests, against dist/
uv run python parity/run_parity.py       # 299 rows, both measurement points
uv run python parity/run_parity_scale.py # 100,000 adversarial rows
./tools/clean_install_python.sh          # wheel contents, then install and predict
./tools/clean_install_js.sh              # tarball contents, then install and predict
```

Build the JavaScript bundle **before** `pytest`. The parity harness refuses a
`dist/` older than `src/` rather than measuring a stale bundle, and 23 tests fail
with a staleness message that looks like a parity failure if you get this wrong.

Both parity numbers must be exactly `0.0`. A small nonzero value is a bit-level
defect, not a tolerance — diagnose it (`CLAUDE.md`).

## 5. Commit and push

The version bump and the regenerated corpus go together. CI runs on
`workflow_dispatch`; push-triggered runs arrive late or not at all, so dispatch it:

```bash
gh workflow run ci.yml --ref main
```

## 6. Dispatch the release

```bash
gh workflow run release.yml --ref main \
  -f confirm=publish -f targets=both -f allow_partial=false
```

Then approve the `release` environment when it asks. It asks twice — once per publish
job.

**For a release candidate**, use TestPyPI first:

```bash
gh workflow run release-testpypi.yml --ref main -f confirm=rehearse
```

That publishes to TestPyPI and then proves the artifact came from there: metadata
lookup, host assertion, sha256 match against the digest TestPyPI recorded,
`--no-index` install of that exact file, and a PEP 610 `direct_url.json` check. It
carries `skip-existing`, so you can re-run it without spending a version.

---

## What the guards refuse, and why

| Guard | Refuses | Why it exists |
|---|---|---|
| `verify` runs the whole gate | Anything the suites catch | The gate runs in a job with **no** credential, so a dependency executing an install script cannot mint a publishing token (D054) |
| A non-prerelease must use `targets=both` | `1.0.0` with `npm-only` | Half-publishing a real release. Override with `allow_partial=true` **only** to complete a split |
| Version floors in the npm job | Node < 22.14.0, npm < 11.5.1 | Trusted Publishing needs both. Without the check the registry returns a token-shaped error that names nothing useful |
| No credential configured | An `.npmrc` token or `NODE_AUTH_TOKEN` | npm sends a found credential *instead of* doing the OIDC exchange. `setup-node`'s `registry-url` writes a placeholder token that does exactly this (D062) |
| `verify-dist-tags` | `latest` not pointing at a final release | `npm install <pkg>` resolves `latest`. An RC left there is a silent wrong-version-installed path |
| Fixture provenance tests | A corpus claiming the previous version | Stale fixtures are a mislabelled record, and a mislabelled record is a record of nothing (D057) |

## If it fails

**Diagnose before remediating.** Remediation is irreversible here, and the failure
you can see is not always the failure you have. On the 1.0.0 release the run showed
`success / failure` across two publish jobs — indistinguishable from a split release
— and **both packages had published correctly**; the broken thing was the
verification step. Measuring both registries first is what told them apart:

```bash
curl -s https://pypi.org/pypi/xgboost-bridge/json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"
npm view xgboost-predictor dist-tags --json
```

Then, by case:

- **`verify` failed** — nothing published. Fix and re-dispatch freely.
- **`publish-python` failed** — nothing published; the npm job is skipped by design,
  because its condition requires the PyPI job to have succeeded *or been skipped*,
  never failed.
- **`publish-python` succeeded, `publish-javascript` failed** — this is the split.
  The version is spent on PyPI. Do **not** re-dispatch `both`: it will fail on the
  duplicate before reaching npm. Fix the npm-side cause, then
  `-f targets=npm-only -f allow_partial=true`.
- **Both published, a later job failed** — the release is complete. Fix the check,
  and do not re-publish anything.

## After a release

- Verify from outside the repository, not from the workflow's output:

  ```bash
  uv pip install "xgboost-bridge==<version>"        # into an empty venv
  npm install xgboost-predictor                      # into an empty project
  ```

  Then predict against a committed fixture and compare bit patterns.
- Update `VERIFICATION.md` (the release table and the test counts) and `COMPAT.md`.
- Add a `docs/DECISIONS.md` entry if anything was learned. Every release so far has
  taught something, and all of it was in the release mechanics rather than the
  library.

## The one thing to remember

The refusals are the published contract. **Narrowing what either package accepts is a
breaking change**, which is why D055 and D058 tightened several refusals deliberately
*before* 1.0.0 rather than after. If a release would need to refuse something new,
that is a major version.
