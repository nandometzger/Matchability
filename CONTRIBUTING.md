# Contributing

## Development setup

```bash
pip install -e ".[dev,viz]"   # torch + kornia come in as core deps
pytest -m "not dedode"        # fast unit + property tests (no model weights)
pytest -m dedode              # slow: loads the real DeDoDe v2 model (downloads weights once)
ruff check .
```

## Conventional Commits (required)

Commit messages drive automated versioning via
[release-please](https://github.com/googleapis/release-please). Use the
[Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(optional scope): <description>
```

| Type        | Version effect       | Example                                            |
| ----------- | -------------------- | -------------------------------------------------- |
| `feat:`     | minor (0.x.0)        | `feat: add elastic-warp distortion`                |
| `fix:`      | patch (0.0.x)        | `fix: guard empty-union case in matchability_error`|
| `feat!:` / `BREAKING CHANGE:` | major (x.0.0) | `feat!: rename tau to epipolar_threshold` |
| `docs:`     | none (changelog only)| `docs: document working-resolution default`        |
| `test:`     | none                 | `test: cover vertical-shift trend`                 |
| `ci:`       | none                 | `ci: cache DeDoDe weights`                         |
| `refactor:` | none                 | `refactor: extract jaccard helper`                 |
| `chore:`    | none                 | `chore: bump ruff`                                 |

Pre-1.0, `feat!` bumps the minor (0.x.0) rather than the major.

## Test-Driven Development

This project is built test-first. Every behavior change starts with a failing
test. New distortions, matcher backends, or metric options must come with tests.
