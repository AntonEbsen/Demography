# GitHub Infrastructure

This directory manages the automated CI/CD and repository health workflows.

## Workflows

- `test.yml`: Runs the Python test suite (Pytest), linter (Ruff), and type-checker (MyPy) on every push to `main`.
- `deploy.yml`: Handles the automated deployment of the Astro sites to GitHub Pages.
- `dependabot.yml`: Manages weekly dependency updates to keep the research environment secure.

## Automation Goals

The goal of this infrastructure is to ensure that the research remains **executable and accurate** over long time horizons, satisfying the highest standards of scientific reproducibility.
