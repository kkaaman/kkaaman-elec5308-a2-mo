# Contributing: Group 7, Assignment 2

## Branches

- `main` is protected. Changes reach it only through reviewed pull requests
  with green CI.
- Use one branch per task, named `<type>/<TASK-ID>-short-slug`. Types are
  `feat`, `fix`, `test`, `docs`, `refactor` and `chore`. Examples:
  `feat/BL-F-01-contracts`, `fix/RV-03-lane-shims`.
- Branch from the latest `main`. Merge `main` into your branch again before
  asking for review.
- You can work from a personal fork (as in A1) or from a branch in this
  repository.

## Pull requests

- **Title:** `[TASK-ID] Task title`, copied from the task board. Aim for one
  task per PR. Combine tasks only when they can't be tested apart.
- **Template:** fill in the PR template: acceptance criteria from the board,
  verification commands and evidence.
- **Reviewer:** the reviewer named on the task board, by default your review
  partner. Review before any dependent task starts.
- **Merge conditions:** merge only when all of these hold:
  - CI is green (`lint`, `test`, `docs`).
  - The reviewer has approved.
  - Simulator evidence is linked for tasks with `CARLA Required = Y`.
- **Merge method:** use **Create a merge commit**. It keeps every author's
  commits and the `Task:` trailers, as in A1.
- **Public APIs:** keep A1 public imports and constructors working. If an API
  has to change, add a compatibility shim and a test for it in the same PR.

## Commits

- Subject: imperative mood, 72 characters or fewer, optionally prefixed with
  `[TASK-ID]`.
- Add a trailer line `Task: TASK-ID` to the commit body.
- **AI disclosure (course requirement).** Every AI-assisted commit ends with:

  ```text
  AI-assisted: <tool> – <what>; reviewed and edited by <name>
  ```

  Also log significant interactions in your private AI Journal while you
  work, not afterwards.
- Never commit secrets, CARLA binaries or datasets. Model weights (`*.pth`) go
  through Git LFS, which `.gitattributes` already sets up. Record the SHA256
  of any new checkpoint in `docs/`.
- Before pushing, run:

  ```bash
  ruff check .
  python -m pytest
  sphinx-build -W docs docs/_build/html
  ```

## Design rules agreed for A2

- **Simulator boundary.** Only `src/carla_adapter/` and runtime entry
  scripts import `carla`. Core packages take and return plain data.
- **Evaluation boundary.** Ground truth from the simulator (actor poses,
  boxes, light states) feeds evaluation and metrics only, never the
  obstacle-control decision.
- **Offline by default.** The baseline must start without network access.
  Optional cloud features stay behind flags that default to off.
- **Evidence.** Never edit raw run outputs. Keep failed attempts as well.
  Results start as `NOT RUN` until a saved run supports them.

## Repository settings (admin, once)

Add a branch protection rule (or ruleset) for `main`:

- Require a pull request before merging, with 1 approval.
- Require the status checks `lint`, `test` and `docs` to pass, and require
  branches to be up to date.
- Block force pushes and deletions.
- Allow merge commits. Squash merging can be left on, but merge commits are
  preferred.

Then confirm that all six members can open PRs, either with write access here
or from a fork.
