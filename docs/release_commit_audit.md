# Commit history audit — 31 October 2026

This audit applies the new policy prospectively. Published main and contributor histories were not rewritten. No force push, rebase, squash or cosmetic commit-message rewrite occurred.

## New commits under this policy

| Commit | Subject | Status |
|---|---|---|
| dc10656 | docs(review): refresh runtime disclosure after human inspection | PASS |

The commit publishing this audit uses `docs(release): audit commit history and refresh demo scope`; its subject and complete new-commit range are validated again before push, and its exact SHA is recorded in the PR117 audit comment/final response. No new PR merge or local merge commit was created in this policy pass. Any later PR merge must explicitly use `merge: <description> PR #N`.

## Historical published exceptions

Canonical main contains **281 commits**. **114** historical subjects lack the Conventional Commit shape; another **40** historical merge subjects are conventionally shaped but do not satisfy the new exact PR-merge title rule. Thus **154** old commits fail the combined prospective subject/merge-title audit. They predate this enforcement policy and remain unchanged to preserve SHA, admission, evidence and contributor references. The complete 114-subject list and full combined audit are retained in the adjacent JSON artifacts.

| Historical main commit | Subject |
|---|---|
| cf970f9 | Merge pull request #115 from USYD-ELEC5308/codex/40-lidar-geometry-investigation |
| 3f8a348 | Validate front LiDAR geometry for issue40 near-field support |
| 8d9a797 | Merge pull request #114 from USYD-ELEC5308/codex/live-session-20261029 |
| 5111209 | Merge pull request #103 from USYD-ELEC5308/feat/E2-01-04-lidar-obstacle-path |
| 138bf98 | Merge remote-tracking branch 'upstream/main' into codex/phase-j-lidar-validation |
| d0fc56f | Merge pull request #113 from USYD-ELEC5308/codex/phase-j-final-execution |
| 9de3755 | Merge pull request #112 from USYD-ELEC5308/test/63-offline-fault-suite |
| 6c5b9b6 | Merge remote-tracking branch 'upstream/main' into codex/phase-j-faults |
| ed07850 | Merge pull request #111 from RickyPurani/feat/e1-03-weather-evaluator |
| 0b1f9b0 | Merge remote-tracking branch 'upstream/main' into codex/phase-j-faults |
| 2542c96 | Merge remote-tracking branch 'upstream/main' into codex/phase-j-weather-eval |
| 9cc45d6 | Merge pull request #102 from USYD-ELEC5308/feat/E3-01-03-wayfinder-v1 |

## Published topic-branch exceptions

The pre-push range `upstream/main..HEAD` was inspected, not just the newest commit. PR99 has ten pre-policy exceptions and PR101 has fifteen (including shared inherited commits); their SHA/subject/merge classification is retained in `commit-audit-feat_*.json`. Examples include `2b1c69e Add shared perception-control cycle with fault handling`, `16a0bc4 Add gated CARLA first-stop evidence runner` and `e9f644a docs(RV-08), feat(TEST-17, TEST-05): ...`. These are already published contributor history, not new policy-controlled failures; no force rewording was performed. The three already-published PR117 commits before this pass all satisfy Conventional Commits.

Manual auditing is sufficient for this pass. No new commitlint dependency or repository hook is introduced. Policy-controlled commits must all PASS; existing published exceptions are reported rather than hidden or cosmetically rewritten.
