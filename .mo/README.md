# Private parallel workflow (not delivered to the team repo)

| Remote (any name; the tool finds them by URL) | Repo | Role |
|---|---|---|
| `mine` on your PC, `origin` for Claude | `kkaaman/kkaaman-elec5308-a2-mo` (private) | Full implementation; Claude pushes here |
| `team` | `USYD-ELEC5308/26-S2-Assignment2-Group7` | Receives individual tasks as PRs under your name |

**Rules**
- One commit per task on `main`, subject `[TASK-ID] Title`, trailer `Task: TASK-ID`. Later fixes reuse the same trailer, and `deliver` carries them along.
- Commits are authored as `kkaaman <judehcrabs12@gmail.com>` with the course AI footer (`AI-assisted: … reviewed and edited by Mohammad`). Review a PR before merging it, because that footer is your attestation.
- `.mo/` is private tooling. Task commits never touch it. **Never push `main` to `team`.**
- The tag `a1-final` marks the untouched A1 code that both histories start from.
- CARLA-dependent tasks (`scope = build+carla` in `tasks.csv`) are coded and unit-tested here. The CARLA run and evidence capture happen on your CARLA machine, using the script each task provides.

**On your machine (once, in `~/elec5308-a2`)**
```bash
git fetch mine
git checkout -B main mine/main     # local main now follows your private repo
git branch -u mine/main
python .mo/deliver.py setup        # checks remotes, sets identity
gh auth login                      # optional, enables automatic PR creation
```

**Day to day**
```bash
git pull                                   # get what Claude pushed
python .mo/deliver.py status --todo        # what's left (filter: --owner Alan)
python .mo/deliver.py deliver BL-F-02 --dry-run   # preview, including dependencies pulled in
python .mo/deliver.py deliver BL-F-02      # stage changes on a fresh branch; you commit, then `pr`
python .mo/deliver.py deliver BL-F-02 --commit    # commit as you (no co-author lines), push, open PR
python .mo/deliver.py deliver BL-F-02 --commit --journal journal.json   # include your AI Journal export
python .mo/deliver.py sync                 # teammates' commits to merge into your main
```

- **First delivery.** If team `main` is empty, add `--seed-team`, which pushes the A1 baseline (`a1-final`) as team `main`. If it has only a README, the A1 history is merged under it automatically.
- **What reaches the team repo.** New commits made on your machine as `kkaaman`, one per task, with `Task:` and `AI-assisted:` lines and no co-author lines. `--keep-commits` copies the private commits instead, co-author lines included.
- **AI Journal in VS Code.** Run `deliver <ID>` without `--commit`, open `../elec5308-a2-deliver` in VS Code, Ctrl+Shift+P → *Commit with AI Journal*, start the message with `[<ID>]`, then `python .mo/deliver.py pr deliver/<ID>` from the main folder. The tool recognises a task by `[ID]` at the start of the commit title or a `Task: ID` line anywhere in the message.
- **AI Journal from a file.** `--journal FILE` puts your exported journal (JSON, as in your A1 commits) into the requested task's commit message, above the `Task:`/`AI-assisted:` lines so git still reads them. Dependencies pulled in don't get it.
- **Conflict.** Delivery branches are built in a sibling folder `../elec5308-a2-deliver`. Resolve there, `git add`, commit, then `python .mo/deliver.py pr deliver/<ID>`.
- **A teammate already delivered their own version.** Use `--skip <ID>` to leave yours out, then `sync` and merge theirs into your `main`.

**GitHub issues from the task board (one-off, no sync)**
```bash
python -m pip install openpyxl                                  # once, to read .xlsx
python .mo/import_issues.py BOARD.xlsx --dry-run                # offline; check .mo/issues_preview.md
python .mo/import_issues.py BOARD.xlsx --limit 3                # first 3 only; look at them on GitHub
python .mo/import_issues.py BOARD.xlsx --done RV-01             # the rest; RV-01 closed as completed
```
- One issue per Task ID, titled `[ID] Title`, with the board's labels, a G0–G5 milestone, the owner as assignee and the template's sections. Dependencies are issue links, created dependencies-first so every link resolves.
- Reruns skip any Task ID that already has an issue, open or closed. Pull requests are ignored, and existing issues are never edited.
- Issue numbers go to `.mo/issue_numbers.csv`, and `deliver` then adds `Closes #N` to each PR.
- The token comes from `GH_TOKEN`, `gh auth token`, or the one Git stored when you pushed, so issues are created as you. No `@` mentions in bodies, but every watcher still gets an email per new issue.
