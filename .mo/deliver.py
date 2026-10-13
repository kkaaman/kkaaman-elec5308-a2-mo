#!/usr/bin/env python3
"""Parallel-delivery helper for ELEC5308 A2 (Mo's private repo).

Model
-----
* private remote = kkaaman/kkaaman-elec5308-a2-mo (found by URL; any name). Its ``main`` holds the full
  implementation. Every task lands as commit(s) carrying a ``Task: <ID>``
  trailer; follow-up fixes reuse the same trailer.
* team remote    = USYD-ELEC5308/26-S2-Assignment2-Group7 (found by URL). Nothing is pushed to
  team ``main`` directly except the one-off A1 baseline when it is empty.
* ``deliver`` builds ``deliver/<ID>`` from team/main, cherry-picks the task's
  commits (plus any undelivered implemented dependencies), pushes it to
  ``team`` and opens a PR under your name.
* ``.mo/`` is private tooling. Task commits never touch it, so it is never
  delivered. Never push private ``main`` to ``team``.

Commands (run from anywhere inside the repo)
--------------------------------------------
  python .mo/deliver.py setup                   # once per clone
  python .mo/deliver.py status [--owner Alan] [--todo]
  python .mo/deliver.py deliver BL-F-01 [BL-F-02 ...] [--no-deps] [--dry-run]
        default: stage the changes on a fresh branch for you to commit yourself
        --commit: commit each task as you (no co-author lines), push, open PR
  python .mo/deliver.py pr deliver/BL-F-01      # push + PR after resolving a conflict
  python .mo/deliver.py sync                    # show teammates' commits to merge in

Only the standard library and git are required; ``gh`` is optional (without
it the compare URL is printed instead of opening the PR automatically).
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TASKS_CSV = HERE / "tasks.csv"

PRIVATE_SLUG = "kkaaman/kkaaman-elec5308-a2-mo"
PRIVATE_REMOTE = "origin"  # resolved by URL at startup (see _resolve_remotes)
TEAM_REMOTE = "team"       # resolved by URL at startup
TEAM_SLUG = "USYD-ELEC5308/26-S2-Assignment2-Group7"
TEAM_URL = f"https://github.com/{TEAM_SLUG}.git"
AUTHOR_NAME = "kkaaman"
AUTHOR_EMAIL = "judehcrabs12@gmail.com"
BASE_TAG = "a1-final"
PRIVATE_MAIN = f"{PRIVATE_REMOTE}/main"
WORKTREE = ROOT.parent / f"{ROOT.name}-deliver"  # delivery branches are built here
TEAM_MAIN = f"{TEAM_REMOTE}/main"


# --------------------------------------------------------------------------- git
def git(*args: str, check: bool = True, capture: bool = True, cwd: Path = ROOT,
        input_text: str | None = None) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8",
        capture_output=capture, input=input_text,
    )
    if check and proc.returncode != 0:
        err = proc.stderr if capture else ""
        sys.exit(f"git {' '.join(args)} failed ({proc.returncode})\n{err}".rstrip())
    return proc.stdout.strip() if capture else ""


def ref_exists(ref: str) -> bool:
    return subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=ROOT, capture_output=True,
    ).returncode == 0


def git_version() -> tuple[int, ...]:
    m = re.search(r"(\d+)\.(\d+)", git("--version"))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def fetch_all() -> None:
    git("fetch", "--quiet", "--tags", PRIVATE_REMOTE)
    if TEAM_REMOTE in git("remote").split():
        git("fetch", "--quiet", TEAM_REMOTE)


def _resolve_remotes() -> None:
    """Find the private and team remotes by URL, whatever they are called."""
    global PRIVATE_REMOTE, TEAM_REMOTE, PRIVATE_MAIN, TEAM_MAIN
    for name in git("remote").split():
        url = git("remote", "get-url", name).lower()
        if PRIVATE_SLUG.lower() in url:
            PRIVATE_REMOTE = name
        elif TEAM_SLUG.lower() in url:
            TEAM_REMOTE = name
    PRIVATE_MAIN = f"{PRIVATE_REMOTE}/main"
    TEAM_MAIN = f"{TEAM_REMOTE}/main"


# ------------------------------------------------------------------------- tasks
def load_tasks() -> dict[str, dict]:
    with TASKS_CSV.open(encoding="utf-8", newline="") as fh:
        tasks = {row["id"]: row for row in csv.DictReader(fh)}
    for row in tasks.values():
        row["deps"] = [d.strip() for d in row["depends_on"].split(",") if d.strip()]
    return tasks


def task_commits(ref: str) -> dict[str, list[str]]:
    """Map Task ID -> commit hashes (oldest first) on ``ref``."""
    if not ref_exists(ref):
        return {}
    out = git("log", ref, "--reverse",
              "--format=%H%x1f%(trailers:key=Task,valueonly,separator=%x2C)%x1e")
    result: dict[str, list[str]] = {}
    for rec in out.split("\x1e"):
        rec = rec.strip()
        if not rec:
            continue
        sha, _, ids = rec.partition("\x1f")
        for tid in (t.strip() for t in ids.split(",")):
            if tid:
                result.setdefault(tid, []).append(sha)
    return result


def closure(requested: list[str], tasks: dict, mine: dict, delivered: set,
            with_deps: bool) -> list[str]:
    """Implemented, undelivered tasks needed for ``requested``."""
    want: set[str] = set()
    seen: set[str] = set()
    stack = list(requested)
    while stack:
        tid = stack.pop()
        if tid in seen:
            continue
        seen.add(tid)
        if tid in delivered and tid not in requested:
            continue
        if tid in mine and tid not in delivered:
            want.add(tid)
        if with_deps:
            stack.extend(tasks.get(tid, {}).get("deps", []))
    return sorted(want)


# ---------------------------------------------------------------------- commands
def cmd_setup(_: argparse.Namespace) -> None:
    remotes = [git("remote", "get-url", r).lower() for r in git("remote").split()]
    if not any(TEAM_SLUG.lower() in u for u in remotes):
        git("remote", "add", TEAM_REMOTE, TEAM_URL)
        print(f"added remote '{TEAM_REMOTE}' -> {TEAM_URL}")
    if not any(PRIVATE_SLUG.lower() in u for u in remotes):
        sys.exit(f"no remote points at {PRIVATE_SLUG}; add one, e.g. git remote add mine https://github.com/{PRIVATE_SLUG}.git")
    _resolve_remotes()
    git("config", "user.name", AUTHOR_NAME)
    git("config", "user.email", AUTHOR_EMAIL)
    git("config", f"remote.{TEAM_REMOTE}.tagOpt", "--no-tags")
    fetch_all()
    print(f"identity: {AUTHOR_NAME} <{AUTHOR_EMAIL}>")
    print(f"team main: {'present' if ref_exists(TEAM_MAIN) else 'EMPTY (deliver will offer to seed it)'}")


def cmd_status(args: argparse.Namespace) -> None:
    fetch_all()
    tasks = load_tasks()
    mine = task_commits(PRIVATE_MAIN)
    team = task_commits(TEAM_MAIN)
    rows = []
    for tid, t in tasks.items():
        if args.owner and t["owner"].lower() != args.owner.lower():
            continue
        if t["scope"].startswith("skip") and not args.all:
            continue
        done_mine, done_team = tid in mine, tid in team
        if args.todo and done_mine:
            continue
        rows.append((tid, t["owner"], t["due"], "Y" if done_mine else "-",
                     "Y" if done_team else "-", t["scope"], t["title"][:60]))
    print(f"{'ID':<14}{'Owner':<10}{'Due':<12}{'Mine':<6}{'Team':<6}{'Scope':<13}Title")
    for r in rows:
        print(f"{r[0]:<14}{r[1]:<10}{r[2]:<12}{r[3]:<6}{r[4]:<6}{r[5]:<13}{r[6]}")
    buildable = [t for t in tasks if tasks[t]["scope"].startswith("build")]
    print(f"\nbuildable tasks implemented: {sum(t in mine for t in buildable)}/{len(buildable)}"
          f" | delivered to team: {sum(t in team for t in buildable)}")


def _seed_empty_team() -> None:
    if not ref_exists(BASE_TAG):
        sys.exit(f"tag {BASE_TAG} missing; cannot seed team main")
    print(f"team repo has no main. Seeding it with the A1 baseline ({BASE_TAG}) ...")
    git("push", TEAM_REMOTE, f"{BASE_TAG}^{{commit}}:refs/heads/main", capture=False)
    git("fetch", "--quiet", TEAM_REMOTE)


def cmd_deliver(args: argparse.Namespace) -> None:
    fetch_all()
    tasks = load_tasks()
    unknown = [t for t in args.ids if t not in tasks]
    if unknown:
        sys.exit(f"unknown task id(s): {', '.join(unknown)}")
    mine = task_commits(PRIVATE_MAIN)
    missing = [t for t in args.ids if t not in mine]
    if missing:
        sys.exit(f"not implemented on {PRIVATE_MAIN} yet: {', '.join(missing)}")
    delivered = set(task_commits(TEAM_MAIN)) - set(args.redeliver)
    skip = set(args.skip)
    ids = [t for t in closure(args.ids, tasks, mine, delivered, not args.no_deps) if t not in skip]
    if not ids:
        sys.exit("nothing to deliver: everything requested is already on team main")

    order = git("rev-list", "--reverse", PRIVATE_MAIN).split()
    pos = {sha: i for i, sha in enumerate(order)}
    commits = sorted({c for t in ids for c in mine[t]}, key=pos.__getitem__)
    branch = args.branch or "deliver/" + "-".join(args.ids)[:60]

    print(f"branch : {branch}")
    print(f"tasks  : {', '.join(ids)}")
    extra = sorted(set(ids) - set(args.ids))
    if extra:
        print(f"  (dependencies pulled in: {', '.join(extra)}; use --no-deps or --skip to drop)")
    print(f"commits: {len(commits)}")
    for c in commits:
        print("   ", git("log", "-1", "--format=%h %s", c))
    if args.dry_run:
        return

    if not ref_exists(TEAM_MAIN):
        if not args.seed_team:
            sys.exit("team repo is empty. Re-run with --seed-team to push the A1 baseline as team main first.")
        _seed_empty_team()

    shares_history = subprocess.run(
        ["git", "merge-base", TEAM_MAIN, PRIVATE_MAIN], cwd=ROOT, capture_output=True,
    ).returncode == 0
    _fresh_worktree(branch, TEAM_MAIN if shares_history else BASE_TAG)
    if not shares_history:
        # Team main is an unrelated README-only history: graft A1 baseline under it.
        print(f"team main has unrelated history; basing on {BASE_TAG} + merge of team main")
        git("merge", "--allow-unrelated-histories", "-X", "ours", "--no-edit",
            "-m", "Merge initial team repository commit into A1 baseline", TEAM_MAIN, cwd=WORKTREE)

    if args.keep_commits:
        empty_flag = ["--empty=drop"] if git_version() >= (2, 45) else ["--keep-redundant-commits"]
        proc = subprocess.run(["git", "cherry-pick", *empty_flag, *commits], cwd=WORKTREE)
        if proc.returncode != 0:
            print(f"\nCherry-pick stopped (conflict or empty commit) in {WORKTREE}\n"
                  f"  cd \"{WORKTREE}\"  -> resolve files, git add, git cherry-pick --continue (repeat)\n"
                  f"  then, back in your main checkout: python .mo/deliver.py pr {branch}")
            sys.exit(1)
        _push_and_pr(branch, ids, tasks)
        return

    # Default: apply the changes only. Commits are made on this machine, as you.
    groups = _task_groups(commits)
    for tid, group in groups:
        proc = subprocess.run(["git", "cherry-pick", "--no-commit", *group], cwd=WORKTREE)
        if proc.returncode != 0:
            print(f"\nConflict while applying {tid} in {WORKTREE}. Resolve the files, `git add` them,\n"
                  "then commit there yourself with the message below and re-run deliver for any\n"
                  "remaining tasks:\n\n" + _commit_message(tid, tasks, group))
            sys.exit(1)
        if not args.commit:
            continue
        if not git("diff", "--cached", "--name-only", cwd=WORKTREE):
            print(f"  {tid}: no changes left to apply (already on team main), skipped")
            continue
        git("commit", "--quiet", "--no-verify", "-F", "-", cwd=WORKTREE,
            input_text=_commit_message(tid, tasks, group))
        print(f"  committed {tid}: {git('log', '-1', '--format=%h %an <%ae>', cwd=WORKTREE)}")

    if args.commit:
        _push_and_pr(branch, ids, tasks)
        return
    print(f"\nChanges for {', '.join(ids)} are staged (not committed) in:\n  {WORKTREE}\n"
          "Review them there, commit as yourself, then push and open the PR:\n"
          f"  cd \"{WORKTREE}\"\n  git status\n  git commit\n"
          f"  cd \"{ROOT}\"\n  python .mo/deliver.py pr {branch}\n"
          "\nSuggested commit message:\n\n" + _commit_message(ids[0], tasks, commits, all_ids=ids))


def _task_groups(commits: list[str]) -> list[tuple[str, list[str]]]:
    """Split commits (in order) into consecutive runs belonging to one task."""
    groups: list[tuple[str, list[str]]] = []
    for c in commits:
        tid = git("log", "-1", "--format=%(trailers:key=Task,valueonly,separator=%x2C)", c).split(",")[0].strip()
        if groups and groups[-1][0] == tid:
            groups[-1][1].append(c)
        else:
            groups.append((tid, [c]))
    return groups


def _commit_message(tid: str, tasks: dict, commits: list[str], all_ids: list[str] | None = None) -> str:
    """Commit message for the team repo: task title, Task trailer(s), AI disclosure; no co-author lines."""
    ids = all_ids or [tid]
    subject = f"[{tid}] {tasks[tid]['title']}" if len(ids) == 1 else f"[{', '.join(ids)}] " + tasks[ids[0]]["title"]
    ai = ""
    for c in commits:
        found = git("log", "-1", "--format=%(trailers:key=AI-assisted,valueonly)", c).strip()
        if found:
            ai = found.splitlines()[0]
            break
    lines = [subject, ""] + [f"Task: {t}" for t in ids]
    if ai:
        lines.append(f"AI-assisted: {ai}")
    return "\n".join(lines) + "\n"


def _fresh_worktree(branch: str, base: str) -> None:
    """(Re)create the delivery worktree on ``branch`` starting at ``base``."""
    if WORKTREE.exists():
        busy = git("rev-parse", "--git-path", "CHERRY_PICK_HEAD", cwd=WORKTREE)
        if (WORKTREE / busy).exists() or Path(busy).exists():
            sys.exit(f"a cherry-pick is still in progress in {WORKTREE}; finish or abort it first")
        git("worktree", "remove", "--force", str(WORKTREE))
    git("worktree", "prune")
    git("worktree", "add", "--quiet", "-B", branch, str(WORKTREE), base)


def _pr_body(ids: list[str], tasks: dict, branch: str) -> str:
    lines = ["## Tasks", ""]
    for tid in ids:
        t = tasks[tid]
        lines.append(f"- **{tid}** {t['title']} (planned owner: {t['owner']}, due {t['due']})")
        if t["acceptance"]:
            lines.append(f"  - Acceptance: {t['acceptance']}")
        if t["files"]:
            lines.append(f"  - Files: `{t['files']}`")
    lines += ["", "## Verification", "", "- [ ] `ruff check .`", "- [ ] `python -m pytest`"]
    if any(tasks[t]["carla"] == "Y" for t in ids):
        lines.append("- [ ] CARLA evidence linked (run logs / summary JSON)")
    disclosures = sorted(set(re.findall(
        r"^AI-assisted:\s*(.+)$",
        git("log", "--format=%B", f"{TEAM_MAIN}..{branch}") if ref_exists(TEAM_MAIN) else "",
        flags=re.M)))
    lines += ["", "## AI disclosure", ""]
    lines += [f"AI-assisted: {d}" for d in disclosures] or [
        "AI-assisted: <tool> – <what>; reviewed and edited by Mohammad"]
    return "\n".join(lines) + "\n"


def _push_and_pr(branch: str, ids: list[str], tasks: dict) -> None:
    git("push", "--force-with-lease", "-u", TEAM_REMOTE, f"{branch}:{branch}", capture=False)
    title = f"[{', '.join(ids)}] " + (tasks[ids[0]]["title"] if len(ids) == 1 else "multiple tasks")
    body = _pr_body(ids, tasks, branch)
    if shutil.which("gh"):
        proc = subprocess.run(
            ["gh", "pr", "create", "--repo", TEAM_SLUG, "--base", "main", "--head", branch,
             "--title", title, "--body-file", "-"],
            cwd=ROOT, input=body, text=True, encoding="utf-8",
        )
        if proc.returncode == 0:
            return
        print("gh pr create failed (PR may already exist); compare URL below.")
    print(f"\nOpen the PR: https://github.com/{TEAM_SLUG}/compare/main...{branch}?expand=1")
    print("\n--- PR body ---\n" + body)


def cmd_pr(args: argparse.Namespace) -> None:
    tasks = load_tasks()
    base = TEAM_MAIN if ref_exists(TEAM_MAIN) else BASE_TAG
    ids = sorted(set(task_commits(args.branch)) - set(task_commits(base)))
    if not ids:
        sys.exit(f"no task commits found on {args.branch} beyond {base}")
    _push_and_pr(args.branch, ids, tasks)


def cmd_sync(_: argparse.Namespace) -> None:
    fetch_all()
    if not ref_exists(TEAM_MAIN):
        print("team main is empty; nothing to sync")
        return
    out = git("log", f"{PRIVATE_MAIN}..{TEAM_MAIN}", "--no-merges",
              "--format=%h%x1f%an%x1f%(trailers:key=Task,valueonly,separator=%x2C)%x1f%s")
    theirs = [line.split("\x1f") for line in out.splitlines() if line]
    theirs = [r for r in theirs if not r[2]]  # commits without our Task trailer
    if not theirs:
        print("team main has nothing new from teammates")
        return
    print("Teammate commits on team main not yet in your main:")
    for sha, author, _, subj in theirs:
        print(f"  {sha} {author:<16} {subj}")
    print(f"\nTo absorb them:  git switch main && git merge {TEAM_MAIN} && git push {PRIVATE_REMOTE} main")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    s = sub.add_parser("status")
    s.add_argument("--owner")
    s.add_argument("--todo", action="store_true", help="only tasks not yet implemented")
    s.add_argument("--all", action="store_true", help="include other members' private/team-only tasks")
    s.set_defaults(fn=cmd_status)
    d = sub.add_parser("deliver")
    d.add_argument("ids", nargs="+")
    d.add_argument("--no-deps", action="store_true", help="do not pull in undelivered dependencies")
    d.add_argument("--skip", nargs="*", default=[], help="task ids to leave out (e.g. teammate delivered their own)")
    d.add_argument("--redeliver", nargs="*", default=[], help="treat these as not yet on team main")
    d.add_argument("--branch")
    d.add_argument("--seed-team", action="store_true", help="push A1 baseline as team main if it is empty")
    d.add_argument("--commit", action="store_true",
                   help="commit each task as you (local git identity) and push + open the PR")
    d.add_argument("--keep-commits", action="store_true",
                   help="copy the private commits as they are (includes their co-author lines)")
    d.add_argument("--dry-run", action="store_true")
    d.set_defaults(fn=cmd_deliver)
    r = sub.add_parser("pr")
    r.add_argument("branch")
    r.set_defaults(fn=cmd_pr)
    sub.add_parser("sync").set_defaults(fn=cmd_sync)
    args = p.parse_args()
    _resolve_remotes()
    args.fn(args)


if __name__ == "__main__":
    main()
