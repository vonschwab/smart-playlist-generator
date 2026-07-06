"""PreToolUse hook: keep simultaneous sessions from clobbering each other in one tree.

This repo runs multiple Claude sessions in a single SHARED working checkout
(worktrees were retired — they broke on data/ symlinks + SQLite WAL aliasing).
One working tree means one shared index and one HEAD, so the classic failure is
an agent staging or committing another session's in-flight work. This guard
denies the index-sweeping and tree-destroying git ops and points at the safe,
explicit-path form the repo already uses (`git add <paths>`, `git commit --only
-- <paths>`). See CLAUDE.md "Session discipline" + memory feedback_shared_checkout_commit_only.

DENY (safe alternative always exists):
  - git add -A / --all / -u / --update / . / * / :/     -> stage explicit paths
  - git commit -a / --all / -am                          -> git commit --only -- <paths>
  - bare git commit (no `--`/`--only` pathspec)          -> git commit --only -- <paths>
    (allowed while a merge/rebase/cherry-pick/revert is in progress, or --amend/
     --squash/--fixup/--no-edit — those are legitimate conclusion commits)
  - git reset --hard / git clean -f / git checkout . / git restore .

WARN (allowed, but flagged): git switch / git checkout -b|-B — changes the
shared HEAD for every session; make sure no one else is mid-edit.

Contract mirrors the repo's other PreToolUse hooks: read the tool-call JSON on
stdin; emit permissionDecision "deny" to block, additionalContext to warn, stay
silent to allow. FAIL OPEN — any error returns without blocking.
"""

import json
import os
import re
import shlex
import sys

PROJECT_DIR = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()

_SAFE_ADD = "Stage explicit paths instead: `git add <path1> <path2>` (never -A/-u/.)."
_SAFE_COMMIT = (
    "On a shared checkout a bare/`-a` commit sweeps other sessions' staged work. "
    "Commit explicit paths: `git commit --only -- <path1> <path2>` "
    "(verify first with `git diff --cached --name-only`)."
)

_BROAD_ADD = {"-A", "--all", "-u", "--update", ".", "*", ":/", "-Au", "-uA", "-uAll"}
_MERGE_MARKERS = ("MERGE_HEAD", "rebase-merge", "rebase-apply", "CHERRY_PICK_HEAD", "REVERT_HEAD")
_COMMIT_CONCLUSION_FLAGS = {"--amend", "--squash", "--fixup", "--no-edit"}


def _merge_or_rebase_in_progress():
    """True if a git operation that legitimately ends in a bare `git commit` is live."""
    gitdir = os.path.join(PROJECT_DIR, ".git")
    return any(os.path.exists(os.path.join(gitdir, m)) for m in _MERGE_MARKERS)


def _segments(command):
    # Evaluate each command in a compound (git add -A && git commit ...) on its own.
    return re.split(r"&&|\|\||[;&|\n]", command or "")


def _tokens(segment):
    try:
        toks = shlex.split(segment, posix=False)  # posix=False keeps Windows backslashes
    except ValueError:
        toks = segment.split()
    return [t.strip("\"'") for t in toks]


def _git_args(tokens):
    """Return the args after `git`, skipping global options (-C <path>, -c k=v, ...)."""
    if "git" not in tokens:
        return None
    args = tokens[tokens.index("git") + 1:]
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in ("-C", "-c") else 1
    return args[i:]  # [subcommand, *subargs] or []


def _has_a_cluster(flag):
    # single-dash short-flag cluster containing 'a' (-a, -am, -ma) but not --amend
    return bool(re.fullmatch(r"-[a-z]*a[a-z]*", flag))


def analyze_segment(tokens):
    """Return ('deny', msg) | ('warn', msg) | None for one command segment."""
    args = _git_args(tokens)
    if not args:
        return None
    sub, sargs = args[0], args[1:]

    if sub == "add":
        if any(a in _BROAD_ADD for a in sargs):
            return ("deny", f"`git add` with a broad selector stages every session's changes. {_SAFE_ADD}")
        return None

    if sub == "commit":
        if any(a == "--all" or _has_a_cluster(a) for a in sargs):
            return ("deny", f"`git commit -a/--all` stages and commits all tracked changes. {_SAFE_COMMIT}")
        has_pathspec = "--" in sargs or "--only" in sargs or "--include" in sargs or "-i" in sargs
        conclusion = any(a in _COMMIT_CONCLUSION_FLAGS for a in sargs)
        if not has_pathspec and not conclusion and not _merge_or_rebase_in_progress():
            return ("deny", f"Bare `git commit` commits the shared index. {_SAFE_COMMIT}")
        return None

    if sub == "reset":
        if "--hard" in sargs:
            return ("deny", "`git reset --hard` discards every session's uncommitted work in the shared tree. Use `git reset <paths>` (unstage) or `git restore -- <paths>` for your own files only.")
        return None

    if sub == "clean":
        if any(a == "--force" or re.fullmatch(r"-[a-z]*f[a-z]*", a) for a in sargs):
            return ("deny", "`git clean -f` deletes untracked files, including other sessions' new files. Delete specific paths explicitly instead.")
        return None

    if sub in ("checkout", "restore"):
        if "." in sargs:
            return ("deny", f"`git {sub} .` discards all working-tree changes, including other sessions'. Restore specific files: `git {sub} -- <paths>`.")
        if sub == "checkout" and any(a in ("-b", "-B") for a in sargs):
            return ("warn", "Hook note: creating/switching a branch changes the shared HEAD for every session in this checkout — confirm no other session is mid-edit first.")
        return None

    if sub == "switch":
        return ("warn", "Hook note: `git switch` changes the shared HEAD for every session in this checkout — confirm no other session is mid-edit first.")

    return None


def analyze(command):
    """First deny wins; else first warn; else None."""
    warn = None
    for seg in _segments(command):
        result = analyze_segment(_tokens(seg))
        if result is None:
            continue
        if result[0] == "deny":
            return result
        warn = warn or result
    return warn


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    try:
        if (data.get("tool_name") or "") not in ("Bash", "PowerShell"):
            return
        command = (data.get("tool_input") or {}).get("command") or ""
        result = analyze(command)
        if result is None:
            return
        kind, message = result
        if kind == "deny":
            out = {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": message,
            }
        else:
            out = {"hookEventName": "PreToolUse", "additionalContext": message}
        print(json.dumps({"hookSpecificOutput": out}))
    except Exception:
        return  # fail-open: never brick a session on a guard bug


if __name__ == "__main__":
    main()
