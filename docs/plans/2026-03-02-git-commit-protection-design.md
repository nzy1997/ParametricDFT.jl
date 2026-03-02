# Git Commit Protection Design

**Date:** 2026-03-02
**Problem:** Git commits fail intermittently because 22 of 256 `.git/objects/` subdirectories are root-owned (from container build). When a new object's SHA hashes into a root-owned directory, `git add`/`git commit` fails with "insufficient permission for adding an object to repository database". This affects ~20-30% of commits probabilistically. Previous workarounds (`GIT_OBJECT_DIRECTORY`, alternates) created orphaned objects and missing tree references.

## Approach: Repack + CLAUDE.md guardrails

### Part 1: Fix root cause via repack

1. **`git repack -a -d`** — Consolidates all loose objects (including those in root-owned directories) into a single pack file under `.git/objects/pack/`, owned by claude-user. After this, git reads from the pack file and never needs the root-owned loose object directories.

2. **Remove workarounds** — Delete `.git/new_objects/` directory and clear `.git/objects/info/alternates` reference to it.

3. **Verify** — `git fsck --no-dangling` must report zero errors. `git log`, `git diff`, `git add`, `git commit` must work without `GIT_OBJECT_DIRECTORY` env var.

### Part 2: CLAUDE.md commit guardrails

Add three rules to the existing `## Git Safety` section:

1. **Never use `GIT_OBJECT_DIRECTORY` workaround** — If `git add`/`git commit` fails with permission errors on `.git/objects`, fix permissions via `git repack -a -d` rather than env var workarounds that create orphaned objects.

2. **Verify after commit** — Run `git fsck --connectivity-only` after each commit to catch corruption immediately.

3. **Never manipulate `.git` internals directly** — Don't copy/move individual objects between directories. Use `git repack` to fix object storage issues.

## Why not other approaches

- **Fresh clone:** Loses 9 unpushed local commits; requires push first
- **Rebuild .git:** More complex, same push requirement
- **Repack is simplest:** Non-destructive, fast, preserves all history and unpushed commits
