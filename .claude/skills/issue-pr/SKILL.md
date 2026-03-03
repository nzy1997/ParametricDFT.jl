---
name: issue-pr
description: Use after writing or modifying code to automatically create a GitHub issue, commit changes, and open a PR. Fully automatic — no human input needed.
---

# Auto Issue + PR

You are creating a GitHub issue, committing code changes, and opening a PR. Do this fully automatically with no user prompts.

## Pre-flight Checks

Run these commands and analyze the output:

```bash
git status
git diff --stat
git diff
```

**If there are no changes (clean working tree):** Stop. Say "No changes to commit." and exit.

**If `gh` is not available:** Stop. Say "GitHub CLI (gh) not installed. Install it: https://cli.github.com/" and exit.

## Step 1: Analyze the diff

From the diff output, determine:

1. **Change type** — pick exactly one:
   - `feat` — new functionality (GitHub label: `enhancement`)
   - `fix` — bug fix (GitHub label: `bug`)
   - `refactor` — restructuring without behavior change (GitHub label: `refactor`)
   - `docs` — documentation only (GitHub label: `documentation`)
   - `chore` — maintenance, CI, tooling (GitHub label: `chore`)

2. **Short description** — 3-6 words, lowercase, hyphens for spaces, max 50 chars. Example: `add-riemannian-adam`

3. **Issue title** — conventional commit format: `{Type}: {description}`. Capitalize the type. Example: `Feat: add Riemannian Adam optimizer`

4. **Issue body** — describe what changed and why in 2-4 sentences.

5. **List of changed files** — from `git diff --stat`.

## Step 2: Handle branching

Check the current branch:

```bash
git branch --show-current
```

- **If on `main` or `master`:** Create a new branch:
  ```bash
  git checkout -b {type}/{short-description}
  ```
  If the branch name already exists, append `-2`, `-3`, etc.

- **If on any other branch:** Stay on it. Use the existing branch.

## Step 3: Create the GitHub issue

Run:

```bash
gh issue create \
  --title "{issue_title}" \
  --label "{github_label}" \
  --body "$(cat <<'ISSUE_EOF'
## Description
{issue_body}

## Changes
{list of changed files with one-line summary each}
ISSUE_EOF
)"
```

Capture the issue number from the output URL (e.g., `https://github.com/.../issues/46` gives `46`).

## Step 4: Commit changes

Stage specific files and commit:

```bash
git add {specific files from the diff}
git commit -m "$(cat <<'COMMIT_EOF'
{type}: {short description}

{1-2 sentence summary}

Refs #{issue_number}

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
COMMIT_EOF
)"
```

**Important:** Stage specific files, not `git add -A`. Never commit `.env`, credentials, or large binaries.

After committing, verify integrity:

```bash
git fsck --connectivity-only
```

## Step 5: Push and create PR

Push the branch:

```bash
git push -u origin $(git branch --show-current)
```

Check if a PR already exists for this branch:

```bash
gh pr list --head "$(git branch --show-current)" --state open --json number
```

**If no open PR exists**, create one:

```bash
gh pr create \
  --title "{issue_title}" \
  --body "$(cat <<'PR_EOF'
## Summary
{2-3 bullet points describing the changes}

Closes #{issue_number}

🤖 Generated with [Claude Code](https://claude.com/claude-code)
PR_EOF
)"
```

**If an open PR already exists**, just push — the new commit appears on the existing PR.

## Step 6: Print summary

Output:

```
## Issue + PR Created

- **Issue:** {issue_url}
- **PR:** {pr_url} (or "pushed to existing PR")
- **Branch:** {branch_name}
- **Commit:** {short_sha}
```

## Label Mapping Reference

| Type     | GitHub Label    | Branch Prefix |
|----------|-----------------|---------------|
| feat     | enhancement     | feat/         |
| fix      | bug             | fix/          |
| refactor | refactor        | refactor/     |
| docs     | documentation   | docs/         |
| chore    | chore           | chore/        |
