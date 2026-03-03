# Issue-PR Skill Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fully automatic post-change skill that creates a GitHub issue, commits changes, and opens a PR without human input.

**Architecture:** Single SKILL.md file at `.claude/skills/issue-pr/SKILL.md` containing step-by-step instructions for Claude to execute. The skill reads the diff, infers metadata, and runs `gh` CLI commands. No external dependencies beyond `gh`.

**Tech Stack:** GitHub CLI (`gh`), git, shell commands

---

### Task 1: Create missing GitHub labels

The repo only has `bug`, `enhancement`, `documentation`. The skill needs `refactor` and `chore` labels too.

**Files:**
- None (GitHub API only)

**Step 1: Create the `refactor` label**

Run:
```bash
gh label create refactor --description "Code refactoring without behavior change" --color "e6e6e6"
```
Expected: Label created successfully.

**Step 2: Create the `chore` label**

Run:
```bash
gh label create chore --description "Maintenance tasks, CI, tooling" --color "c5def5"
```
Expected: Label created successfully.

**Step 3: Verify labels exist**

Run:
```bash
gh label list | grep -E "refactor|chore"
```
Expected: Both labels appear in output.

**Step 4: Commit**

No files changed — skip commit.

---

### Task 2: Write the issue-pr SKILL.md

This is the core task. The skill must be fully self-contained instructions that Claude follows automatically.

**Files:**
- Modify: `.claude/skills/issue-pr/SKILL.md`

**Step 1: Write the skill file**

Write the following content to `.claude/skills/issue-pr/SKILL.md`:

```markdown
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
   - `feat` — new functionality (maps to GitHub label `enhancement`)
   - `fix` — bug fix (maps to GitHub label `bug`)
   - `refactor` — restructuring without behavior change (maps to GitHub label `refactor`)
   - `docs` — documentation only (maps to GitHub label `documentation`)
   - `chore` — maintenance, CI, tooling (maps to GitHub label `chore`)

2. **Short description** — 3-6 words describing the change, lowercase, hyphens for spaces. Max 50 chars. Example: `add-riemannian-adam`

3. **Issue title** — conventional commit format: `{Type}: {Description}`. Capitalize the type. Example: `Feat: add Riemannian Adam optimizer`

4. **Issue body** — describe what changed and why in 2-4 sentences.

5. **List of changed files** — from `git diff --stat`.

## Step 2: Handle branching

Check the current branch:

```bash
git branch --show-current
```

- **If on `main` or `master`:** Create a new branch and switch to it:
  ```bash
  git checkout -b {type}/{short-description}
  ```
  If the branch name already exists, append `-2`, `-3`, etc.

- **If on any other branch:** Stay on it. Use the existing branch name.

## Step 3: Create the GitHub issue

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

Capture the issue number from the output URL (e.g., `https://github.com/.../issues/46` → `46`).

## Step 4: Commit changes

Stage all modified and new files, then commit with a message referencing the issue:

```bash
git add {specific files from the diff}
git commit -m "$(cat <<'COMMIT_EOF'
{type}: {short description}

{1-2 sentence summary of what changed}

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

Push the branch and create a PR:

```bash
git push -u origin $(git branch --show-current)
```

Then create the PR. If a PR already exists for this branch, skip PR creation.

```bash
gh pr list --head "$(git branch --show-current)" --state open --json number | grep -q "number"
```

If no open PR exists:

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

If an open PR already exists, just push — the new commit will appear on the existing PR.

## Step 6: Print summary

Output the result:

```
## Issue + PR Created

- **Issue:** {issue_url}
- **PR:** {pr_url} (or "pushed to existing PR")
- **Branch:** {branch_name}
- **Commit:** {commit_sha_short}
```

## Label Mapping Reference

| Type     | GitHub Label    | Branch Prefix |
|----------|-----------------|---------------|
| feat     | enhancement     | feat/         |
| fix      | bug             | fix/          |
| refactor | refactor        | refactor/     |
| docs     | documentation   | docs/         |
| chore    | chore           | chore/        |
```

**Step 2: Verify the skill file is valid**

Read the file back and confirm:
- Frontmatter has `name` and `description`
- All 6 steps are present
- No placeholder text remains (all `{...}` are instruction templates, not literals)

**Step 3: Commit**

```bash
git add .claude/skills/issue-pr/SKILL.md
git commit -m "feat: add issue-pr skill for automatic issue and PR creation

Fully automatic post-change workflow that creates GitHub issues,
commits changes on feature branches, and opens PRs. Claude infers
all metadata from the code diff with no human input.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Update CLAUDE.md skill reference

The CLAUDE.md currently references the issue-pr skill but with no description. Update it.

**Files:**
- Modify: `.claude/CLAUDE.md`

**Step 1: Update the skill entry**

Find this line in `.claude/CLAUDE.md`:
```
- [issue-pr](skills/issue-pr/SKILL.md) — Create an issue and a pull request from the current branch.
```

Replace with:
```
- [issue-pr](skills/issue-pr/SKILL.md) — Auto-create GitHub issue, commit changes, and open PR after code modifications. Fully automatic.
```

**Step 2: Commit**

```bash
git add .claude/CLAUDE.md
git commit -m "docs: update issue-pr skill description in CLAUDE.md

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Test the skill end-to-end

Verify the skill works by simulating a small code change.

**Step 1: Make a trivial test change**

Add a comment to any file (e.g., a TODO in `src/ParametricDFT.jl`).

**Step 2: Invoke the skill**

Run `/issue-pr` and verify:
- Issue is created on GitHub with correct title, label, and body
- Commit is made with issue reference
- PR is created (or existing PR is updated) with `Closes #N`
- Summary is printed

**Step 3: Clean up**

Close the test issue and PR. Revert the trivial change.

```bash
gh issue close {test_issue_number}
gh pr close {test_pr_number}
git revert HEAD
```

**Step 4: Commit cleanup**

```bash
git add -A
git commit -m "chore: revert test change from skill validation

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
