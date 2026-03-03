# Issue-PR Skill Design

**Date:** 2026-03-03
**Status:** Approved

## Problem

Commits on the repository are not consistently tied to GitHub issues and PRs. This makes it hard to trace why changes were made and breaks the issue-to-PR audit trail.

## Solution

A fully automatic post-change skill that creates a GitHub issue, commits changes, and opens a PR — all without human input. Claude infers all metadata (title, description, type, labels) from the code diff.

## Trigger

After every code modification. The skill runs automatically once Claude has finished writing/editing code.

## Workflow

1. **Read the diff** — run `git diff` and `git status` to understand what changed.
2. **Infer metadata** — from the diff context, derive:
   - **Type**: feat/fix/refactor/docs/chore (mapped to GitHub labels)
   - **Title**: concise summary of the change
   - **Description**: what changed and why
   - **Branch name**: `{type}/{short-description}`
3. **Create GitHub issue** — `gh issue create` with auto-generated body. Capture issue number.
4. **Handle branching**:
   - If on `main`: create a new feature branch from HEAD.
   - If already on a feature branch: stay on it.
5. **Commit changes** — stage and commit with a message referencing the issue.
6. **Create PR** — `gh pr create` with body containing `Closes #{issue_number}`.
7. **Print summary** — issue URL, PR URL, branch name.

## Issue Body Template

```markdown
## Description
{auto-generated from diff context}

## Changes
{list of modified files and summary of changes}

## Branch
`{type}/{short-description}`
```

## PR Body Template

```markdown
## Summary
{auto-generated from diff and commit messages}

Closes #{issue_number}
```

## Label Mapping

| Inferred type | GitHub label    | Branch prefix  |
|---------------|-----------------|----------------|
| Feature       | `enhancement`   | `feat/`        |
| Bug fix       | `bug`           | `fix/`         |
| Refactor      | `refactor`      | `refactor/`    |
| Docs          | `documentation` | `docs/`        |
| Chore         | `chore`         | `chore/`       |

## Branch Naming

- Format: `{type}/{short-description}`
- Lowercase, hyphens for spaces, max ~50 chars
- Example: `feat/add-riemannian-adam`
- If on a feature branch already, reuse it — no new branch created.

## Edge Cases

- **Already on feature branch:** Use it. Don't create a new one.
- **On `main`:** Create a new feature branch before committing.
- **No changes:** Skip — nothing to do.
- **Branch name collision:** Append numeric suffix (e.g., `feat/fix-training-2`).
- **`gh` not available:** Error with clear message to install GitHub CLI.

## Scope

Fully automatic. No user prompts during execution. Claude derives everything from the diff.
