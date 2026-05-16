---
title: Git Master Cheatsheet
sidebar_position: 14
---

# Git Master Cheatsheet

## Daily workflow

| Method | Description | Code example |
|---|---|---|
| `git status` | Shows changed, staged, untracked files, and branch state. | `git status --short` |
| `git add` | Stages files for the next commit. | `git add docs/cheetsheet/git-master-cheatsheet.md` |
| `git commit` | Records staged changes with a message. | `git commit -m "Add Git master cheatsheet"` |
| `git diff` | Shows unstaged changes. Use `--staged` for staged diff. | `git diff`<br/>`git diff --staged` |
| `git log` | Shows commit history. | `git log --oneline --graph --decorate --all` |
| `git show` | Shows one commit or object. | `git show HEAD` |

## Branching and integration

| Method | Description | Code example |
|---|---|---|
| `git branch` | Lists, creates, or deletes branches. | `git branch`<br/>`git branch feature/api` |
| `git switch` | Switches branches. Prefer over older `checkout` for branch moves. | `git switch main`<br/>`git switch -c feature/cheatsheets` |
| `git merge` | Combines another branch into the current branch with a merge commit or fast-forward. | `git switch main`<br/>`git merge feature/cheatsheets` |
| `git rebase` | Replays commits onto a new base for a linear history. | `git switch feature/cheatsheets`<br/>`git rebase main` |
| `git cherry-pick` | Applies a specific commit onto the current branch. | `git cherry-pick abc1234` |
| `git stash` | Temporarily shelves local changes. | `git stash push -m "wip docs"`<br/>`git stash pop` |

## Undo operations

| Method | Description | Code example |
|---|---|---|
| `git restore` | Restores files from index or commits. Use carefully. | `git restore path/to/file.py` |
| `git restore --staged` | Unstages files without changing working tree content. | `git restore --staged docs/file.md` |
| `git revert` | Creates a new commit that undoes another commit. Safe for shared branches. | `git revert abc1234` |
| `git reset --soft` | Moves branch pointer but keeps changes staged. | `git reset --soft HEAD~1` |
| `git reset --mixed` | Moves branch pointer and unstages changes. Default reset mode. | `git reset HEAD~1` |
| `git clean` | Removes untracked files. Preview with `-n` first. | `git clean -nd`<br/>`git clean -fd` |

## Remotes and GitHub PRs

| Method | Description | Code example |
|---|---|---|
| `git remote` | Lists or configures remotes. | `git remote -v` |
| `git fetch` | Downloads refs without modifying your working branch. | `git fetch origin` |
| `git pull` | Fetches and integrates remote changes. | `git pull --rebase origin main` |
| `git push` | Uploads local commits. | `git push -u origin feature/cheatsheets` |
| GitHub PR | Push branch, open PR, request review, and merge after checks pass. | `gh pr create --fill --draft` |
| Review checkout | Check out a pull request locally. | `gh pr checkout 123` |

## Hooks, LFS, and large files

| Method | Description | Code example |
|---|---|---|
| Pre-commit hook | Runs checks before commit. | `cat .git/hooks/pre-commit`<br/>`chmod +x .git/hooks/pre-commit` |
| `pre-commit` tool | Manages hooks from config. | `pre-commit install`<br/>`pre-commit run --all-files` |
| Git LFS install | Tracks large binary files outside normal Git objects. | `git lfs install` |
| Git LFS track | Tracks model files, checkpoints, and datasets. | `git lfs track "*.pt"`<br/>`git add .gitattributes` |
| Blame | Finds who last changed lines. | `git blame path/to/file.py` |
| Bisect | Binary search history to find a bad commit. | `git bisect start`<br/>`git bisect bad`<br/>`git bisect good v1.0.0` |

## Common patterns

| Method | Description | Code example |
|---|---|---|
| Feature branch | Start work from updated main. | `git switch main`<br/>`git pull --rebase`<br/>`git switch -c feature/ml-api` |
| Commit only part of a file | Stage hunks interactively. | `git add -p src/app.py` |
| Amend last commit | Update last commit message or staged content. | `git add docs/file.md`<br/>`git commit --amend` |
| Squash local commits | Clean branch before PR. | `git rebase -i main` |
| Resolve conflict | Edit conflict markers, stage, continue. | `git status`<br/>`git add fixed_file.py`<br/>`git rebase --continue` |
| See file history | Inspect commits touching one file. | `git log --oneline -- path/to/file.py` |
| Recover lost commit | Use reflog to find previous HEADs. | `git reflog`<br/>`git switch -c recovered abc1234` |
| Ignore local file | Add patterns to `.gitignore`. | `.venv/`<br/>`__pycache__/`<br/>`.env` |

## Senior workflow and release engineering

| Method | Description | Code example |
|---|---|---|
| Trunk-based workflow | Keep branches short-lived and integrate behind feature flags. | `git switch main`<br/>`git pull --rebase`<br/>`git switch -c small-change` |
| Release branch | Stabilize a release while main continues. | `git switch -c release/2026.05 main`<br/>`git tag v2026.05.0` |
| Signed tags | Mark trusted release points. | `git tag -s v1.2.0 -m "Release v1.2.0"`<br/>`git push origin v1.2.0` |
| Conventional commits | Make changelog automation easier. | `git commit -m "feat(api): add batch prediction endpoint"` |
| Changelog from tags | Generate release notes from commit history. | `git log v1.1.0..v1.2.0 --oneline` |
| CODEOWNERS | Require reviews from responsible teams. | `docs/cheetsheet/ @docs-team`<br/>`src/api/ @backend-team` |
| Protected branch | Require PR checks before merge. | `# Configure in GitHub branch protection: require CI, review, linear history.` |
| Release hotfix | Branch from tag, patch, tag, merge back. | `git switch -c hotfix/v1.2.1 v1.2.0`<br/>`git cherry-pick fix_commit` |

## Advanced debugging and history surgery

| Method | Description | Code example |
|---|---|---|
| Bisect with script | Automatically find regression commit. | `git bisect start`<br/>`git bisect bad`<br/>`git bisect good v1.0.0`<br/>`git bisect run ./scripts/test.sh` |
| Pickaxe search | Find commits that added or removed a string. | `git log -S "predict_batch" -- src` |
| Move commits to new branch | Recover work accidentally committed on wrong branch. | `git branch feature/recovered`<br/>`git reset --hard origin/main` |
| Split a commit | Reset softly, stage hunks, recommit. | `git reset --soft HEAD~1`<br/>`git add -p`<br/>`git commit -m "part one"` |
| Preserve merges during rebase | Rebase complex branch while keeping merge structure. | `git rebase --rebase-merges main` |
| Rerere | Reuse recorded conflict resolutions. | `git config rerere.enabled true` |
| Worktrees | Check out multiple branches without stashing. | `git worktree add ../repo-release release/2026.05` |
| Sparse checkout | Work with part of a large monorepo. | `git sparse-checkout init --cone`<br/>`git sparse-checkout set docs src/api` |

## ML repository governance

| Method | Description | Code example |
|---|---|---|
| Keep data out of Git | Version data by URI/checksum, not raw datasets. | `data_uri: s3://bucket/datasets/churn/v4`<br/>`sha256: abc123` |
| Track model artifacts | Use LFS only for small-to-medium artifacts; prefer registries for large models. | `git lfs track "*.onnx"`<br/>`git lfs track "*.safetensors"` |
| Reproducibility metadata | Commit code and log run metadata together. | `git rev-parse HEAD`<br/>`git diff --stat` |
| Pre-commit quality gate | Run formatters, linters, secret scans. | `repos:`<br/>`  - repo: https://github.com/pre-commit/pre-commit-hooks` |
| Secret scanning | Prevent credentials in history. | `detect-secrets scan > .secrets.baseline` |
| Large file audit | Find oversized files before push. | `git rev-list --objects --all &#124; git cat-file --batch-check` |
