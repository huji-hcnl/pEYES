---
name: promote-to-main
description: >
  Automates this repo's dev -> main promotion flow (documented in docs/GIT_WORKFLOW.md): merge the
  current session branch into dev with --no-ff, push dev, and open a PR from dev into main. Also
  handles finishing up afterward - fast-forwarding local main and dev once the PR is merged, pushing
  dev back to origin, and rebasing the session branch onto the refreshed dev. Use this whenever the
  user wants to "promote", "ship", "merge to main", "open a PR to main", "put this in dev", or "finish
  up" / "sync main and dev" / "rebase onto dev" after a PR they merged manually on GitHub. Covers both
  a manual-review mode (default: stop after opening the PR so the user can review it themselves) and
  an auto mode (merge the PR immediately and finish the sync in one go) - ask which one they want if
  it isn't clear from their phrasing.
---

# Promote to main

This encodes the "dev -> main" half of the git workflow in [docs/GIT_WORKFLOW.md](../../../docs/GIT_WORKFLOW.md).
Read that file if anything here seems to conflict with it - it's the source of truth, this skill is
just automation on top of it.

The workflow has two phases that may happen in the same conversation or two different ones (the user
might come back in a fresh session after reviewing the PR on GitHub). Figure out which phase applies
from what the user asks for and the current repo state - don't assume.

- **Open** - merge the session branch into `dev`, push it, open the PR. Ends either by stopping for
  manual review, or by continuing straight into Sync if the user asked for "auto".
- **Sync** - after the PR is merged (by you or by the user on GitHub), fast-forward local `main` and
  `dev` to match, push `dev`, and rebase the session branch onto the new `dev`.

If the user's request doesn't make the mode obvious, ask: manual review (default - safer, since
merging into `main` is hard to reverse and affects a shared branch) or auto (merge immediately and
finish the sync in the same run)?

## Before doing anything: orient yourself

Run these and reason from the actual output - don't assume branch layout or worktree locations,
since this repo runs sessions in worktrees and keeps a separate persistent checkout for `main`.

```bash
git status --porcelain          # must be empty - abort and tell the user to commit/stash otherwise
git branch --show-current       # this is your session branch - remember it, you'll need it later
git worktree list --porcelain   # shows every checkout and which branch (if any) each has out
```

From the worktree list, note the filesystem path (if any) where `main` is checked out, and likewise
for `dev`. A branch not listed there isn't checked out anywhere, which means you're free to move its
ref directly (`git branch -f <branch> <target>`) or check it out in your own worktree temporarily.
A branch that *is* checked out elsewhere must be updated there instead, via `git -C <that-path> ...`
- never assume the branch you need is checked out in your own working directory just because that
would be convenient. Getting this wrong risks silently moving the wrong branch (see the pitfall
section in docs/GIT_WORKFLOW.md) or failing outright (git refuses to update a branch checked out
elsewhere).

If your own current branch is `main` or `dev`, stop and ask the user what they meant - this skill
promotes a session branch that already has commits on it, not `main`/`dev` themselves.

## Phase: Open

1. **Merge into `dev` with `--no-ff`.** If `dev` isn't checked out anywhere, check it out in your own
   worktree, merge, then switch back to your session branch afterward. If it's checked out elsewhere,
   run the merge there with `git -C <dev-path> merge --no-ff <session-branch> -m "..."`. Either way,
   use a real merge commit message describing what the branch contributed - not a generic "merge"
   message - so `--no-ff` actually earns its keep of keeping history legible.

2. **Push `dev`:** `git push origin dev`.

3. **Open the PR:**
   ```bash
   gh pr create --base main --head dev --title "<title>" --body "<body>"
   ```
   Use a title/body the user gave you, or write one from what actually changed (`git log
   main..dev --oneline`, `git diff main...dev --stat`) - don't just say "promote to main". Note the
   PR URL/number gh prints; you'll want it for Sync.

   Before creating it, it's worth telling the user in one line what's actually about to ride along -
   `dev` is a shared integration branch, so if anything besides the session branch you just merged is
   already sitting on `dev` unmerged, this PR will carry that too. That's not a bug, but it can
   surprise someone who forgot about earlier work-in-progress on `dev`.

4. **Pick a mode:**
   - **Manual (default):** Stop here. Tell the user the PR is open, give them the URL, and say that
     once they've reviewed and merged it (on GitHub or via `gh pr merge`), they should come back and
     ask you to finish up / sync - that resumes at Phase: Sync below.
   - **Auto:** Merge it yourself right away, then continue straight into Phase: Sync in the same run:
     ```bash
     gh pr merge <number> --merge
     ```
     Always `--merge` here, never `--squash`/`--rebase` - the whole point of the `--no-ff` merge into
     `dev` was to keep that structure visible, and squashing/rebasing at the `main` PR would throw it
     away.

## Phase: Sync

This can start a fresh conversation with no memory of Phase: Open, so don't assume you know which PR
or branch is involved - work it out from the current repo state.

1. **Find the relevant PR and confirm it's actually merged** before touching anything:
   ```bash
   gh pr list --base main --head dev --state merged --limit 1
   ```
   or, if you already have a PR number from Phase: Open, `gh pr view <number> --json state,mergeCommit`.
   If it's not merged yet, stop and tell the user to merge it first - don't guess or proceed anyway.

2. `git fetch origin`.

3. **Fast-forward local `main`** to `origin/main`. Use the worktree-location logic from the orient
   step: if `main` is checked out somewhere, `git -C <main-path> merge --ff-only origin/main` there;
   if it's genuinely not checked out anywhere, `git branch -f main origin/main` directly. Either way
   this must be a fast-forward - if it isn't, something unexpected changed `main` out from under you
   and it needs a human look, not a forced merge.

4. **Fast-forward local `dev`** to the now-updated `main` the same way (checked out somewhere -> merge
   `--ff-only` there; not checked out anywhere -> `git branch -f dev main`).

5. **Push `dev`:** `git push origin dev` - this is what keeps `origin/dev` from quietly lagging behind
   `origin/main`, which is the whole reason this step exists (see docs/GIT_WORKFLOW.md).

6. **Rebase the session branch onto the refreshed `dev`.** This means whatever branch you (the
   current session) are actually on - if that's `dev` or `main` itself, there's nothing to rebase,
   just report that everything's in sync. Otherwise: `git rebase dev`.

7. Report the final state briefly - confirm `main`, `dev`, and the session branch all point where
   expected (a quick `git log --oneline --graph -5` on the session branch is a good sanity check).

## Guardrails

These aren't arbitrary caution - each one maps to a specific way this can go wrong in a
multi-worktree repo:

- Never push to `origin/main` directly - `main` only moves via the reviewed PR.
- Never force-update a branch ref that's checked out in another worktree - git will refuse, and
  working around that refusal (e.g. by deleting/recreating the worktree) is not this skill's job;
  surface it to the user instead.
- Never treat a merge as done because you *think* it should be - check `gh pr view`/`gh pr list`
  state before running Sync steps that assume it.
- If any precondition fails (dirty working tree, `main` not found in any worktree, PR not actually
  merged, a fast-forward that isn't actually fast-forwardable), stop and explain what's wrong rather
  than working around it or guessing intent.
