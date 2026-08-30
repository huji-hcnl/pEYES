# Git workflow

This document is the source of truth for how Claude Code should use git in this repository. It applies for the
rest of the current session and to every future session, per [CLAUDE.md](../CLAUDE.md).

## Branches

- `main` is off limits. Never commit to it, merge into it, rebase it, reset it, or check it out, except on an
  explicit, specific instruction from the user in the current session.
- `dev` is the integration branch. All work funnels through it before ever reaching `main`.
- Each session/agent works on its own branch (a worktree branch if the harness supports worktrees, otherwise a
  regular feature branch), never directly on `dev` or `main`.

## The flow

worktree branch -> `dev` -> `origin/dev` -> (eventually, via PR) `main`. Each arrow needs the user to ask for it -
don't advance a step just because the previous one succeeded.

1. **Work on the session's own branch.** Small, atomic commits, each standalone where possible. Commit as you go
   rather than batching everything into one commit at the end.
2. **Rebase onto local `dev`** to pick up other changes, never onto `main`.
3. **Merge into local `dev` only when the user says so** (e.g. a feature is complete, or a meaningful chunk of
   analysis/work concluded) - not after every commit. Use a merge commit (`--no-ff`) so the branch's shape stays
   visible in history rather than being flattened.
4. **Push to `origin/dev` only when the user says so**, and only after step 3. Never push a feature/worktree
   branch straight to the remote. Never push to `origin/main`.
5. **After merging into `dev`, return to the session's own branch and rebase it onto the new `dev`** so it stays
   current for the next round of work.

## Testing cadence

Match the cost of the check to the size of the change:
- **Per commit:** run only the tests covering what you touched, not the full suite.
- **Before merging into `dev`:** run the full suite and say plainly if anything fails.

## Backup tags

Tag a backup ref (e.g. `backup/<description>`) at meaningful milestones - before a large rebase that will replay
many commits, or after completing a body of work worth being able to return to. Not on every merge, and not for
small incremental changes - a tag per commit is noise that buries the real checkpoints.

## `dev` -> `main` (PR workflow)

Only when the user explicitly asks.

1. Open a PR from `dev` into `main` (e.g. `gh pr create --base main --head dev ...`, adapting to whatever
   forge/CLI this project uses).
2. Merge it with a merge commit, not squash or rebase, to keep the same `--no-ff` convention as step 3 above.
3. Fetch, then fast-forward local `main` to the merged commit.
4. Sync local `dev` to the same commit and push it, so `dev` doesn't lag behind what just landed on `main`.
5. Tag the merged commit last, once `main` and `dev` (local and remote) all agree - ask the user for the tag name
   rather than inventing one.
6. Return to the session's own branch and rebase it onto `dev`.

**Pitfall to guard against:** if there's a persistent "main checkout" separate from per-session worktrees, don't
assume it has `main` checked out - it commonly has `dev` checked out instead, since that's what sessions rebase
onto. A bare fast-forward merge (`git merge --ff-only origin/main`) updates whichever branch is *actually* checked
out there, so it can silently move `dev` instead of `main`. Always confirm with `git branch --show-current` (or
compare `git rev-parse main dev` before and after) rather than assuming the merge landed on the intended branch.
If it lands on the wrong one, fix the other ref directly with `git branch -f <branch> origin/<branch>` - safe only
when that branch isn't checked out anywhere, so check `git worktree list` (or equivalent) first.

## General principles behind this workflow

- `main` is a protected release line; everything gets a chance to be reviewed/tested via `dev` before reaching it.
- History stays legible: `--no-ff` merges mean you can always see where a body of work started and ended, rather
  than an interleaved flat history.
- Nothing gets pushed or merged upward without an explicit ask - the workflow is safe to run unattended for local
  commits, but every step that affects shared state (a push, a merge into an integration branch, a PR) is a
  deliberate, user-requested action, not an automatic consequence of finishing a task.
