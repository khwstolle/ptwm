#!/usr/bin/env bash
# Pre-merge guard against stale-base squash-merges silently reverting master.
#
# WHY: our bot-generated PRs are often branched from a commit that predates
# recently-merged features. GitHub's *squash* merge takes the PR head's tree
# effect relative to its merge-base, so a stale branch can DELETE files that
# landed on master after that base — silently reverting a merged feature (this
# happened once: a stale PR's squash reverted the whole LMDB/WebDataset feature).
#
# HOW TO USE: before squash-merging any PR whose base is behind master, either
# click "Update branch" on the PR *or* run this guard against the PR head:
#
#     scripts/check_pr_no_reverts.sh <pr-branch-or-sha> [base-ref]
#
# It flags every file that exists on the base but would be DELETED by merging
# the head (the stale-revert signature) and exits non-zero if any are found.
# A clean PR — or one that has been updated onto current master — exits 0.
#
# The systemic fix is the repo setting "Require branches to be up to date before
# merging" on master; this script is the local/manual backstop.
set -euo pipefail

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "usage: $(basename "$0") <pr-branch-or-sha> [base-ref (default: origin/master)]" >&2
    exit 2
fi

HEAD_REF="$1"
BASE_REF="${2:-origin/master}"

# Refresh the base so the comparison reflects the true current master. Only the
# default origin/master base is auto-fetched; an explicit base is used as-is.
if [[ "$BASE_REF" == "origin/master" ]]; then
    git fetch --quiet origin master
fi

if ! git rev-parse --verify --quiet "$HEAD_REF^{commit}" >/dev/null; then
    echo "error: cannot resolve head ref '$HEAD_REF'" >&2
    exit 2
fi
if ! git rev-parse --verify --quiet "$BASE_REF^{commit}" >/dev/null; then
    echo "error: cannot resolve base ref '$BASE_REF'" >&2
    exit 2
fi

# Two-dot tree diff, filtered to deletions: files present on BASE but absent in
# HEAD's tree. Merging HEAD (esp. via squash) would drop these from BASE.
mapfile -t DELETED < <(git diff --diff-filter=D --name-only "$BASE_REF" "$HEAD_REF")

echo "Net tree diff of '$HEAD_REF' vs '$BASE_REF':"
git diff --stat "$BASE_REF" "$HEAD_REF" | sed 's/^/  /'
echo

if [[ ${#DELETED[@]} -gt 0 ]]; then
    echo "REVERT RISK: merging '$HEAD_REF' would delete ${#DELETED[@]} file(s) that exist on '$BASE_REF':" >&2
    printf '  - %s\n' "${DELETED[@]}" >&2
    echo >&2
    echo "If these deletions are NOT intentional, the branch is stale — update it" >&2
    echo "onto '$BASE_REF' (rebase / 'Update branch') and re-run before merging." >&2
    exit 1
fi

echo "OK: no files on '$BASE_REF' would be deleted by merging '$HEAD_REF'."
