#!/bin/bash
# =============================================================================
# sync_code_to_eris.sh
# Mirror the repo (everything NOT gitignored) to the lab's ERIS code folder.
# Full mirror: local adds/changes are copied over, AND anything gitignored
# or removed locally is deleted remotely too -- remote always matches the
# local working tree exactly.
#
# This is separate from sync_storage_to_eris.sh, which targets a different
# remote folder and is additive-only (never deletes).
#
# Usage:
#   bash hpc/sync_code_to_eris.sh
# =============================================================================
set -euo pipefail

REMOTE_HOST="vs552@erisxdl6.research.partners.org"
REMOTE_DEST="/data/oasiis-lab/funxion_code"
LOCAL_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/"

ssh "$REMOTE_HOST" "mkdir -p '${REMOTE_DEST}'"

# --delete: remote is an exact mirror of local. --filter=':- .gitignore'
# applies each directory's .gitignore as exclude rules (same effect as
# `git ls-files`, without requiring git on the remote side).
rsync -avz --delete \
    --exclude='.git/' \
    --filter=':- .gitignore' \
    "$LOCAL_SRC" \
    "${REMOTE_HOST}:${REMOTE_DEST}/"

echo "Code sync complete: ${REMOTE_HOST}:${REMOTE_DEST}"
