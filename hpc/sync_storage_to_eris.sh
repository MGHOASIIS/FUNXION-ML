#!/bin/bash
# =============================================================================
# sync_storage_to_eris.sh
# Push storage/ to the lab's ERIS storage share, additive-only: new/changed
# local files are copied over, but nothing already on the remote is ever
# deleted -- even if it no longer exists locally, and even if it was put
# there by someone/something else. Remote is always a superset of local.
#
# Usage:
#   bash hpc/sync_storage_to_eris.sh
# =============================================================================
set -euo pipefail

REMOTE_HOST="vs552@erisxdl6.research.partners.org"
REMOTE_DEST="/data/oasiis-lab/funxion_classifier/storage"
LOCAL_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/storage/"

ssh "$REMOTE_HOST" "mkdir -p '${REMOTE_DEST}'"

# No --delete: files removed locally, or already present remotely from
# another source, are left untouched on the remote side.
rsync -avz \
    "$LOCAL_SRC" \
    "${REMOTE_HOST}:${REMOTE_DEST}/"

echo "Sync complete: ${REMOTE_HOST}:${REMOTE_DEST}"
