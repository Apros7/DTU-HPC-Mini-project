#!/usr/bin/env bash
# Upload local data/ to the GPU device.
#
# Run this ON YOUR LOCAL MACHINE (the laptop where fetch_data.sh deposited
# files). It will rsync the contents of $SRC into ${DEST_HOST}:${REMOTE}.
#
# Existing remote files are left in place: rsync only ships files that are
# missing or newer locally, and there is NO --delete flag, so anything
# already on the device under data/ stays.
#
# Configuration via env vars:
#   SRC        local source dir
#              (default: /Users/lucasvilsen/Documents/DTU/mini_project_hpc/data)
#   DEST_HOST  ssh user@host (default: easysort@100.115.209.2)
#   REMOTE     remote target dir, relative to $HOME on the device
#              (default: DTU-HPC-Mini-project/data/)
#   SSH_OPTS   extra options to pass to ssh (default empty)

set -euo pipefail

SRC=${SRC:-/Users/lucasvilsen/Documents/DTU/mini_project_hpc/data}
DEST_HOST=${DEST_HOST:-easysort@100.115.209.2}
REMOTE=${REMOTE:-DTU-HPC-Mini-project/data/}
SSH_OPTS=${SSH_OPTS:-}

if [[ ! -d "$SRC" ]]; then
    echo "error: SRC does not exist: $SRC" >&2
    exit 1
fi

echo "Uploading"
echo "  from : $SRC"
echo "  to   : ${DEST_HOST}:${REMOTE}"

# Make sure the remote directory exists. -p is a no-op if it already does.
ssh ${SSH_OPTS} "$DEST_HOST" "mkdir -p ${REMOTE}"

# rsync flags:
#   -a       archive mode (recurse, preserve times, etc.)
#   -v       verbose
#   -h       human-readable sizes
#   --progress  per-file progress
#   --partial   keep partial transfers so an interrupted run can resume
# No --delete: never remove anything already on the device.
rsync -avh --progress --partial \
    ${SSH_OPTS:+-e "ssh $SSH_OPTS"} \
    "$SRC/" \
    "${DEST_HOST}:${REMOTE}"

echo
echo "Done. Remote dir: ${DEST_HOST}:${REMOTE}"
