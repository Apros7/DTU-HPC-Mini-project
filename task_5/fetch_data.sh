#!/usr/bin/env bash
# Fetch a subset of the wall-heating floorplan data from DTU HPC to a
# local directory on your Mac/laptop.
#
# Run this ON YOUR LOCAL MACHINE (not on the HPC).
#
# Requires:
#   - rsync
#   - ssh access to DTU HPC, configured under the host alias "HPC"
#     (this is what you have in ~/.ssh/config:
#        Host HPC
#          HostName login.hpc.dtu.dk
#          User    s224195
#     )
#
# Configuration via env vars:
#   N        number of floorplans to download (default 100)
#   DEST     local destination dir
#            (default: /Users/lucasvilsen/Documents/DTU/mini_project_hpc/data)
#   HPC_HOST ssh host alias (default: HPC)
#   REMOTE   remote data dir
#            (default: /dtu/projects/02613_2025/data/modified_swiss_dwellings/)

set -euo pipefail

N=${N:-1000}
DEST=${DEST:-/Users/lucasvilsen/Documents/DTU/mini_project_hpc/data}
HPC_HOST=${HPC_HOST:-HPC}
REMOTE=${REMOTE:-/dtu/projects/02613_2025/data/modified_swiss_dwellings/}

mkdir -p "$DEST"

echo "Fetching $N floorplans"
echo "  from : ${HPC_HOST}:${REMOTE}"
echo "  to   : $DEST"

# 1. Get the index file.
rsync -avh --progress \
    "${HPC_HOST}:${REMOTE}building_ids.txt" \
    "$DEST/"

# 2. Build the file list of {bid}_domain.npy + {bid}_interior.npy
#    for the first N building IDs.
TMP_LIST=$(mktemp)
trap 'rm -f "$TMP_LIST"' EXIT
head -n "$N" "$DEST/building_ids.txt" | awk '{
    print $1"_domain.npy"
    print $1"_interior.npy"
}' > "$TMP_LIST"

# 3. Pull only those files.
rsync -avh --progress --files-from="$TMP_LIST" \
    "${HPC_HOST}:${REMOTE}" \
    "$DEST/"

echo ""
echo "Done. Local data dir: $DEST"
echo "Then run:"
echo "    cd task_5"
echo "    LOAD_DIR=$DEST N=$N ./run_speedup_local.sh"
echo "    uv run analyze_speedup.py"
