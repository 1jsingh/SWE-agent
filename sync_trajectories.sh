#!/bin/bash
# Sync script to run periodically every second

CONTAINER_ID="acae6c2929dd"
CONTAINER_PATH="/app/trajectories/root"  # Assuming data is written here
HOST_PATH="$(pwd)/trajectories/"  # Sync to this directory on host
JSON_FILE="root/azure-gpt4__SWE-bench_Lite__default__t-0.00__p-0.95__c-2.00__install-1__run_dev_3/all_preds.jsonl"  # Path to the JSON Lines file

# Function to check if the Docker container is running
container_is_running() {
    docker inspect -f '{{.State.Running}}' "$CONTAINER_ID" 2>/dev/null || return 1
}

# Main loop to sync data and track progress
while container_is_running; do
    # Sync data
    docker cp "$CONTAINER_ID:$CONTAINER_PATH" "$HOST_PATH" || break

    # Count entries in JSON Lines file and print progress
    entry_count=$(wc -l < "$HOST_PATH/$JSON_FILE")
    echo "Number of entries in all_preds.jsonl: $entry_count"

    sleep 1  # Wait for 1 second before syncing again
done

echo "Container $CONTAINER_ID is closed."