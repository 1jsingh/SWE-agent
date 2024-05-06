#!/bin/bash

# The first positional argument
predictions_path=$1

# Check if predictions_path is not provided
if [ -z "$predictions_path" ]; then
    echo "Usage: $0 <predictions_path> [dataset_name_or_path] [results_dir] [testbed_dir]"
    exit 1
fi

# Default values for the optional arguments
dataset_name_or_path="${2:-princeton-nlp/SWE-bench}"
results_dir="${3:-results}"
# testbed_dir="${4:-testbed}"
testbed_dir="${4:-/testbed}" # place the testbed dir in the root to make the path length smaller


# If results or testbed directories do not exist, create them
if [ ! -d "$results_dir" ]; then
    mkdir -p "$results_dir"
    echo "Created results directory at $results_dir"
fi

if [ ! -d "$testbed_dir" ]; then
    mkdir -p "$testbed_dir"
    echo "Created testbed directory at $testbed_dir"
fi

# # Check if the user wants to use Docker or not
# use_docker=true

# ####################################################################################################
# # run the SWE benchmark
# ####################################################################################################
# if [ "$use_docker" = true ]; then
#     docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock \
#         -v "$(pwd)/keys.cfg:/app/keys.cfg" \
#         -v "$(pwd)/trajectories:/app/trajectories" \
#         sweagent/swe-agent-run:latest \
#         python run.py --image_name=sweagent/swe-agent:latest \
#             --model_name "$model_name" \
#             --per_instance_cost_limit "$per_instance_cost_limit" \
#             --config_file "$config_file" \
#             --suffix "$suffix" \
#             --split "$split"
# else
#     python run.py --model_name "$model_name" --per_instance_cost_limit "$per_instance_cost_limit" --config_file "$config_file" --suffix "$suffix" --split "$split"
# fi

# Check if the user wants to use Docker or not
use_docker=false

####################################################################################################
# evaluation on the SWE benchmark
####################################################################################################

if [ "$use_docker" = true ]; then
    docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock \
        -v "$(pwd)/keys.cfg:/app/keys.cfg" \
        -v "$(pwd)/../trajectories:/trajectories" \
        -v "$(pwd)/results:/results" \
        -v "$(pwd)/testbed:/testbed" \
        sweagent/swe-eval:latest \
        python /evaluation.py \
            --predictions_path "$predictions_path" \
            --swe_bench_tasks "$dataset_name_or_path" \
            --log_dir "$results_dir" \
            --testbed "$testbed_dir" \
            --skip_existing \
            --timeout 900 \
            --verbose
else
    python evaluation.py \
        --predictions_path "$predictions_path" \
        --swe_bench_tasks "$dataset_name_or_path" \
        --log_dir "$results_dir" \
        --testbed "$testbed_dir" \
        --timeout 900 \
        --verbose \
        --num_processes 4 \
        --skip_existing
fi

# # Run the Python script with the specified arguments
# python evaluation.py \
#     --predictions_path "$predictions_path" \
#     --swe_bench_tasks "$dataset_name_or_path" \
#     --log_dir "$results_dir" \
#     --testbed "$testbed_dir" \
#     --skip_existing \
#     --timeout 900 \
#     --verbose

# docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock \
#     -v "$(pwd)/keys.cfg:/app/keys.cfg" \
#     -v "$(pwd)/../trajectories:/trajectories" \
#     -v "$(pwd)/results:/results" \
#     -v "$(pwd):/evaluation" \
#     sweagent/swe-eval:latest \
#     bash