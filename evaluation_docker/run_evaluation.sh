#!/bin/bash

# if [ "$#" -lt 2 ]; then
#     echo "Usage: $0 <directory> <swe_bench_tasks>"
#     exit 1
# fi

directory=$1
# swe_bench_tasks=$2-princeton-nlp/SWE-bench
swe_bench_tasks="${2:-princeton-nlp/SWE-bench}"
predictions_path=$(pwd)/${directory}/all_preds.jsonl


# Default values for the optional arguments
dataset_name_or_path="${2:-princeton-nlp/SWE-bench}"
dataset_name_or_path="${2:-princeton-nlp/SWE-bench_Lite}"
results_dir="${3:-results}"
# testbed_dir="${4:-testbed}"
testbed_dir="${4:-/testbed}" # place the testbed dir in the root to make the path length smaller

# create the results dir in the folder corresponding to the predictions all_preds.jsonl file
results_dir=$(dirname "$predictions_path")/eval_logs
# get absolute path for the results dir
results_dir=$(realpath "$results_dir")

echo "results_dir: $results_dir"

# If results or testbed directories do not exist, create them
if [ ! -d "$results_dir" ]; then
    mkdir -p "$results_dir"
    echo "Created results directory at $results_dir"
fi

if [ ! -d "$testbed_dir" ]; then
    mkdir -p "$testbed_dir"
    echo "Created testbed directory at $testbed_dir"
fi

python run_evaluation.py \
 --predictions_path ${predictions_path} \
 --log_dir ${results_dir} \
 --swe_bench_tasks ${swe_bench_tasks} \
 --num_processes 10
#  --skip_existing

python generate_report.py \
 --predictions_path ${predictions_path} \
 --log_dir ${results_dir} \
 --output_dir $(pwd)/${directory} \
 --swe_bench_tasks ${swe_bench_tasks}
