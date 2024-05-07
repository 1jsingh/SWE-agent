# This script is used to run the SWE benchmark with a lite configuration.
# It sets the model_name variable to "gpt4".

####################################################################################################
# development model names
####################################################################################################
# model_name="gpt4"
# model_name="azure:gpt-4-turbo"
# model_name="azure:gpt-4"

####################################################################################################
# model names with docker
####################################################################################################
model_name="azure:gpt4"


####################################################################################################
# experiment configurations
####################################################################################################
per_instance_cost_limit="10.0"
use_hepllm=false
skip_existing=false

# data split
split="dev"
split="test"


####################################################################################################
# gold patch result files
####################################################################################################
GOLD_PATCH_TEST="/home/jas/project/qstar/SWE-agent/trajectories/gold/azure-gpt4__SWE-bench_Lite__test__default-v4-root-level__t-0.00__p-0.95__c-10.00__install-1__test_0_all_run1_gold_patch/results.json"
GOLD_PATCH_DEV="/home/jas/project/qstar/SWE-agent/trajectories/gold/azure-gpt4__SWE-bench_Lite__dev__default-v4-root-level__t-0.00__p-0.95__c-10.00__install-1__dev_0_all_gold_patch/results.json"

if [ "$split" = "dev" ]; then
    gold_patch_results_path="$GOLD_PATCH_DEV"
else
    gold_patch_results_path="$GOLD_PATCH_TEST"
fi

use_gold_patch_filter=true
####################################################################################################

# Number of tasks to run the evaluation on (default is -1, which means all tasks)
num_tasks=1
start_index=86
if [ "$num_tasks" -eq -1 ]; then
    num_tasks_text="all"
else
    num_tasks_text="$num_tasks"
fi

####################################################################################################
# default configuration file
config_file="./config/default.yaml"
suffix="${split}_${start_index}_${num_tasks_text}_baseline"
if [ "$num_tasks" -eq 1 ] && [ "$start_index" -eq 0 ]; then
    suffix="${split}_baseline"
else
    suffix="${split}_${start_index}_${num_tasks_text}_baseline"
fi

####################################################################################################
# State-react configuration file
# config_file="./config/default_epllm-v0.1.yaml"

####################################################################################################
# hep-llm configuration file
# config_file="./config/default_hepllm_v0.1.yaml"

config_file="./config/hepllm/default-v7-root-level.yaml"
# config_file="./config/hepllm/default-v5-root-level.yaml"
# config_file="./config/hepllm/default-v6-root-level.yaml"

use_hepllm=true
suffix="${split}_hepllm-lv2-r7-l5__indv-run-5"

# suffix="${split}_${start_index}_${num_tasks_text}_hepllm-lv2-r5-l5__testrun_3"

# experiment suffix
# suffix="state-react__run_1"
# suffix="state-reactv2__testrun_7"
# suffix="${split}_${start_index}_${num_tasks}_baseline__testrun_2"

####################################################################################################
exp_subdir="hepllm-v0.3"

####################################################################################################
# use gold patch
####################################################################################################
use_gold_patches=false

if [ "$use_gold_patches" = true ]; then
    num_tasks_text="all"
    exp_subdir="gold"
    # suffix="${split}_${start_index}_${num_tasks_text}_run1_gold_patch"
    suffix="${split}_gold_patch"
fi
####################################################################################################

# sleep for 2 sec to allow saving the logs
# sleep 2

run_inference=false
# define run_eval to be true if run_inference is false
if [ "$run_inference" = true ]; then 
    run_eval=false
else
    run_eval=true
fi
run_eval=true

if [ "$run_inference" = true ]; then 

    # Check if the user wants to use Docker or not
    use_docker=false

    ####################################################################################################
    # run the SWE benchmark
    ####################################################################################################
    if [ "$use_docker" = true ]; then
        docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock \
            -v "$(pwd)/keys.cfg:/app/keys.cfg" \
            -v "$(pwd)/trajectories:/app/trajectories" \
            sweagent/swe-agent-run:latest \
            python run.py --image_name=sweagent/swe-agent:latest \
                --model_name "$model_name" \
                --per_instance_cost_limit "$per_instance_cost_limit" \
                --config_file "$config_file" \
                --suffix "$suffix" \
                --split "$split"
    else
        python run.py \
            --model_name "$model_name" \
            --per_instance_cost_limit "$per_instance_cost_limit" \
            --config_file "$config_file" \
            --suffix "$suffix" \
            --split "$split" \
            --num_tasks="$num_tasks" \
            --start_index="$start_index" \
            --use_gold_patches="$use_gold_patches" \
            --use_hepllm="$use_hepllm" \
            --hepllm_levels=2 \
            --exp_subdir="$exp_subdir" \
            --filter_gold_patch_positives="$use_gold_patch_filter" \
            --gold_patch_results_file="$gold_patch_results_path" \
            --skip_existing="$skip_existing"
    fi
fi

####################################################################################################
# evaluation on the SWE benchmark
####################################################################################################

if [ "$run_eval" = true ]; then
    # The first positional argument ... compute run_path based on the run script
    dataset_name_or_path="princeton-nlp/SWE-bench_Lite"
    data_stem=$(basename $dataset_name_or_path)
    # config stem without the .yaml in the end
    config_stem=$(basename $config_file .yaml)
    # cost upto two decimals
    cost=$(printf "%.2f" $per_instance_cost_limit)
    predictions_path="trajectories/${exp_subdir}/azure-gpt4__${data_stem}__${split}__${config_stem}__t-0.00__p-0.95__c-${cost}__install-1__${suffix}/all_preds.jsonl"
    # predictions_path="trajectories/gold/check-harness.jsonl"
    echo "evaluating predictions_path ... $predictions_path"
    # predictions_path=$1
    # Check if predictions_path is not provided
    if [ -z "$predictions_path" ]; then
        echo "Usage: $0 <predictions_path> [dataset_name_or_path] [results_dir] [testbed_dir]"
        exit 1
    fi
    # Default values for the optional arguments
    dataset_name_or_path="${2:-princeton-nlp/SWE-bench_Lite}"
    dataset_name_or_path="princeton-nlp/SWE-bench"
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
    # Check if the user wants to use Docker or not
    use_docker=false
    ####################################################################################################
    # evaluation on the SWE benchmark
    ####################################################################################################
    if [ "$use_docker" = true ]; then
        docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock \
            -v "$(pwd)/keys.cfg:/app/keys.cfg" \
            -v "$(pwd)/trajectories:/trajectories" \
            -v "$(pwd)/evaluation:/evaluation" \
            -v "$(pwd)/results:/results" \
            -v "/testbed:/testbed" \
            sweagent/swe-eval:latest \
            python /evaluation/evaluation.py \
                --predictions_path "$predictions_path" \
                --swe_bench_tasks "$dataset_name_or_path" \
                --log_dir "$results_dir" \
                --testbed "$testbed_dir" \
                --skip_existing \
                --timeout 900 \
                --verbose
    else
        python evaluation/evaluation.py \
            --predictions_path "$predictions_path" \
            --swe_bench_tasks "$dataset_name_or_path" \
            --log_dir "$results_dir" \
            --testbed "$testbed_dir" \
            --timeout 900 \
            --num_processes 4 \
            --verbose 
            # --skip_existing
    fi
fi