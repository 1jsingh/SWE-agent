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
# Number of tasks to run the evaluation on (default is -1, which means all tasks)
num_tasks=1
start_index=20

# data split
split="dev"
# split="test"


####################################################################################################
# default configuration file
config_file="./config/default.yaml"
suffix="${split}_${start_index}_${num_tasks}_baseline__testrun_1"

####################################################################################################
# State-react configuration file
# config_file="./config/default_epllm-v0.1.yaml"

####################################################################################################
# hep-llm configuration file
# config_file="./config/default_hepllm_v0.1.yaml"

config_file="./config/hepllm/default-v4-root-level.yaml"
use_hepllm=true
suffix="${split}_${start_index}_${num_tasks}_hepllm-stateless-level-2__testrun_1"
# experiment suffix
# suffix="state-react__run_1"
# suffix="state-reactv2__testrun_7"
# suffix="${split}_${start_index}_${num_tasks}_baseline__testrun_2"


####################################################################################################
# use gold patch
####################################################################################################
use_gold_patches=false

if [ "$use_gold_patches" = true ]; then
    suffix="${split}_${start_index}_${num_tasks}_gold_patch"
fi
####################################################################################################

run_inference=true
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
            --skip_existing="$skip_existing"
    fi
fi

####################################################################################################
# evaluation on the SWE benchmark
####################################################################################################

run_eval=true

if [ "$run_eval" = true ]; then
    # The first positional argument ... compute run_path based on the run script
    dataset_name_or_path="princeton-nlp/SWE-bench_Lite"
    data_stem=$(basename $dataset_name_or_path)
    # config stem without the .yaml in the end
    config_stem=$(basename $config_file .yaml)
    # cost upto two decimals
    cost=$(printf "%.2f" $per_instance_cost_limit)
    predictions_path="trajectories/$USER/azure-gpt4__${data_stem}__${split}__${config_stem}__t-0.00__p-0.95__c-${cost}__install-1__${suffix}/all_preds.jsonl"
    echo "evaluating predictions_path ... $predictions_path"
    # predictions_path=$1
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
        python evaluation/evaluation.py \
            --predictions_path "$predictions_path" \
            --swe_bench_tasks "$dataset_name_or_path" \
            --log_dir "$results_dir" \
            --testbed "$testbed_dir" \
            # --skip_existing \
            --timeout 900 \
            --verbose
    fi
fi