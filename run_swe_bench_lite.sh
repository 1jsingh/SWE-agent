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
start_index=0

# data split
split="dev"
# split="test"

####################################################################################################
# default configuration file
# config_file="./config/default.yaml"
# suffix="${split}_${start_index}_${num_tasks}_baseline__testrun_2"

####################################################################################################
# State-react configuration file
# config_file="./config/default_epllm-v0.1.yaml"

####################################################################################################
# hep-llm configuration file
# config_file="./config/default_hepllm_v0.1.yaml"

config_file="./config/hepllm/default-v2-root-level.yaml"
use_hepllm=true
suffix="${split}_${start_index}_${num_tasks}_hepllm-stateless-level-2__testrun_7"


# experiment suffix
# suffix="state-react__run_1"
# suffix="state-reactv2__testrun_7"
# suffix="${split}_${start_index}_${num_tasks}_baseline__testrun_2"



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
        --use_hepllm="$use_hepllm" \
        --hepllm_levels=2 \
        --skip_existing="$skip_existing"
fi