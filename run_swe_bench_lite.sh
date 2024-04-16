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
per_instance_cost_limit="2.00"
config_file="./config/default.yaml"
suffix="run_2"
# split="dev"
split="test"

# Check if the user wants to use Docker or not
use_docker=true

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
    python run.py --model_name "$model_name" --per_instance_cost_limit "$per_instance_cost_limit" --config_file "$config_file" --suffix "$suffix" --split "$split"
fi