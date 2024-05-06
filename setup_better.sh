#!/usr/bin/env bash

# Bash strict mode to avoid common mistakes
set -euo pipefail

# Default architecture if not provided as a command line argument
TARGETARCH="${1:-amd64}"

# Enabling Docker BuildKit for better build outputs
export DOCKER_BUILDKIT=1

# echo "Setting up docker image for swe-agent with architecture $TARGETARCH..."
# docker build --build-arg TARGETARCH="$TARGETARCH" -t swehelper/swe-agent:latest -f docker/swe.Dockerfile .

# Setting up Docker image for evaluation with the specified architecture
echo "Setting up docker image for evaluation with architecture $TARGETARCH..."
docker build --build-arg TARGETARCH="$TARGETARCH" -t swehelper/swe-eval:latest -f docker/eval.Dockerfile .

# Check if Docker build was successful
if [ $? -eq 0 ]; then
    echo "Docker image setup completed successfully!"
else
    echo "Docker image setup failed."
    exit 1
fi

# #!/usr/bin/env bash

# # bash strict mode
# set -euo pipefail

# # Default architecture if not provided as a command line argument
# TARGETARCH="${1:-amd64}"
# export DOCKER_BUILDKIT=1

# # echo "Setting up docker image for swe-agent with architecture $TARGETARCH..."
# # docker build --build-arg TARGETARCH="$TARGETARCH" -t sweagent/swe-agent:latest -f docker/swe.Dockerfile .

# echo "Setting up docker image for evaluation with architecture $TARGETARCH..."
# docker build --build-arg TARGETARCH="$TARGETARCH" -t swehelper/swe-eval:latest -f docker/eval.Dockerfile .

# echo "Done with setup!"

####################################################################################################
# old setup.sh -- newer one also has TARGETARCH
####################################################################################################

# #!/usr/bin/env bash

# # bash strict mode
# set -euo pipefail

# echo "Setting up docker image for swe-agent..."
# docker build -t sweagent/swe-agent:latest -f docker/swe.Dockerfile .

# echo "Setting up docker image for evaluation..."
# docker build -t sweagent/swe-eval:latest -f docker/eval.Dockerfile .

# echo "Done with setup!"
