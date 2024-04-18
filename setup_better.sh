#!/usr/bin/env bash

# bash strict mode
set -euo pipefail

# Default architecture if not provided as a command line argument
TARGETARCH="${1:-amd64}"

echo "Setting up docker image for swe-agent with architecture $TARGETARCH..."
docker build --build-arg TARGETARCH="$TARGETARCH" -t sweagent/swe-agent:latest -f docker/swe.Dockerfile .

echo "Setting up docker image for evaluation with architecture $TARGETARCH..."
docker build --build-arg TARGETARCH="$TARGETARCH" -t sweagent/swe-eval:latest -f docker/eval.Dockerfile .

echo "Done with setup!"

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
