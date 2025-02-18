#!/bin/bash

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# S3 source
S3_SOURCE="s3://ml-checkpoints/diffusion_policy/"

# Check if aws cli is installed
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed. Please install it first."
    exit 1
fi

echo "Starting download of checkpoint files from $S3_SOURCE..."

# Download all files from S3
aws s3 sync "$S3_SOURCE" "$SCRIPT_DIR" \
    --exclude "*.sh" \
    --exclude "*.md" \
    --exclude ".*"

echo "Download completed successfully!"
