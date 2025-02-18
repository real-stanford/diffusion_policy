#!/bin/bash

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# S3 destination
S3_DEST="s3://ml-checkpoints/diffusion_policy/"

# Check if aws cli is installed
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if we have any files to upload
if [ -z "$(ls -A $SCRIPT_DIR)" ]; then
    echo "No files found in $SCRIPT_DIR"
    exit 0
fi

echo "Starting upload of checkpoint files to $S3_DEST..."

# Upload all files in the directory to S3
aws s3 sync "$SCRIPT_DIR" "$S3_DEST" \
    --exclude "*.sh" \
    --exclude "*.md" \
    --exclude ".*"

echo "Upload completed successfully!"
