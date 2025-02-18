#!/bin/bash

# Name of the tmux session
SESSION_NAME="diffusion_policy_server"

# Check if the tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Session $SESSION_NAME already exists"
    tmux attach-session -t $SESSION_NAME
    exit 0
fi

# Create a new tmux session
tmux new-session -d -s $SESSION_NAME

# Send commands to the tmux session
# Activate conda environment and run the server
tmux send-keys -t $SESSION_NAME "conda activate robodiff" C-m
tmux send-keys -t $SESSION_NAME "python pa_arm_server.py" C-m

echo "Session $SESSION_NAME created and activated"
echo "Server started on port 10012"
echo "To attach to the session, run: \"tmux a -t $SESSION_NAME\""
