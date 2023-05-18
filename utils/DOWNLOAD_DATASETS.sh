mkdir ../data
wget --recursive --no-parent --no-host-directories --cut-dirs=2 --relative -A zip https://diffusion-policy.cs.columbia.edu/data/training/ -P ../data
unzip "*.zip"

