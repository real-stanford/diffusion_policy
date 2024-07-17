if [ "$2" -lt 0 ]; then
  echo "# of episodes not valid"
  exit
fi

echo "Task: $1"
for (( i=0; i<$2; i++ ))
do
  echo "Starting episode $i"
  python3 record_episodes.py --task "$1"
  if [ $? -ne 0 ]; then
    echo "Failed to execute command. Returning"
    exit
  fi
done