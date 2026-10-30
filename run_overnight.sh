#!/bin/zsh
# Keeps the Mac awake and restarts training from the last checkpoint if it crashes.
# usage: ./run_overnight.sh <hours> [extra train.py args...]
cd "$(dirname "$0")"
HOURS=$1; shift
for attempt in 1 2 3 4 5 6; do
  echo "=== attempt $attempt $(date) ===" >> run/overnight.log
  caffeinate -ims .venv/bin/python train.py --hours "$HOURS" --resume "$@" >> run/overnight.log 2>&1
  [ -f run/DONE ] && break
  sleep 20
done
echo "=== finished $(date) ===" >> run/overnight.log
