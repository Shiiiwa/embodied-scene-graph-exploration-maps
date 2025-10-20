#!/usr/bin/env bash
set -euo pipefail

# Projektroot bestimmen (2 Ebenen hoch von /src/scripts)
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Logs liegen relativ zum Projektroot
LOGDIR="$PROJECT_ROOT/data/runs"
PORT="${PORT:-7007}"
HOST="${HOST:-127.0.0.1}"

# Alten Prozess auf Port killen
if lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT ist belegt. Beende Prozess..."
  lsof -tiTCP:$PORT -sTCP:LISTEN | xargs -r kill
  sleep 1
fi

# Sicherstellen, dass es den Ordner gibt
mkdir -p "$LOGDIR"

echo "Starte TensorBoard: logdir=$LOGDIR host=$HOST port=$PORT"
nohup tensorboard --logdir="$LOGDIR" --host="$HOST" --port="$PORT" --reload_interval=2 > "$PROJECT_ROOT/tensorboard.log" 2>&1 &
echo $! > "$PROJECT_ROOT/tensorboard.pid"
echo "TensorBoard PID: $(cat "$PROJECT_ROOT/tensorboard.pid")"
echo "Öffne: http://$HOST:$PORT/"
