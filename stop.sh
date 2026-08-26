#!/bin/bash

# Stop script for Clean Bot (PanelCleaner Server)
# Kills all processes cleanly

echo "Stopping Clean Bot..."

# Kill by PID files
for pidfile in api.pid bot.pid; do
    if [ -f "$pidfile" ]; then
        PID=$(cat "$pidfile")
        if kill -0 "$PID" 2>/dev/null; then
            echo "  Killing $pidfile (PID: $PID)..."
            kill "$PID" 2>/dev/null
        fi
        rm -f "$pidfile"
    fi
done

# Kill by process name as backup
echo "  Killing remaining processes..."
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "node index.js" 2>/dev/null || true

# Kill anything on port 10100
lsof -ti:10100 | xargs kill -9 2>/dev/null || true

# Wait a moment for processes to die
sleep 2

# Verify everything is dead
REMAINING=$(ps aux | grep -E "uvicorn main:app|node index.js" | grep -v grep | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo "  Force killing remaining processes..."
    pkill -9 -f "uvicorn main:app" 2>/dev/null || true
    pkill -9 -f "node index.js" 2>/dev/null || true
fi

echo "Clean Bot stopped."
