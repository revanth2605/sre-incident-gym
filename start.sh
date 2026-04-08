#!/usr/bin/env bash
# Simple launcher for backend (FastAPI) + frontend (Streamlit)
set -e

# Start FastAPI backend in background
python main.py &
BACKEND_PID=$!

echo "Started backend (pid=$BACKEND_PID). Waiting for startup..."
sleep 2

# Start Streamlit (bind to 0.0.0.0:7860 for HF Spaces)
exec streamlit run dashboard.py --server.port 7860 --server.address 0.0.0.0
