#!/usr/bin/env bash
# Simple launcher for backend (FastAPI) + frontend (Streamlit)
set -e

# 1. Start FastAPI on port 8000 (internal)
echo "Starting FastAPI backend on port 8000 (0.0.0.0)..."
python main.py --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Started backend (pid=$BACKEND_PID). Waiting for startup..."
sleep 5

# 2. Start Streamlit (bind to 0.0.0.0:7860 for HF Spaces)
echo "Starting Streamlit on port 7860..."
exec streamlit run dashboard.py --server.port 7860 --server.address 0.0.0.0
