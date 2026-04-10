#!/usr/bin/env bash
# Launcher for backend (FastAPI) + frontend (Streamlit)
set -e

# 1. Start FastAPI backend on port 8000
# We use & to run it in the background
echo "Starting FastAPI backend on port 8000..."
python main.py &

# 2. Wait for the Backend to be healthy before starting the UI
# This prevents the "Connection Refused" error in Streamlit
echo "Waiting for FastAPI to be ready..."
MAX_RETRIES=30
COUNT=0

while ! curl -s http://localhost:8000/health > /dev/null; do
    sleep 1
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "Error: Backend failed to start after $MAX_RETRIES seconds."
        exit 1
    fi
done

echo "Backend is UP and healthy! Starting Streamlit..."

# 3. Start Streamlit (Port 7860 is required for Hugging Face)
# 'exec' replaces the shell with the streamlit process so it catches shutdown signals correctly
exec streamlit run dashboard.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.enableCORS False \
    --server.enableXsrfProtection False