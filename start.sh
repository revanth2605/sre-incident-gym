#!/usr/bin/env bash
set -e

# 1. Start FastAPI backend on port 7860 (from the new location)
echo "Starting FastAPI backend on port 7860..."
python -m server.app &
FASTAPI_PID=$!

echo "Waiting for FastAPI to be ready..."
MAX_RETRIES=30
COUNT=0
while ! curl -s http://localhost:7860/health > /dev/null; do
    sleep 1
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "Error: Backend failed to start after $MAX_RETRIES seconds."
        exit 1
    fi
done

echo "Backend is UP! Starting Streamlit on port 8501..."
# Note: Streamlit stays at the root or wherever your dashboard.py is
streamlit run dashboard.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.enableCORS False \
    --server.enableXsrfProtection False &

# Keep the container alive
wait $FASTAPI_PID