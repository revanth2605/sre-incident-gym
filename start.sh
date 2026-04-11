#!/usr/bin/env bash
set -e

# Start FastAPI backend on internal port 8000
echo "Starting OpenEnv FastAPI server on port 8000..."
python -m uvicorn server.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --proxy-headers \
    --forwarded-allow-ips="*" &

# Wait for FastAPI to be healthy
echo "Waiting for FastAPI..."
COUNT=0
while ! curl -s http://localhost:8000/health > /dev/null; do
    sleep 1
    COUNT=$((COUNT+1))
    if [ $COUNT -ge 30 ]; then
        echo "Error: FastAPI failed to start."
        exit 1
    fi
done
echo "FastAPI is UP!"

# Start Streamlit on port 7860 (public HF Spaces port)
echo "Starting Streamlit on port 7860..."
exec streamlit run dashboard.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.enableCORS False \
    --server.enableXsrfProtection False