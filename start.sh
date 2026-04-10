#!/usr/bin/env bash
set -e

# 1. Start FastAPI backend on port 7860 (IMPORTANT CHANGE)
echo "Starting FastAPI backend on port 7860..."
python main.py &

# 2. Wait for backend
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

echo "Backend is UP and healthy! Starting Streamlit..."

# 3. Start Streamlit on DIFFERENT PORT (8501)
exec streamlit run dashboard.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.enableCORS False \
    --server.enableXsrfProtection False