#!/usr/bin/env bash
set -e

# 1. Start Streamlit on internal port 8501
echo "Starting Streamlit on internal port 8501..."
streamlit run dashboard.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.enableCORS False \
    --server.enableXsrfProtection False &

# 2. Start FastAPI on port 7860 (primary public port — validator hits this)
echo "Starting FastAPI on port 7860..."
python -m uvicorn server.app:app \
    --host 0.0.0.0 \
    --port 7860 \
    --proxy-headers \
    --forwarded-allow-ips="*" &
FASTAPI_PID=$!

# 3. Wait for FastAPI to be healthy
echo "Waiting for FastAPI..."
MAX_RETRIES=30
COUNT=0
while ! curl -s http://localhost:7860/health > /dev/null; do
    sleep 1
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "Error: FastAPI failed to start after $MAX_RETRIES seconds."
        exit 1
    fi
done
echo "FastAPI is UP on port 7860!"

# 4. Keep container alive by waiting on FastAPI process
wait $FASTAPI_PID