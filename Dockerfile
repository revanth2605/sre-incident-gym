FROM python:3.10-slim

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install python dependencies first for caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Make start script executable
RUN chmod +x /app/start.sh || true

# Create non-root user with UID 1000 and set ownership
RUN useradd -m -u 1000 user || true
RUN chown -R user:user /app

EXPOSE 7860

# Healthcheck to ensure service is up
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://127.0.0.1:7860/health || exit 1

USER user

CMD ["/app/start.sh"]
