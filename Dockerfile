FROM python:3.10-slim

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app

WORKDIR $HOME/app

# Copy files and install dependencies (cache-friendly)
COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir -r $HOME/app/requirements.txt

# Copy project files and ensure the 'user' owns them
COPY --chown=user . $HOME/app

# Train the RL model during build so it's ready at runtime
RUN mkdir -p $HOME/app/models && \
    python rl_train.py --timesteps 30000

# Ensure the script is executable
RUN chmod +x $HOME/app/start.sh

EXPOSE 7860

# FastAPI is on 7860 (primary public port — validator hits this)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
  CMD curl -f http://127.0.0.1:7860/health || exit 1

CMD ["./start.sh"]