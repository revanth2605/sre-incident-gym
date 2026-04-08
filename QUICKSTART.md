# Quick Start Guide 🚀

## 30-Second Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python main.py
```

You'll see:
```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:7860
```

### 3. Test in Another Terminal
```bash
# Test health endpoint
curl http://localhost:7860/health

# Test reset (Task 1)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": 1}'

# Test step (restart service)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "restart_service"}}'

# Get current state
curl http://localhost:7860/state
```

---

## Full Evaluation

```bash
# Terminal 1: Start server
python main.py

# Terminal 2: Run inference script
export API_BASE_URL="http://localhost:7860"
export MODEL_NAME="gpt-4"
export OPENAI_API_KEY="sk-..."
python inference.py
```

Expected output:
```
[START] task=task_easy_restart env=sre-gym model=gpt-4
[STEP] step=1 action=check_logs reward=0.20 done=false error=null
[STEP] step=2 action=restart_service reward=0.80 done=true error=null
[END] success=true steps=2 score=0.80 rewards=0.20,0.80
...
Overall Score: 0.80/1.00
```

---

## Docker Quick Start

```bash
# Build image
docker build -t sre-incident-gym:latest .

# Run container
docker run -p 7860:7860 sre-incident-gym:latest

# Test
curl http://localhost:7860/health
```

---

## File Structure

```
sre-incident-gym/
├── models.py              # Pydantic V2 models
├── environment.py         # Core gym logic
├── main.py               # FastAPI server
├── inference.py          # Judge evaluation script
├── openenv.yaml          # OpenEnv specification
├── Dockerfile            # Deployment container
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Package configuration
├── README.md             # Full documentation
├── LICENSE               # MIT License
├── .gitignore            # Git ignore rules
└── QUICKSTART.md         # This file
```

---

## Common Issues

### Port 7860 Already in Use
```bash
# Find process using port
lsof -i :7860

# Kill it
kill -9 <PID>
```

### ImportError: No module named 'fastapi'
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Docker build fails
```bash
# Build without cache
docker build --no-cache -t sre-incident-gym:latest .
```

---

## Next Steps

1. **Read Full Documentation:** See `README.md` for complete API and task details
2. **Understand Tasks:** Review Task 1, 2, 3 in README.md
3. **Implement Agent:** Build your AI agent using the `/reset`, `/step`, `/state` endpoints
4. **Validate:** Run `openenv validate` to check compliance
5. **Deploy:** Push to Hugging Face Spaces

---

## API Documentation

Interactive API docs available at:
- **Swagger UI:** http://localhost:7860/docs
- **ReDoc:** http://localhost:7860/redoc

---

## Support

- Check README.md for comprehensive documentation
- Review example agent sessions in README.md
- Check openenv.yaml for complete spec

Good luck! 🎉
