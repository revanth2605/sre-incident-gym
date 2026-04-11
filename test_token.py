# test_token.py
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or "no-key"
print(f"API Key being used: {api_key[:20]}..." if api_key != "no-key" else "API Key: no-key")
print(f"Token length: {len(api_key)}")
print(f"First 10 chars: {api_key[:10] if api_key else 'EMPTY'}")