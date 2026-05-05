
#%%
# ============================================================
# config.py — Multi-Agent AI Business Report Generator
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_FAST = "llama-3.1-8b-instant"    # for multi-agent
LLM_MODEL      = "llama-3.3-70b-versatile"  # for single calls
# Rate limit settings
REQUEST_DELAY = 15  # seconds between agent calls
# Paths
OUTPUT_FOLDER = "outputs"
DATA_FOLDER   = "data"

import os
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

print("✅ Config loaded")
# %%
