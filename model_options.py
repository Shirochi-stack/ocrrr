# model_options.py
"""
Centralized model catalog for ocrrr.
Returned list should mirror the main GUI model dropdown.
Updated: 2026-03-10
"""
from typing import List

def get_model_options() -> List[str]:
    return [
    
        # OpenAI Models (as of March 2026)
        # - GPT-4o/4o-mini/4-turbo/4.1/3.5-turbo retired from ChatGPT Feb 13 2026; 4o still on API but legacy
        # - GPT-5.1 retiring March 11 2026, removed
        # - GPT-5.4 released March 5 2026, GPT-5.3 Instant released March 3 2026
        "gpt-5.5","gpt-5.4", "gpt-5.4-pro",
        "gpt-5.3-chat-latest", "gpt-5.3-codex", "gpt-5.3-codex-spark",
        "gpt-5.2", "gpt-5.2-pro", "gpt-5.2-chat-latest",
        "gpt-5-mini","gpt-5","gpt-5-nano", "gpt-5-chat-latest", "gpt-5-codex", "gpt-5-pro", "gpt-5-pro-2025-10-06",
        "gpt-4.1-nano",
        "gpt-4o-mini",  # Still on API, legacy
        "o3",        
        "gemini-3.1-pro-preview","gemini-3.1-flash-lite",
        "gemini-3-flash-preview", 
        "gemini-2.5-flash","gemini-2.5-flash-lite", "gemini-2.5-pro",
        "gemini-2.0-flash","gemini-2.0-flash-lite",
        
        # Anthropic Claude Models
        "claude-opus-4-7","claude-opus-4-6", "claude-opus-4-5-20251101", "claude-opus-4-1-20250805", "claude-opus-4-20250514", "claude-sonnet-4-6", 
        "claude-sonnet-4-5", "claude-sonnet-4-20250514", "claude-haiku-4-5-20251001",
        "claude-3-haiku-20240307",       
        
        # Grok Models
        "grok-4.3","grok-4.20-beta-0309-reasoning","grok-4.20-beta-0309-non-reasoning", "grok-4.20-multi-agent-beta-0309",
        "grok-4.20-multi-agent-experimental-beta-0304","grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning","grok-4-0709", "grok-4-fast",
        "grok-4-fast-reasoning", "grok-4-fast-non-reasoning",  "grok-4-fast-reasoning-latest", "grok-3", "grok-3-mini",
        
        # Local LLMs (LM Studio / Ollama / llama.cpp — OpenAI-compatible API)
        "lmstudio/default", "lmstudio/qwen2-vl-7b", "lmstudio/llama-3.2-vision-11b",
        "lmstudio/phi-3-vision", "lmstudio/minicpm-v-2.6", "lmstudio/internvl2-8b",
        "ollama/llama3.2-vision", "ollama/llava", "ollama/minicpm-v",
        "ollama/qwen2-vl", "ollama/internvl2", "ollama/granite3.2-vision",
        
  
    ]
