"""Configuration: env loading + LLM factory."""
import os
from dotenv import load_dotenv


def load_env():
    """Load .env file if present."""
    load_dotenv()


def get_llm():
    """Return configured DeepSeek chat model."""
    from langchain_deepseek import ChatDeepSeek

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set. Copy .env.example to .env and fill in your key.")
    return ChatDeepSeek(model="deepseek-chat", api_key=api_key)
