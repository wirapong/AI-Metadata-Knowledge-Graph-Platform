import os
import streamlit as st


def get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


APP_TITLE = get_secret("APP_TITLE", "AI Metadata + Knowledge Graph Platform")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
DEFAULT_EMBEDDING_MODEL = get_secret("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_LLM_MODEL = get_secret("DEFAULT_LLM_MODEL", "gpt-4.1-mini")
VECTOR_BACKEND = get_secret("VECTOR_BACKEND", "faiss").lower()

SUPPORTED_EXTENSIONS = [
    "pdf", "docx", "txt", "md", "csv", "json", "html", "htm", "xml", "py"
]