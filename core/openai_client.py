import json
from typing import List
from openai import OpenAI
from core.settings import OPENAI_API_KEY


def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing. Add it to Streamlit secrets.")
    return OpenAI(api_key=OPENAI_API_KEY)


def get_response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text

    try:
        parts = []
        for item in response.output:
            for content in getattr(item, "content", []):
                if getattr(content, "type", "") == "output_text":
                    parts.append(content.text)
        return "\n".join(parts).strip()
    except Exception:
        return str(response)


def safe_json_parse(text: str):
    try:
        return json.loads(text)
    except Exception:
        return {"raw_output": text}


def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    client = get_client()
    response = client.embeddings.create(model=model, input=texts)
    return [x.embedding for x in response.data]


def generate_text(prompt: str, model: str) -> str:
    client = get_client()
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return get_response_text(response)