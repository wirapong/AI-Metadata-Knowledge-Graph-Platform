import json
from core.openai_client import generate_text, safe_json_parse
from core.prompts import ONTOLOGY_PROMPT


def generate_ontology_bundle(corpus_df, top_entities, model: str):
    sample = "\n\n".join(corpus_df["text"].head(12).tolist())[:9000]
    entities_json = json.dumps(top_entities[:200], ensure_ascii=False)

    prompt = f"""
{ONTOLOGY_PROMPT}

ENTITIES:
{entities_json}

CORPUS SAMPLE:
{sample}
"""
    text = generate_text(prompt, model)
    return safe_json_parse(text)