import json
import re
from collections import Counter

from core.openai_client import generate_text, safe_json_parse
from core.prompts import METADATA_PROMPT


def simple_entity_extraction(text: str):
    entities = set()

    for x in re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
        entities.add(x.strip())

    for x in re.findall(r"\b[A-Z]{2,}\b", text):
        entities.add(x.strip())

    for x in re.findall(r"\b(?:18|19|20)\d{2}\b", text):
        entities.add(x.strip())

    return sorted(list(entities))[:30]


def enrich_entities(corpus_df):
    all_entities = []
    entity_col = []

    for _, row in corpus_df.iterrows():
        ents = simple_entity_extraction(row["text"])
        entity_col.append(ents)
        all_entities.extend(ents)

    corpus_df = corpus_df.copy()
    corpus_df["entities"] = entity_col
    top_entities = [x for x, _ in Counter(all_entities).most_common(100)]
    return corpus_df, top_entities


def generate_metadata_bundle(corpus_df, model: str):
    sample = "\n\n".join(corpus_df["text"].head(20).tolist())[:12000]
    prompt = f"{METADATA_PROMPT}\n\nCORPUS:\n{sample}"
    text = generate_text(prompt, model)
    return safe_json_parse(text)