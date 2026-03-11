import json
import pandas as pd
from docx import Document
from pypdf import PdfReader


def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()

    if name.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs]).strip()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df.to_csv(index=False)

    if name.endswith(".json"):
        data = json.load(uploaded_file)
        return json.dumps(data, ensure_ascii=False, indent=2)

    raw = uploaded_file.read()
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")