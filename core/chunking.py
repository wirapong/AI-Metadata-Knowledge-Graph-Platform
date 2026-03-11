from typing import List


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 150) -> List[str]:
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks