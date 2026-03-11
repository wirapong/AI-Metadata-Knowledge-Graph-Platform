import os
import pickle
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd

from core.openai_client import embed_texts


class FaissStore:
    def __init__(self):
        self.index = None
        self.matrix = None

    def build(self, embeddings: List[List[float]]):
        self.matrix = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(self.matrix)
        self.index = faiss.IndexFlatIP(self.matrix.shape[1])
        self.index.add(self.matrix)

    def search(self, query_embedding: List[float], top_k: int = 5):
        q = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)
        return scores[0].tolist(), idxs[0].tolist()

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "matrix.pkl"), "wb") as f:
            pickle.dump(self.matrix, f)

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "matrix.pkl"), "rb") as f:
            self.matrix = pickle.load(f)


class LanceStore:
    def __init__(self, uri: str = "data/lancedb", table_name: str = "chunks"):
        self.uri = uri
        self.table_name = table_name
        self.db = None
        self.table = None

    def connect(self):
        import lancedb
        self.db = lancedb.connect(self.uri)

    def build(self, df: pd.DataFrame):
        self.connect()
        records = []
        for _, row in df.iterrows():
            records.append({
                "id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "file_name": row["file_name"],
                "text": row["text"],
                "entities": row["entities"],
                "vector": row["embedding"],
            })

        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)

        self.table = self.db.create_table(self.table_name, data=records)

    def search(self, query_embedding: List[float], top_k: int = 5):
        result = self.table.search(query_embedding).limit(top_k).to_list()
        scores = [float(x.get("_distance", 0.0)) for x in result]
        ids = [x["id"] for x in result]
        return scores, ids


def build_corpus_df(uploaded_files, embedding_model: str, chunker, loader):
    rows = []

    for doc_num, f in enumerate(uploaded_files, start=1):
        text = loader(f)
        chunks = chunker(text)

        if not chunks:
            continue

        embeddings = embed_texts(chunks, embedding_model)

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings), start=1):
            rows.append({
                "doc_id": f"doc_{doc_num}",
                "file_name": f.name,
                "chunk_id": f"{f.name}_chunk_{i}",
                "text": chunk,
                "embedding": emb,
            })

    return pd.DataFrame(rows)


def semantic_search_faiss(query: str, corpus_df: pd.DataFrame, store: FaissStore, embedding_model: str, top_k: int = 5):
    query_emb = embed_texts([query], embedding_model)[0]
    scores, idxs = store.search(query_emb, top_k=top_k)

    rows = []
    for score, idx in zip(scores, idxs):
        if idx < 0 or idx >= len(corpus_df):
            continue
        record = corpus_df.iloc[idx].to_dict()
        record["score"] = round(float(score), 4)
        rows.append(record)

    return pd.DataFrame(rows)


def semantic_search_lance(query: str, corpus_df: pd.DataFrame, store: LanceStore, embedding_model: str, top_k: int = 5):
    query_emb = embed_texts([query], embedding_model)[0]
    scores, ids = store.search(query_emb, top_k=top_k)

    id_to_row = {row["chunk_id"]: row for _, row in corpus_df.iterrows()}
    rows = []
    for score, chunk_id in zip(scores, ids):
        row = id_to_row.get(chunk_id)
        if row is None:
            continue
        item = dict(row)
        item["score"] = round(float(score), 4)
        rows.append(item)

    return pd.DataFrame(rows)