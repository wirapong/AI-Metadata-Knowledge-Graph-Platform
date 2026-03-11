from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


def corpus_stats(corpus_df, kg):
    entity_nodes = [n for n, d in kg.nodes(data=True) if d.get("node_type") == "entity"]
    return {
        "documents": int(corpus_df["doc_id"].nunique()),
        "chunks": int(len(corpus_df)),
        "entities": int(len(entity_nodes)),
        "relations": int(len(kg.edges())),
    }


def entity_frequency_df(corpus_df, top_n=20):
    all_entities = []
    for _, row in corpus_df.iterrows():
        all_entities.extend(row.get("entities", []))
    freq = Counter(all_entities)
    rows = [{"entity": k, "count": v} for k, v in freq.most_common(top_n)]
    return pd.DataFrame(rows)


def plot_entity_frequency(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["entity"], df["count"])
    ax.set_title("Top Entities")
    ax.set_xlabel("Entity")
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig