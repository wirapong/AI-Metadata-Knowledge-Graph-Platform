import json
import networkx as nx
from pyvis.network import Network


def build_knowledge_graph(corpus_df):
    G = nx.Graph()

    for _, row in corpus_df.iterrows():
        chunk_node = row["chunk_id"]
        G.add_node(
            chunk_node,
            label=row["chunk_id"],
            node_type="chunk",
            file_name=row["file_name"],
        )

        entities = row.get("entities", []) or []

        for ent in entities:
            G.add_node(ent, label=ent, node_type="entity")
            G.add_edge(chunk_node, ent, relation="mentions", weight=1)

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a, b = entities[i], entities[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] = G[a][b].get("weight", 1) + 1
                    G[a][b]["relation"] = "co_occurs"
                else:
                    G.add_edge(a, b, relation="co_occurs", weight=1)

    return G


def render_graph_html(G, max_nodes: int = 250):
    net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#222")
    net.barnes_hut()

    nodes = list(G.nodes(data=True))[:max_nodes]
    allowed = {n for n, _ in nodes}

    for node, attrs in nodes:
        node_type = attrs.get("node_type", "entity")
        color = "#f59e0b" if node_type == "chunk" else "#3b82f6"
        size = 16 if node_type == "chunk" else 20
        title = json.dumps(attrs, ensure_ascii=False, indent=2)
        net.add_node(node, label=str(node)[:60], title=title, color=color, size=size)

    for s, t, attrs in G.edges(data=True):
        if s in allowed and t in allowed:
            net.add_edge(
                s, t,
                label=attrs.get("relation", ""),
                width=min(attrs.get("weight", 1), 8)
            )

    return net.generate_html()