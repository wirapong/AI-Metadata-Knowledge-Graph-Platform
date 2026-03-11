import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

from core.settings import (
    APP_TITLE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    VECTOR_BACKEND,
)
from core.document_loader import extract_text_from_upload
from core.chunking import chunk_text
from core.vector_store import (
    FaissStore,
    LanceStore,
    build_corpus_df,
    semantic_search_faiss,
    semantic_search_lance,
)
from core.metadata_service import enrich_entities, generate_metadata_bundle
from core.ontology_service import generate_ontology_bundle
from core.kg_service import build_knowledge_graph, render_graph_html
from core.exporters import export_rdf_turtle, export_jsonld
from core.dashboard import corpus_stats, entity_frequency_df, plot_entity_frequency
from core.openai_client import generate_text
from core.prompts import ANSWER_PROMPT_TEMPLATE

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Version 2.1 — production-oriented GraphRAG-style platform for Digital Humanities / Library Science Research")

if "corpus_df" not in st.session_state:
    st.session_state.corpus_df = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "metadata_bundle" not in st.session_state:
    st.session_state.metadata_bundle = None
if "ontology_bundle" not in st.session_state:
    st.session_state.ontology_bundle = None
if "kg" not in st.session_state:
    st.session_state.kg = None

with st.sidebar:
    st.header("Settings")
    embedding_model = st.selectbox("Embedding model", [DEFAULT_EMBEDDING_MODEL], index=0)
    llm_model = st.selectbox("LLM model", [DEFAULT_LLM_MODEL], index=0)
    chunk_size = st.slider("Chunk size", 300, 1200, 700, 50)
    overlap = st.slider("Chunk overlap", 50, 300, 150, 25)
    top_k = st.slider("Top-K retrieval", 3, 15, 5, 1)
    vector_backend = st.selectbox(
        "Vector backend",
        ["faiss", "lancedb"] if VECTOR_BACKEND == "lancedb" else ["faiss", "lancedb"],
        index=0 if VECTOR_BACKEND == "faiss" else 1,
    )

st.subheader("1. Upload corpus")
uploaded_files = st.file_uploader(
    "Upload one or more files",
    type=["pdf", "docx", "txt", "md", "csv", "json", "html", "htm", "xml", "py"],
    accept_multiple_files=True,
)

col1, col2 = st.columns([1, 1])
build_clicked = col1.button("Build Corpus + Index", type="primary", use_container_width=True)
clear_clicked = col2.button("Clear Session", use_container_width=True)

if clear_clicked:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

if build_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    else:
        with st.spinner("Building corpus, vectors, metadata, ontology, and knowledge graph..."):
            corpus_df = build_corpus_df(
                uploaded_files=uploaded_files,
                embedding_model=embedding_model,
                chunker=lambda text: chunk_text(text, chunk_size=chunk_size, overlap=overlap),
                loader=extract_text_from_upload,
            )

            if corpus_df.empty:
                st.error("No usable text could be extracted from the uploaded files.")
                st.stop()

            corpus_df, top_entities = enrich_entities(corpus_df)

            if vector_backend == "faiss":
                store = FaissStore()
                store.build(corpus_df["embedding"].tolist())
            else:
                try:
                    store = LanceStore(uri="data/lancedb", table_name="chunks")
                    store.build(corpus_df)
                except Exception as e:
                    st.warning(f"LanceDB backend unavailable; falling back to FAISS. Reason: {e}")
                    store = FaissStore()
                    store.build(corpus_df["embedding"].tolist())
                    vector_backend = "faiss"

            metadata_bundle = generate_metadata_bundle(corpus_df, model=llm_model)
            ontology_bundle = generate_ontology_bundle(corpus_df, top_entities, model=llm_model)
            kg = build_knowledge_graph(corpus_df)

            st.session_state.corpus_df = corpus_df
            st.session_state.vector_store = store
            st.session_state.metadata_bundle = metadata_bundle
            st.session_state.ontology_bundle = ontology_bundle
            st.session_state.kg = kg

        st.success("Corpus indexed successfully.")

if st.session_state.corpus_df is not None:
    corpus_df = st.session_state.corpus_df
    store = st.session_state.vector_store
    metadata_bundle = st.session_state.metadata_bundle
    ontology_bundle = st.session_state.ontology_bundle
    kg = st.session_state.kg

    st.subheader("2. Corpus dashboard")
    stats = corpus_stats(corpus_df, kg)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents", stats["documents"])
    m2.metric("Chunks", stats["chunks"])
    m3.metric("Entities", stats["entities"])
    m4.metric("Relations", stats["relations"])

    with st.expander("Corpus preview", expanded=False):
        st.dataframe(
            corpus_df[["doc_id", "file_name", "chunk_id", "text", "entities"]],
            use_container_width=True,
            height=320,
        )

    st.subheader("3. Metadata bundle")
    st.json(metadata_bundle)

    meta_df = pd.DataFrame([metadata_bundle])
    st.download_button(
        "Download metadata CSV",
        data=meta_df.to_csv(index=False).encode("utf-8"),
        file_name="metadata_bundle.csv",
        mime="text/csv",
    )

    st.subheader("4. Ontology proposal")
    st.json(ontology_bundle)

    st.subheader("5. Knowledge Graph")
    html = render_graph_html(kg)
    components.html(html, height=680, scrolling=True)

    st.subheader("6. Semantic search dashboard")
    ef_df = entity_frequency_df(corpus_df, top_n=20)
    if not ef_df.empty:
        st.pyplot(plot_entity_frequency(ef_df))

    query = st.text_input("Ask a semantic question about the corpus")

    if query:
        if vector_backend == "faiss":
            results_df = semantic_search_faiss(
                query=query,
                corpus_df=corpus_df,
                store=store,
                embedding_model=embedding_model,
                top_k=top_k,
            )
        else:
            results_df = semantic_search_lance(
                query=query,
                corpus_df=corpus_df,
                store=store,
                embedding_model=embedding_model,
                top_k=top_k,
            )

        if results_df.empty:
            st.warning("No relevant chunks found.")
        else:
            st.markdown("**Top retrieved chunks**")
            st.dataframe(
                results_df[["score", "file_name", "chunk_id", "entities", "text"]],
                use_container_width=True,
                height=320,
            )

            context = "\n\n".join(results_df["text"].tolist())[:14000]
            prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, query=query)
            answer = generate_text(prompt, llm_model)

            st.markdown("**Answer**")
            st.write(answer)

    st.subheader("7. Linked Data export")
    rdf_ttl = export_rdf_turtle(corpus_df, metadata_bundle, ontology_bundle)
    jsonld_text = export_jsonld(corpus_df, metadata_bundle)

    st.download_button(
        "Download RDF Turtle",
        data=rdf_ttl.encode("utf-8"),
        file_name="corpus_graph.ttl",
        mime="text/turtle",
    )
    st.download_button(
        "Download JSON-LD",
        data=jsonld_text.encode("utf-8"),
        file_name="corpus_graph.jsonld",
        mime="application/ld+json",
    )
else:
    st.info("Upload files and click 'Build Corpus + Index' to start.")