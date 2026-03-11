# AI Metadata + Knowledge Graph Platform (Version 2.1)

Production-oriented Streamlit application for:

- AI metadata generation
- Multi-document corpus processing
- GraphRAG-style semantic retrieval
- Knowledge Graph visualization
- Auto ontology proposal
- RDF / JSON-LD export
- Digital Humanities / Library Science research workflows

## Features

- Upload multiple files (`pdf`, `docx`, `txt`, `csv`, `json`, `md`, `html`, `xml`, `py`)
- Automatic text extraction and chunking
- OpenAI embeddings for semantic retrieval
- FAISS vector backend by default
- Optional LanceDB backend
- Metadata bundle generation
- Ontology proposal generation
- Knowledge Graph visualization with PyVis
- Linked Data export in RDF Turtle and JSON-LD

## Repository structure

```text
metadata_ai_generator_v21/
├── app.py
├── requirements.txt
├── requirements-lancedb.txt
├── README.md
├── core/
│   ├── settings.py
│   ├── prompts.py
│   ├── openai_client.py
│   ├── document_loader.py
│   ├── chunking.py
│   ├── vector_store.py
│   ├── metadata_service.py
│   ├── ontology_service.py
│   ├── kg_service.py
│   ├── exporters.py
│   └── dashboard.py
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example