```
                         ┌─────────────────────────┐
                         │      STREAMLIT UI       │
                         │  (upload, chat, sources) │
                         └───────────┬──────────────┘
                                     │
                ┌────────────────────┴────────────────────┐
                │                                          │
        INGESTION PIPELINE                         QUERY PIPELINE
        (runs once per doc)                    (runs once per question)
                │                                          │
   ┌────────────▼────────────┐              ┌──────────────▼──────────────┐
   │ 1. PDF Loader (PyPDF)    │              │ 1. Embed the user question   │
   │ 2. Text Splitter         │              │    (same embedding model)    │
   │ 3. Embedding Model       │              │ 2. FAISS similarity search    │
   │    (Sentence Transformers)│             │    → top-k chunks             │
   │ 4. Store vectors + text  │              │ 3. Build prompt = question   │
   │    + metadata in FAISS   │              │    + retrieved chunks         │
   └────────────┬────────────┘              │ 4. Send prompt to Ollama LLM │
                │                            │ 5. LLM generates grounded    │
                ▼                            │    answer                    │
        ┌───────────────┐                    │ 6. Display answer + sources  │
        │  FAISS Index   │◄───────────────────    (doc name + page number)  │
        │  (vector store)│                    └───────────────────────────┘
        └───────────────┘
```