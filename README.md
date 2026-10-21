freshman-modules-rag/
├── data/
│   ├── raw/          # your PDFs (add to .gitignore)
│   └── processed/    # extracted text/chunks
├── src/
│   ├── ingest.py     # PDF loading and cleaning
│   ├── chunk.py
│   ├── embed.py
│   ├── retrieve.py
│   └── generate.py
├── eval/             # your question set and eval scripts
├── notebooks/        # extraction experiments
├── app.py            # UI later
├── requirements.txt
├── .gitignore
└── README.md