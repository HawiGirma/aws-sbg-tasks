# Week 02: Mini Semantic Search

## Goal
Build a small semantic-search app for frequently asked questions (FAQs). It loads a CSV or JSON dataset with Pandas, cleans and chunks text, creates embeddings, stores them in ChromaDB, and returns the top three semantically similar matches.

## Project files
- `semantic_search.py`: reusable command-line application.
- `semantic_search.ipynb`: notebook walkthrough with explanations, code, and example queries.
- `faqs.csv`: sample FAQ dataset.
- `requirements.txt`: Python dependencies.
- `README.md`: setup, usage, and example queries.
- `chroma_db/`: created locally when the script runs; do not commit this generated database unless your instructor specifically asks for it.

## Requirements
- Python 3.10 or newer recommended.
- Internet access on the first run to install packages and download the embedding model.
- A few GB of free disk space may be needed for Python packages and the model cache.

## Setup using a virtual environment

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python semantic_search.py --data faqs.csv --query "I forgot my login details"
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python semantic_search.py --data faqs.csv --query "I forgot my login details"
```

## Example queries
```bash
python semantic_search.py --data faqs.csv --query "I cannot remember how to sign in"
python semantic_search.py --data faqs.csv --query "Where can I find my grades?"
python semantic_search.py --data faqs.csv --query "My tuition payment is missing" --top-k 3
python semantic_search.py --data faqs.csv --query "I need academic records" --category academics
```

Use `--rebuild` to recreate the stored collection from the current input dataset:
```bash
python semantic_search.py --data faqs.csv --rebuild --query "How can I register?"
```

## Data format
CSV or JSON should contain `question` and `answer` columns/keys. `id` and `category` are optional. The app generates IDs and a `general` category if they are absent.

## How it works
1. **Load:** Pandas reads the CSV or JSON.
2. **Clean:** whitespace is normalized; rows with blank questions/answers and duplicate question-answer pairs are removed.
3. **Chunk:** longer question-answer text is split into overlapping word chunks.
4. **Embed:** `sentence-transformers/all-MiniLM-L6-v2` maps chunks and queries into vectors.
5. **Store:** ChromaDB persists vectors, text, and metadata locally.
6. **Retrieve:** ChromaDB returns the nearest matches. Distance is a ranking measure, not a probability.

## Google Colab
Open `semantic_search.ipynb` in Colab, upload `faqs.csv` when prompted (or upload both files), run all cells from top to bottom, then download the executed notebook as `.ipynb`. Model downloads can take time on the first run.

## GitHub submission
Commit the source, notebook, README, requirements file, and sample dataset. Avoid committing `.venv/`, model caches, or the generated `chroma_db/` directory.

Suggested `.gitignore` entries:
```gitignore
.venv/
__pycache__/
*.py[cod]
chroma_db/
.ipynb_checkpoints/
```

## Limitations
This is an educational prototype using a small local dataset. Similarity results should be reviewed for relevance. Do not use it as the sole source for high-stakes decisions.
