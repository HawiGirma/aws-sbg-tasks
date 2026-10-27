"""
Week 02: Mini Semantic Search

Load FAQ data from CSV or JSON, clean it, chunk FAQ text, embed the chunks,
store them in a persistent ChromaDB collection, and retrieve the top matches.

Usage:
    python semantic_search.py --data faqs.csv --query "I forgot my login details"
    python semantic_search.py --data faqs.csv --query "How do I pay tuition?" --top-k 3
    python semantic_search.py --data faqs.csv --rebuild
"""

import argparse
import json
import re
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "aastu_week2_faqs"
DEFAULT_DB_PATH = Path(__file__).with_name("chroma_db")
REQUIRED_COLUMNS = {"question", "answer"}


def load_faqs(file_path: str) -> pd.DataFrame:
    """Load FAQs from a CSV or JSON file using Pandas."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".json":
        frame = pd.read_json(path)
    else:
        raise ValueError("Supported data formats are .csv and .json")

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if "id" not in frame.columns:
        frame["id"] = [f"faq_{i + 1:03d}" for i in range(len(frame))]
    if "category" not in frame.columns:
        frame["category"] = "general"

    return frame


def clean_faqs(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize text, remove unusable rows and duplicate question/answer pairs."""
    cleaned = frame.copy()
    for column in ("id", "question", "answer", "category"):
        cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()

    cleaned["question"] = cleaned["question"].map(
        lambda value: re.sub(r"\s+", " ", value)
    )
    cleaned["answer"] = cleaned["answer"].map(
        lambda value: re.sub(r"\s+", " ", value)
    )
    cleaned["category"] = cleaned["category"].replace("", "general")

    cleaned = cleaned[
        (cleaned["question"] != "") & (cleaned["answer"] != "")
    ].copy()
    cleaned = cleaned.drop_duplicates(
        subset=["question", "answer"], keep="first"
    ).reset_index(drop=True)

    # Chroma IDs must be non-empty and unique.
    used_ids = set()
    unique_ids = []
    for index, value in enumerate(cleaned["id"], start=1):
        candidate = value or f"faq_{index:03d}"
        original = candidate
        suffix = 2
        while candidate in used_ids:
            candidate = f"{original}_{suffix}"
            suffix += 1
        used_ids.add(candidate)
        unique_ids.append(candidate)
    cleaned["id"] = unique_ids
    return cleaned


def chunk_text(text: str, chunk_size: int = 45, overlap: int = 8) -> list[str]:
    """Split text into overlapping word chunks, preserving context at boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), step)
    ]


def prepare_documents(frame: pd.DataFrame, chunk_size: int = 45, overlap: int = 8):
    """Create searchable text chunks and metadata for ChromaDB."""
    documents, ids, metadatas = [], [], []
    for _, row in frame.iterrows():
        combined = f"Question: {row['question']} Answer: {row['answer']}"
        chunks = chunk_text(combined, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"{row['id']}_chunk_{chunk_index + 1}")
            metadatas.append({
                "faq_id": str(row["id"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "category": str(row["category"]),
                "chunk_index": int(chunk_index),
            })
    return documents, ids, metadatas


def build_index(
    data_path: str,
    db_path: Path = DEFAULT_DB_PATH,
    model_name: str = DEFAULT_MODEL,
    collection_name: str = DEFAULT_COLLECTION,
    rebuild: bool = False,
):
    """Load, clean, embed and persist FAQ chunks in ChromaDB."""
    raw = load_faqs(data_path)
    faqs = clean_faqs(raw)
    if faqs.empty:
        raise ValueError("No valid FAQs remain after cleaning.")

    documents, ids, metadatas = prepare_documents(faqs)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        documents, normalize_embeddings=True, show_progress_bar=False
    ).tolist()

    client = chromadb.PersistentClient(path=str(db_path))
    existing = {collection.name for collection in client.list_collections()}
    if collection_name in existing and rebuild:
        client.delete_collection(collection_name)

    if rebuild or collection_name not in existing:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Week 2 FAQ semantic search demo"},
        )
        # Upsert supports rerunning the script without duplicate records.
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
    else:
        collection = client.get_collection(collection_name)
        # Refresh records from current input data.
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    return model, collection, faqs, len(documents)


def search_faqs(query: str, model, collection, top_k: int = 3, category=None):
    """Return up to top_k semantically similar FAQ chunks."""
    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    total = collection.count()
    if total == 0:
        return []

    query_embedding = model.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    )[0].tolist()
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, total),
        "include": ["documents", "metadatas", "distances"],
    }
    if category:
        kwargs["where"] = {"category": category}

    result = collection.query(**kwargs)
    matches = []
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for doc_id, document, metadata, distance in zip(ids, docs, metas, distances):
        matches.append({
            "id": doc_id,
            "question": metadata.get("question", ""),
            "answer": metadata.get("answer", ""),
            "category": metadata.get("category", "general"),
            "chunk": document,
            "distance": float(distance),
        })
    return matches


def main():
    parser = argparse.ArgumentParser(description="Search FAQs by meaning.")
    parser.add_argument("--data", default=str(Path(__file__).with_name("faqs.csv")))
    parser.add_argument("--query", default="I forgot my login details")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--category", default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    model, collection, faqs, chunk_count = build_index(
        args.data, rebuild=args.rebuild
    )
    print(f"FAQs loaded after cleaning: {len(faqs)}")
    print(f"Text chunks indexed: {chunk_count}")
    print(f"Query: {args.query}")
    print("-" * 72)

    results = search_faqs(
        args.query, model, collection, top_k=args.top_k, category=args.category
    )
    if not results:
        print("No matches found.")
        return

    for rank, item in enumerate(results, start=1):
        # Chroma returns cosine distance for cosine-configured collections;
        # lower distance indicates a closer match, not a probability.
        print(f"{rank}. {item['question']}")
        print(f"   Category: {item['category']}")
        print(f"   Answer: {item['answer']}")
        print(f"   Distance: {item['distance']:.4f}")
        print()


if __name__ == "__main__":
    main()
