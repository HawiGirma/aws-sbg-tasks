"""
Week 02: Mini Semantic Search Application

Loads FAQ data from CSV or JSON using Pandas, cleans whitespace and duplicates,
chunks text with boundary overlap, generates dense embeddings with
`sentence-transformers/all-MiniLM-L6-v2`, stores vectors and metadata in ChromaDB
configured with an explicit cosine distance metric, and performs semantic retrieval.

Usage:
    python semantic_search.py --data faqs.csv --query "I forgot my password"
    python semantic_search.py --data faqs.csv --query "Where are my grades?" --top-k 3
    python semantic_search.py --data faqs.csv --query "transcript" --category academics
    python semantic_search.py --data faqs.csv --rebuild
    python semantic_search.py --examples
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "aastu_week2_faqs"
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "chroma_db"
REQUIRED_COLUMNS = {"question", "answer"}


def load_faqs(file_path: str | Path) -> pd.DataFrame:
    """Load FAQ records from a CSV or JSON file using Pandas.

    Args:
        file_path: Path to .csv or .json dataset file.

    Returns:
        Pandas DataFrame containing at least 'question' and 'answer' columns.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file format is unsupported, empty, or missing required fields.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix == ".json":
            frame = pd.read_json(path)
        else:
            raise ValueError(
                f"Unsupported file format '{suffix}'. Supported formats are .csv and .json"
            )
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset file is empty: {path}")
    except (ValueError, json.JSONDecodeError) as err:
        raise ValueError(f"Failed to parse dataset file '{path}': {err}")

    if frame.empty:
        raise ValueError(f"Dataset contains no records: {path}")

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(frame.columns)}"
        )

    # Supply default column values if optional columns are absent
    if "id" not in frame.columns:
        frame["id"] = [f"faq_{i + 1:03d}" for i in range(len(frame))]
    if "category" not in frame.columns:
        frame["category"] = "general"

    return frame


def clean_faqs(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean whitespace, handle missing values, and remove duplicate FAQ records.

    Normalizes consecutive whitespace in question and answer texts, trims
    leading/trailing spaces, drops records with missing or empty question/answer,
    and removes duplicate question-answer pairs.

    Args:
        frame: Raw FAQ DataFrame.

    Returns:
        Cleaned FAQ DataFrame with validated unique IDs.

    Raises:
        ValueError: If no valid records remain after cleaning.
    """
    cleaned = frame.copy()

    for col in ("id", "question", "answer", "category"):
        if col not in cleaned.columns:
            cleaned[col] = "general" if col == "category" else ""
        cleaned[col] = cleaned[col].fillna("").astype(str).str.strip()

    # Normalize internal whitespace
    cleaned["question"] = cleaned["question"].map(
        lambda val: re.sub(r"\s+", " ", val)
    )
    cleaned["answer"] = cleaned["answer"].map(
        lambda val: re.sub(r"\s+", " ", val)
    )
    cleaned["category"] = cleaned["category"].replace("", "general")

    # Drop rows where question or answer is blank
    cleaned = cleaned[
        (cleaned["question"] != "") & (cleaned["answer"] != "")
    ].copy()

    # Drop duplicate question and answer pairs
    cleaned = cleaned.drop_duplicates(
        subset=["question", "answer"], keep="first"
    ).reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("No valid FAQ records remain after cleaning.")

    # Ensure all IDs are non-empty and unique for ChromaDB storage
    used_ids: set[str] = set()
    unique_ids: list[str] = []
    for index, val in enumerate(cleaned["id"], start=1):
        candidate = val or f"faq_{index:03d}"
        base = candidate
        counter = 2
        while candidate in used_ids:
            candidate = f"{base}_{counter}"
            counter += 1
        used_ids.add(candidate)
        unique_ids.append(candidate)
    cleaned["id"] = unique_ids

    return cleaned


def chunk_text(text: str, chunk_size: int = 45, overlap: int = 8) -> list[str]:
    """Split text into overlapping word chunks, preserving context at boundaries.

    Args:
        text: Input string to chunk.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words to overlap between consecutive chunks.

    Returns:
        List of chunk strings.

    Raises:
        ValueError: If chunk_size <= 0 or overlap is invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and strictly smaller than chunk_size")

    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def prepare_documents(
    frame: pd.DataFrame, chunk_size: int = 45, overlap: int = 8
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Convert cleaned FAQ DataFrame rows into chunked documents, IDs, and metadata.

    Args:
        frame: Cleaned FAQ DataFrame.
        chunk_size: Words per chunk.
        overlap: Words overlapping between chunks.

    Returns:
        Tuple of (documents, ids, metadatas).
    """
    documents: list[str] = []
    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for _, row in frame.iterrows():
        combined = f"Question: {row['question']} Answer: {row['answer']}"
        chunks = chunk_text(combined, chunk_size=chunk_size, overlap=overlap)
        for chunk_idx, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"{row['id']}_chunk_{chunk_idx + 1}")
            metadatas.append({
                "faq_id": str(row["id"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "category": str(row["category"]),
                "chunk_index": int(chunk_idx),
            })

    return documents, ids, metadatas


def get_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load the SentenceTransformer embedding model."""
    return SentenceTransformer(model_name)


def build_index(
    data_path: str | Path,
    db_path: Path = DEFAULT_DB_PATH,
    model_name: str = DEFAULT_MODEL,
    collection_name: str = DEFAULT_COLLECTION,
    rebuild: bool = False,
) -> tuple[SentenceTransformer, Any, pd.DataFrame, int]:
    """Load, clean, embed, and store FAQ chunks in persistent ChromaDB.

    Explicitly sets ChromaDB's distance metric to cosine ('hnsw:space': 'cosine').
    Uses collection upsert so repeated runs without the rebuild flag do not create
    duplicate entries.

    Args:
        data_path: Path to CSV or JSON FAQ dataset.
        db_path: Path to ChromaDB storage directory.
        model_name: Name of the SentenceTransformer model.
        collection_name: ChromaDB collection name.
        rebuild: If True, deletes any existing collection before indexing.

    Returns:
        Tuple of (model, collection, cleaned_df, total_chunks_indexed).
    """
    raw_df = load_faqs(data_path)
    cleaned_df = clean_faqs(raw_df)
    documents, ids, metadatas = prepare_documents(cleaned_df)

    model = get_embedding_model(model_name)
    embeddings = model.encode(
        documents, normalize_embeddings=True, show_progress_bar=False
    ).tolist()

    client = chromadb.PersistentClient(path=str(db_path))
    existing_collections = {c.name for c in client.list_collections()}

    if rebuild and collection_name in existing_collections:
        client.delete_collection(collection_name)
        existing_collections.remove(collection_name)

    # Explicitly configure ChromaDB's distance metric to cosine similarity space
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "description": "Week 2 FAQ mini semantic search collection",
        },
    )

    # Upsert ensures repeated runs with the same IDs do not create duplicates
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return model, collection, cleaned_df, len(documents)


def search_faqs(
    query: str,
    model: SentenceTransformer,
    collection: Any,
    top_k: int = 3,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k semantically similar FAQ chunks for a natural-language query.

    Args:
        query: User search query string.
        model: SentenceTransformer instance.
        collection: ChromaDB collection instance.
        top_k: Number of nearest matches to return.
        category: Optional category filter.

    Returns:
        List of match dictionaries containing matched question, answer,
        category, chunk, distance, and similarity score.

    Raises:
        ValueError: If query is blank or top_k < 1.
    """
    query_str = query.strip()
    if not query_str:
        raise ValueError("Search query cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    total_chunks = collection.count()
    if total_chunks == 0:
        return []

    # Encode query with the same embedding model and normalization
    query_vector = model.encode(
        [query_str], normalize_embeddings=True, show_progress_bar=False
    )[0].tolist()

    query_args: dict[str, Any] = {
        "query_embeddings": [query_vector],
        "n_results": min(top_k, total_chunks),
        "include": ["documents", "metadatas", "distances"],
    }
    if category:
        query_args["where"] = {"category": category.strip().lower()}

    results = collection.query(**query_args)
    matches: list[dict[str, Any]] = []

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc_id, document, metadata, distance in zip(ids, docs, metas, distances):
        dist_float = float(distance)
        # For normalized cosine distance: distance = 1 - cosine_similarity
        similarity = 1.0 - dist_float
        matches.append({
            "id": doc_id,
            "question": metadata.get("question", ""),
            "answer": metadata.get("answer", ""),
            "category": metadata.get("category", "general"),
            "chunk": document,
            "distance": dist_float,
            "similarity": similarity,
        })

    return matches


def format_search_results(query: str, results: list[dict[str, Any]]) -> str:
    """Format matched FAQ items into readable text output."""
    lines = [f"Query: {query}", "=" * 72]
    if not results:
        lines.append("No matching FAQ records found.")
        return "\n".join(lines)

    for rank, item in enumerate(results, start=1):
        lines.append(f"{rank}. {item['question']}")
        lines.append(f"   Category:   {item['category']}")
        lines.append(f"   Answer:     {item['answer']}")
        lines.append(
            f"   Distance:   {item['distance']:.4f} (Cosine distance: lower indicates higher semantic similarity)"
        )
        lines.append(
            f"   Similarity: {item['similarity']:.4f} (Cosine similarity ranking measure; not a probability)"
        )
        lines.append("")

    return "\n".join(lines)


def run_example_queries(model: SentenceTransformer, collection: Any, top_k: int = 3) -> None:
    """Execute multiple realistic example queries covering different categories."""
    examples = [
        ("I cannot remember how to sign in to my account", None),
        ("Where can I find my examination grades?", "academics"),
        ("My tuition payment is missing from the portal", None),
        ("The website is completely down and not opening", "support"),
        ("Can students check out books from the university library?", None),
    ]

    print("\n" + "#" * 72)
    print("RUNNING EXAMPLE QUERIES DEMONSTRATION")
    print("#" * 72 + "\n")

    for query_text, cat in examples:
        filter_str = f" [Category filter: {cat}]" if cat else ""
        print(f"--- Example Query: '{query_text}'{filter_str} ---")
        matches = search_faqs(query_text, model, collection, top_k=top_k, category=cat)
        print(format_search_results(query_text, matches))


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Week 02 Mini Semantic Search for FAQ records using SentenceTransformers and ChromaDB."
    )
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parent / "faqs.csv"),
        help="Path to CSV or JSON dataset (default: faqs.csv).",
    )
    parser.add_argument(
        "--query",
        default="I forgot my login details",
        help="Natural language search query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of nearest matches to retrieve (default: 3).",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Optional category filter (e.g., account, academics, payments, support).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing ChromaDB collection and rebuild index from source data.",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run 5 diverse example queries demonstrating semantic retrieval.",
    )
    args = parser.parse_args()

    try:
        model, collection, faqs, chunk_count = build_index(
            data_path=args.data, rebuild=args.rebuild
        )
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    except ValueError as err:
        print(f"Data Error: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"Indexing Error: {err}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded FAQs after cleaning: {len(faqs)}")
    print(f"Indexed text chunks:        {chunk_count}")
    print(f"ChromaDB distance metric:   cosine")

    if args.examples:
        run_example_queries(model, collection, top_k=args.top_k)
        return

    try:
        matches = search_faqs(
            query=args.query,
            model=model,
            collection=collection,
            top_k=args.top_k,
            category=args.category,
        )
        print("\n" + format_search_results(args.query, matches))
        print(
            "Note: Distance scores are ranking metrics (lower distance = closer semantic match), not probabilities."
        )
    except ValueError as err:
        print(f"Query Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
