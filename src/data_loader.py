"""
RAM-friendly loader for datathonFINAL.parquet.

Key facts about the dataset:
- 5,004,813 rows, 9 columns, ~1.54 GB compressed
- 1,597,838 unique author_hash values
- 2,147,605 posts (~43%) have empty author_hash (no author info available)
- 14,962 posts have SHA1("") hash — also no real author
- Date range: 2024-11-19 to 2024-11-24 (5 days)
- Platforms: x.com (57%), reddit.com (27%), bsky.app (10%), youtube.com (3%)
"""

import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "datathonFINAL.parquet"

# SHA1 of empty string — treat same as empty author_hash
_SHA1_EMPTY = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

DTYPE_MAP = {
    "sentiment": "float32",
    "language": "category",
    "main_emotion": "category",
    "primary_theme": "category",
}


def schema_info() -> dict:
    """Return schema and row count without loading data."""
    f = pq.ParquetFile(DATA_PATH)
    meta = f.metadata
    return {
        "schema": f.schema_arrow,
        "num_rows": meta.num_rows,
        "num_row_groups": meta.num_row_groups,
        "compressed_bytes": sum(
            meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)
        ),
    }


def iter_batches(columns: list[str] | None = None, batch_size: int = 100_000):
    """Yield pandas DataFrames in batches. Never loads the full file at once."""
    f = pq.ParquetFile(DATA_PATH)
    for batch in f.iter_batches(batch_size=batch_size, columns=columns):
        df = batch.to_pandas()
        df = _optimize_dtypes(df)
        yield df


def load_sample(frac: float = 0.1) -> pd.DataFrame:
    """
    Load a sample by reading every Nth row group.
    frac=0.1 → ~500k rows, fits comfortably in RAM.
    """
    f = pq.ParquetFile(DATA_PATH)
    n_groups = f.metadata.num_row_groups
    step = max(1, round(1 / frac))

    chunks = []
    for i in range(0, n_groups, step):
        tbl = f.read_row_group(i)
        chunks.append(tbl.to_pandas())

    df = pd.concat(chunks, ignore_index=True)
    df = _optimize_dtypes(df)
    df = _enrich(df)
    return df


def load_known_authors(min_posts: int = 3) -> pd.DataFrame:
    """
    Load only posts that have a real author_hash and come from authors
    with at least `min_posts` posts. Streams the file to stay RAM-friendly.
    """
    from collections import Counter

    # Pass 1: count posts per real author
    counts: Counter = Counter()
    for batch in iter_batches(columns=["author_hash"]):
        real = batch[~batch["author_hash"].isin(["", _SHA1_EMPTY])]
        counts.update(real["author_hash"].tolist())

    valid = {h for h, c in counts.items() if c >= min_posts}

    # Pass 2: collect their rows
    chunks = []
    for batch in iter_batches():
        mask = batch["author_hash"].isin(valid)
        chunk = batch[mask].copy()
        if not chunk.empty:
            chunk = _enrich(chunk)
            chunks.append(chunk)

    if chunks:
        df = pd.concat(chunks, ignore_index=True)
        df = _optimize_dtypes(df)
    else:
        df = pd.DataFrame()
    return df


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in DTYPE_MAP.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used across notebooks and src/."""
    if "date" in df.columns and "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    if "url" in df.columns and "domain" not in df.columns:
        df["domain"] = df["url"].apply(
            lambda x: re.sub(r"^(https?://)?(www\.)?", "", str(x)).split("/")[0]
        ).astype("category")

    if "author_hash" in df.columns and "has_author" not in df.columns:
        df["has_author"] = ~df["author_hash"].isin(["", _SHA1_EMPTY])

    return df
