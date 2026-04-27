"""
HDBSCAN post-level clustering for manipulation map.

Strategy: fit on a stratified 10% sample (~500k posts), then use
approximate_predict for the remaining 90%. All 5M posts get a cluster label.
This is statistically valid — 500k is far more than enough to learn the
density structure; approximate_predict assigns unseen points via the learned
tree without re-running the full MST.

Output:
  outputs/post_clusters.parquet   — her post icin cluster_id + organic_score
  outputs/cluster_summary.csv     — her kume icin dil/platform dagilimi + profil
"""

import sys
import threading
import time
from pathlib import Path

import hdbscan
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).parent.parent))

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models" / "task_1"
OUTPUTS_DIR   = Path(__file__).parent.parent / "outputs"

CLUSTER_FEATURES = [
    "text_len",
    "kw_count",
    "kw_density",
    "sentiment",
    "sentiment_extreme",
    "is_duplicate",
    "cross_author_dup_count",
    "kw_fingerprint_shared",
    "has_author",
]

LOG1P_COLS = ["text_len", "kw_count", "cross_author_dup_count", "kw_fingerprint_shared"]

# Fit fraction: 10% of all posts (~500k for 5M dataset)
SAMPLE_FRAC = 0.10
RANDOM_SEED = 42


def _elapsed_ticker(stop_event: threading.Event, start: float):
    while not stop_event.wait(1):
        elapsed = int(time.time() - start)
        print(f"\r   ... calisıyor ({elapsed}s)", end="", flush=True)
    print()


def _load_data() -> pd.DataFrame:
    needed_cols = CLUSTER_FEATURES + ["language", "domain", "author_hash"]
    print("Veri yukleniyor...")
    df = pd.read_parquet(
        PROCESSED_DIR / "post_features.parquet",
        columns=needed_cols,
    )
    scores = pd.read_parquet(
        OUTPUTS_DIR / "post_scores.parquet",
        columns=["author_hash", "organic_score"],
    )
    df["organic_score"] = scores["organic_score"].values
    print(f"   Toplam post: {len(df):,}")
    return df


def _build_feature_matrix(df: pd.DataFrame):
    X = df[CLUSTER_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).copy()
    for col in LOG1P_COLS:
        X[col] = np.log1p(X[col])
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X.values)
    return X_scaled, scaler


def _run_hdbscan(X_sample: np.ndarray):
    n = len(X_sample)
    print(f"\nHDBSCAN fit baslatiliyor ({n:,} post — %{SAMPLE_FRAC*100:.0f} orneklem)...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2_000,    # orneklemde 2k ~ tam veride 20k
        min_samples=20,
        metric="euclidean",
        cluster_selection_method="eom",
        algorithm="boruvka_kdtree",
        approx_min_span_tree=True,
        leaf_size=40,
        core_dist_n_jobs=-1,
        prediction_data=True,      # approximate_predict icin gerekli
    )

    stop_event = threading.Event()
    ticker = threading.Thread(target=_elapsed_ticker, args=(stop_event, time.time()), daemon=True)
    ticker.start()
    try:
        clusterer.fit(X_sample)
    finally:
        stop_event.set()
        ticker.join()

    n_clusters = int(clusterer.labels_.max()) + 1
    n_noise    = (clusterer.labels_ == -1).sum()
    print(f"   Orneklemde kume: {n_clusters}  |  noise: {n_noise:,}")
    return clusterer


def _predict_remaining(clusterer, X_rest: np.ndarray) -> tuple:
    """approximate_predict in batches to avoid RAM spike."""
    BATCH = 100_000
    labels_all = np.empty(len(X_rest), dtype=np.int32)
    probs_all  = np.empty(len(X_rest), dtype=np.float32)

    n_batches = (len(X_rest) + BATCH - 1) // BATCH
    for i in tqdm(range(n_batches), desc="approximate_predict", unit="batch"):
        lo, hi = i * BATCH, min((i + 1) * BATCH, len(X_rest))
        lbl, prb = hdbscan.approximate_predict(clusterer, X_rest[lo:hi])
        labels_all[lo:hi] = lbl
        probs_all[lo:hi]  = prb

    return labels_all, probs_all


def _build_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    cluster_ids = sorted(df["cluster"].unique())
    rows = []
    for cid in tqdm(cluster_ids, desc="Kume ozeti", unit="kume"):
        sub = df[df["cluster"] == cid]
        row = {
            "cluster":            cid,
            "is_noise":           cid == -1,
            "post_count":         len(sub),
            "mean_organic_score": round(sub["organic_score"].mean(), 4),
        }
        for col in CLUSTER_FEATURES:
            row[f"mean_{col}"] = round(sub[col].mean(), 4)
        row["top_languages"] = (
            sub["language"].value_counts(normalize=True).head(5).round(3).to_dict()
        )
        row["top_domains"] = (
            sub["domain"].value_counts(normalize=True).head(5).round(3).to_dict()
        )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("mean_organic_score")
        .reset_index(drop=True)
    )


def run_hdbscan_pipeline():
    start = time.time()

    # 1. Load
    df = _load_data()
    n_total = len(df)

    # 2. Scale full dataset
    print("RobustScaler fit (tam veri)...")
    X_scaled, scaler = _build_feature_matrix(df)

    # 3. Stratified sample for fit
    n_sample   = max(int(n_total * SAMPLE_FRAC), 50_000)
    rng        = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(n_total, size=n_sample, replace=False)
    rest_idx   = np.setdiff1d(np.arange(n_total), sample_idx)

    X_sample = X_scaled[sample_idx]
    X_rest   = X_scaled[rest_idx]
    print(f"   Fit orneklemi : {n_sample:,}  |  Tahmin edilecek: {len(rest_idx):,}")

    # 4. Fit on sample
    clusterer = _run_hdbscan(X_sample)

    # 5. Assign sample labels
    labels_full = np.full(n_total, -1, dtype=np.int32)
    probs_full  = np.zeros(n_total, dtype=np.float32)
    labels_full[sample_idx] = clusterer.labels_
    probs_full[sample_idx]  = clusterer.probabilities_

    # 6. Predict remaining
    print(f"\napproximate_predict ({len(rest_idx):,} post)...")
    rest_labels, rest_probs = _predict_remaining(clusterer, X_rest)
    labels_full[rest_idx] = rest_labels
    probs_full[rest_idx]  = rest_probs

    df["cluster"]      = labels_full
    df["cluster_prob"] = probs_full

    # 7. Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nKaydediliyor...")
    joblib.dump(clusterer, MODELS_DIR / "hdbscan.pkl")
    joblib.dump(scaler,    MODELS_DIR / "hdbscan_scaler.pkl")

    save_cols = (
        ["author_hash", "cluster", "cluster_prob", "organic_score", "language", "domain"]
        + CLUSTER_FEATURES
    )
    df[save_cols].to_parquet(OUTPUTS_DIR / "post_clusters.parquet", index=False)

    # 8. Summary
    summary = _build_cluster_summary(df)
    summary.to_csv(OUTPUTS_DIR / "cluster_summary.csv", index=False)

    elapsed     = time.time() - start
    n_clusters  = int(df["cluster"].max()) + 1
    n_noise     = (df["cluster"] == -1).sum()
    n_clustered = (df["cluster"] >= 0).sum()

    print(f"\n{'='*55}")
    print(f"HDBSCAN tamamlandi ({elapsed:.1f}s)")
    print(f"{'='*55}")
    print(f"  Kume sayisi   : {n_clusters}")
    print(f"  Kumeli post   : {n_clustered:,}  ({n_clustered/n_total*100:.1f}%)")
    print(f"  Noise (-1)    : {n_noise:,}  ({n_noise/n_total*100:.1f}%)")
    print(f"\nKume ozeti (mean_organic_score sirali):")
    print(
        summary[["cluster", "post_count", "mean_organic_score", "is_noise"]]
        .to_string(index=False)
    )
    print(f"{'='*55}")

    return df, summary


if __name__ == "__main__":
    run_hdbscan_pipeline()
