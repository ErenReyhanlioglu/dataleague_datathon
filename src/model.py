"""
Model training module for Datathon.

Isolation Forest is trained with 15 post+author-level features.
Contamination is set to "auto" for theoretical thresholding.
To enforce a [0, 1] range without outlier distortion, raw scores are
clipped at the 0.1th and 99.9th percentiles before MinMax scaling.
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, RobustScaler

sys.path.append(str(Path(__file__).parent.parent))

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models" / "task_1"
OUTPUTS_DIR   = Path(__file__).parent.parent / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

IF_FEATURES = [
    "text_len",
    "kw_count",
    "kw_density",
    "sentiment",
    "sentiment_extreme",
    "is_duplicate",
    "has_author",
    "cross_author_dup_count",
    "kw_fingerprint_shared",
    "author_posts_per_day",
    "author_min_interval_sec",
    "author_mean_jaccard",
    "author_sentiment_std",
    "author_unique_themes",
    "author_duplicate_ratio",
]

LOG1P_FEATURES = [
    "text_len",
    "kw_count",
    "cross_author_dup_count",
    "kw_fingerprint_shared",
    "author_posts_per_day",
    "author_min_interval_sec",
    "author_unique_themes",
]

AUTHOR_REPORT_COLS = [
    "author_posts_per_day",
    "author_min_interval_sec",
    "author_unique_themes",
    "author_duplicate_ratio",
    "author_sentiment_std",
    "author_mean_jaccard",
]

def train_model():
    print("=" * 60)
    print("Isolation Forest Training")
    print("=" * 60)
    start_time = time.time()

    # 1. Load Data
    print("\n1. Loading post_features.parquet...")
    df = pd.read_parquet(PROCESSED_DIR / "post_features.parquet")

    # 2. Prepare Feature Matrix
    df_feat = df[IF_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).copy()

    for col in LOG1P_FEATURES:
        if col in df_feat.columns:
            df_feat[col] = np.log1p(df_feat[col])

    # 3. RobustScaler
    print("\n3. Fitting RobustScaler...")
    robust_scaler = RobustScaler()
    X_scaled = robust_scaler.fit_transform(df_feat.values)
    joblib.dump(robust_scaler, MODELS_DIR / "scaler.pkl")

    # 4. Isolation Forest
    print("\n4. Training Isolation Forest (contamination='auto')")
    iso_forest = IsolationForest(
        n_estimators=300,
        max_samples=10_000,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_scaled)
    joblib.dump(iso_forest, MODELS_DIR / "isolation_forest.pkl")

    # 5. Extract Raw Scores and Apply Clipped MinMax
    print("\n5. Applying statistical clipping and MinMax scaling...")
    raw_scores = iso_forest.score_samples(X_scaled)
    
    # Calculate bounds to prevent single-outlier scale stretching
    lower_bound = np.percentile(raw_scores, 0.1)
    upper_bound = np.percentile(raw_scores, 99.9)
    
    # Clip scores to designated statistical boundaries
    clipped_scores = np.clip(raw_scores, lower_bound, upper_bound)
    
    # Scale to strict [0, 1] range
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    organic_scores = minmax_scaler.fit_transform(clipped_scores.reshape(-1, 1)).flatten()
    
    joblib.dump(minmax_scaler, MODELS_DIR / "minmax_scaler.pkl")
    
    df["organic_score"] = organic_scores

    print(f"   -> Raw score range : {raw_scores.min():.3f} to {raw_scores.max():.3f}")
    print(f"   -> Clipped bounds  : {lower_bound:.3f} to {upper_bound:.3f}")
    print(f"   -> Organic mean    : {organic_scores.mean():.3f}")

    # 6. Save post_scores
    print("\n6. Saving outputs/post_scores.parquet...")
    post_score_cols = ["author_hash", "organic_score"]
    for opt in ["timestamp", "language", "domain"]:
        if opt in df.columns:
            post_score_cols.append(opt)
    df[post_score_cols].to_parquet(OUTPUTS_DIR / "post_scores.parquet", index=False)

    # 7. Save author_scores
    print("\n7. Saving outputs/author_scores.csv...")
    valid = df[df["has_author"] == 1].copy()

    agg_dict = {"organic_score": "mean"}
    for col in AUTHOR_REPORT_COLS:
        if col in valid.columns:
            agg_dict[col] = "first"

    author_scores = (
        valid.groupby("author_hash")
        .agg(agg_dict)
        .reset_index()
        .rename(columns={"organic_score": "mean_organic_score"})
        .sort_values("mean_organic_score", ascending=True) 
    )
    author_scores.to_csv(OUTPUTS_DIR / "author_scores.csv", index=False)

    # 8. Save top_bots
    print("\n8. Saving outputs/top_bots.csv...")
    author_scores.head(1000).to_csv(OUTPUTS_DIR / "top_bots.csv", index=False)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Completed ({elapsed:.1f}s)")
    print("=" * 60)

if __name__ == "__main__":
    train_model()