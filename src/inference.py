"""
Inference pipeline for social media manipulation detection.

Training mode  : python src/inference.py train
Prediction mode: python src/inference.py "your text here"

Features used (6 — no DL/LLM, any language):
  text_len, kw_count, kw_density,
  is_duplicate, cross_author_dup_count, kw_fingerprint_shared

HDBSCAN cluster labels are loaded separately to explain *why* a post is flagged.
"""

import hashlib
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models" / "task_3"
OUTPUTS_DIR   = Path(__file__).parent.parent / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature config
# ---------------------------------------------------------------------------
INFERENCE_FEATURES = [
    "text_len",
    "kw_count",
    "kw_density",
    "is_duplicate",
    "cross_author_dup_count",
    "kw_fingerprint_shared",
]

LOG1P_COLS = ["text_len", "kw_count", "cross_author_dup_count", "kw_fingerprint_shared"]

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _md5(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _kw_fingerprint(keywords: list) -> str:
    """Top-8 sorted keywords -> MD5 fingerprint (matches features.py logic)."""
    words = sorted([w.strip() for w in keywords if w.strip()])[:8]
    if not words:
        return ""
    return _md5(",".join(words))


def extract_keywords(text: str, max_kw: int = 20) -> list:
    """
    YAKE multilingual keyword extraction — no DL, works for any language.
    Falls back to whitespace tokens if YAKE is unavailable.
    """
    try:
        import yake
        kw_extractor = yake.KeywordExtractor(
            lan="en",       # YAKE's language hint; works on non-English too
            n=1,            # unigrams only (matches features.py token style)
            dedupLim=0.9,
            top=max_kw,
            features=None,
        )
        return [kw for kw, _ in kw_extractor.extract_keywords(text)]
    except ImportError:
        tokens = [t.strip(".,!?\"'") for t in text.split() if len(t) > 3]
        return list(dict.fromkeys(tokens))[:max_kw]


def compute_text_features(text: str, dup_df, kw_df) -> dict:
    """Extract the 6 inference features from raw text + lookup tables."""
    keywords = extract_keywords(text)

    text_len   = len(text)
    kw_count   = len(keywords)
    kw_density = kw_count / (text_len + 1)

    text_hash = _md5(text)
    kw_fp     = _kw_fingerprint(keywords)

    cross_author_dup_count = 0
    kw_fingerprint_shared  = 0

    if dup_df is not None and text_hash:
        row = dup_df[dup_df["text_hash"] == text_hash]
        if not row.empty:
            cross_author_dup_count = int(row["cross_author_dup_count"].iloc[0])

    if kw_df is not None and kw_fp:
        row = kw_df[kw_df["kw_fingerprint"] == kw_fp]
        if not row.empty:
            kw_fingerprint_shared = int(row["kw_fingerprint_shared"].iloc[0])

    is_duplicate = int(cross_author_dup_count > 0)

    return {
        "text_len":               text_len,
        "kw_count":               kw_count,
        "kw_density":             kw_density,
        "is_duplicate":           is_duplicate,
        "cross_author_dup_count": cross_author_dup_count,
        "kw_fingerprint_shared":  kw_fingerprint_shared,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_inference_model():
    """
    Train a 6-feature IsolationForest for live inference.
    Saves: inf_isolation_forest.pkl, inf_scaler.pkl,
           inf_minmax_scaler.pkl, inf_clip_bounds.npy
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import MinMaxScaler, RobustScaler

    print("=" * 60)
    print("Inference Model Training (6 features)")
    print("=" * 60)
    start = time.time()

    print("\n1. Loading post_features.parquet...")
    df = pd.read_parquet(PROCESSED_DIR / "post_features.parquet", columns=INFERENCE_FEATURES)
    print(f"   {len(df):,} posts loaded.")

    print("\n2. Log1p transform + fill...")
    X = df[INFERENCE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).copy()
    for col in LOG1P_COLS:
        X[col] = np.log1p(X[col])

    print("\n3. RobustScaler...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X.values)
    joblib.dump(scaler, MODELS_DIR / "inf_scaler.pkl")

    print("\n4. Training IsolationForest (n_estimators=300)...")
    iso = IsolationForest(
        n_estimators=300,
        max_samples=10_000,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_scaled)
    joblib.dump(iso, MODELS_DIR / "inf_isolation_forest.pkl")

    print("\n5. Clip bounds + MinMax scaling...")
    raw_scores  = iso.score_samples(X_scaled)
    lower_bound = np.percentile(raw_scores, 0.1)
    upper_bound = np.percentile(raw_scores, 99.9)
    np.save(MODELS_DIR / "inf_clip_bounds.npy", np.array([lower_bound, upper_bound]))

    clipped = np.clip(raw_scores, lower_bound, upper_bound)
    mm = MinMaxScaler(feature_range=(0, 1))
    mm.fit(clipped.reshape(-1, 1))
    joblib.dump(mm, MODELS_DIR / "inf_minmax_scaler.pkl")

    scores = mm.transform(clipped.reshape(-1, 1)).flatten()
    print(f"   Raw score range   : {raw_scores.min():.3f} -> {raw_scores.max():.3f}")
    print(f"   Clip bounds       : {lower_bound:.3f} -> {upper_bound:.3f}")
    print(f"   Organic score mean: {scores.mean():.3f}")

    elapsed = time.time() - start
    print(f"\nDone ({elapsed:.1f}s). Models saved to {MODELS_DIR}")


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """
    Load trained models + lookup tables once, then call predict(text) repeatedly.
    """

    MANIPULATION_THRESHOLD = 0.35

    def __init__(self):
        self._load_models()
        self._load_lookups()
        self._load_cluster_summary()

    def _load_models(self):
        iso_path = MODELS_DIR / "inf_isolation_forest.pkl"
        sc_path  = MODELS_DIR / "inf_scaler.pkl"
        mm_path  = MODELS_DIR / "inf_minmax_scaler.pkl"
        cb_path  = MODELS_DIR / "inf_clip_bounds.npy"

        missing = [p for p in (iso_path, sc_path, mm_path, cb_path) if not p.exists()]
        if missing:
            names = ", ".join(p.name for p in missing)
            raise FileNotFoundError(
                f"Model dosyalari eksik: {names}\n"
                "Once su komutu calistir: python src/inference.py train"
            )

        self.iso         = joblib.load(iso_path)
        self.scaler      = joblib.load(sc_path)
        self.mm_scaler   = joblib.load(mm_path)
        self.clip_bounds = np.load(cb_path)

    def _load_lookups(self):
        dup_path = PROCESSED_DIR / "dup_lookup.parquet"
        kw_path  = PROCESSED_DIR / "kw_lookup.parquet"
        self.dup_df = pd.read_parquet(dup_path) if dup_path.exists() else None
        self.kw_df  = pd.read_parquet(kw_path)  if kw_path.exists()  else None

        if self.dup_df is None:
            print("[UYARI] dup_lookup.parquet bulunamadi — capraz kopya tespiti devre disi.")
        if self.kw_df is None:
            print("[UYARI] kw_lookup.parquet bulunamadi — anahtar kelime eslesmesi devre disi.")

    def _load_cluster_summary(self):
        path = OUTPUTS_DIR / "cluster_summary.csv"
        if path.exists():
            self.cluster_summary = pd.read_csv(path)
            median_score = self.cluster_summary["mean_organic_score"].median()
            self.manipulation_clusters = set(
                self.cluster_summary.loc[
                    (self.cluster_summary["mean_organic_score"] < median_score)
                    & (self.cluster_summary["cluster"] >= 0),
                    "cluster",
                ].tolist()
            )
        else:
            self.cluster_summary       = None
            self.manipulation_clusters = set()

        hdbscan_path = MODELS_DIR / "hdbscan.pkl"
        hdbscan_sc   = MODELS_DIR / "hdbscan_scaler.pkl"
        if hdbscan_path.exists() and hdbscan_sc.exists():
            self.hdbscan_model  = joblib.load(hdbscan_path)
            self.hdbscan_scaler = joblib.load(hdbscan_sc)
        else:
            self.hdbscan_model  = None
            self.hdbscan_scaler = None

    def _cluster_label(self, feat: dict):
        """Assign text to HDBSCAN cluster via approximate_predict."""
        if self.hdbscan_model is None or self.cluster_summary is None:
            return None

        HDBSCAN_FEATURES = [
            "text_len", "kw_count", "kw_density", "sentiment",
            "sentiment_extreme", "is_duplicate", "cross_author_dup_count",
            "kw_fingerprint_shared", "has_author",
        ]
        HDBSCAN_LOG1P = ["text_len", "kw_count", "cross_author_dup_count", "kw_fingerprint_shared"]

        x = np.array([[feat.get(f, 0) for f in HDBSCAN_FEATURES]], dtype=float)
        for i, col in enumerate(HDBSCAN_FEATURES):
            if col in HDBSCAN_LOG1P:
                x[0, i] = np.log1p(x[0, i])

        x_sc = self.hdbscan_scaler.transform(x)

        try:
            import hdbscan as _hdbscan
            labels, _ = _hdbscan.approximate_predict(self.hdbscan_model, x_sc)
            cid = int(labels[0])
        except Exception:
            return None

        if cid == -1:
            return {"cluster_id": -1, "label": "noise (kume disi)", "is_manipulation_cluster": False}

        meta = self.cluster_summary[self.cluster_summary["cluster"] == cid]
        if meta.empty:
            return None

        m = meta.iloc[0]
        return {
            "cluster_id":              cid,
            "mean_organic_score":      round(float(m["mean_organic_score"]), 3),
            "is_manipulation_cluster": cid in self.manipulation_clusters,
            "top_languages":           m.get("top_languages", ""),
            "top_domains":             m.get("top_domains", ""),
        }

    def predict(self, text: str) -> dict:
        """Raw text -> result dict with organic_score, verdict, signals, features, cluster."""
        feat = compute_text_features(text, self.dup_df, self.kw_df)

        x = np.array([[feat[f] for f in INFERENCE_FEATURES]], dtype=float)
        for i, col in enumerate(INFERENCE_FEATURES):
            if col in LOG1P_COLS:
                x[0, i] = np.log1p(x[0, i])

        x_sc      = self.scaler.transform(x)
        raw_score = float(self.iso.score_samples(x_sc)[0])
        clipped   = float(np.clip(raw_score, self.clip_bounds[0], self.clip_bounds[1]))
        org_score = float(self.mm_scaler.transform([[clipped]])[0, 0])

        is_manip = org_score < self.MANIPULATION_THRESHOLD
        verdict  = "Manipulatif" if is_manip else "Organik"

        signals = []
        if feat["is_duplicate"]:
            signals.append(
                f"Bu metnin tam kopyasi {feat['cross_author_dup_count']} farkli hesap tarafindan paylasilmis."
            )
        if feat["kw_fingerprint_shared"] > 1:
            signals.append(
                f"Ayni anahtar kelime kombinasyonu {feat['kw_fingerprint_shared']} farkli hesapta gorulmus."
            )
        if feat["kw_density"] > 0.08:
            signals.append("Anahtar kelime yogunlugu anormal derecede yuksek.")
        if feat["text_len"] < 30:
            signals.append("Metin cok kisa (bot tarzi).")
        if not signals and is_manip:
            signals.append("Izolasyon Ormani bu metni genel ornuntuden sapma olarak tespit etti.")

        cluster_info = self._cluster_label(
            {**feat, "sentiment": 0.0, "sentiment_extreme": 0, "has_author": 0}
        )

        return {
            "organic_score": round(org_score, 4),
            "verdict":       verdict,
            "signals":       signals,
            "features":      feat,
            "cluster":       cluster_info,
        }

    def print_result(self, result: dict):
        score   = result["organic_score"]
        bar_len = int(score * 30)
        bar     = "#" * bar_len + "-" * (30 - bar_len)

        print("\n" + "=" * 55)
        print(f"  Organiklik Skoru : {score:.4f}  [{bar}]")
        print(f"  Karar            : {result['verdict']}")
        print("=" * 55)

        if result["signals"]:
            print("\nTespit Edilen Sinyaller:")
            for s in result["signals"]:
                print(f"  * {s}")

        feat = result["features"]
        print("\nCikarilan Ozellikler:")
        print(f"  Metin uzunlugu     : {feat['text_len']}")
        print(f"  Anahtar kelime say.: {feat['kw_count']}")
        print(f"  KW yogunlugu       : {feat['kw_density']:.4f}")
        print(f"  Capraz kopya say.  : {feat['cross_author_dup_count']}")
        print(f"  KW parmak izi esm. : {feat['kw_fingerprint_shared']}")

        cl = result["cluster"]
        if cl:
            print("\nKume Bilgisi:")
            if cl["cluster_id"] == -1:
                print("  Kume: gurultu (benzersiz icerik)")
            else:
                tag = " <- MANIPULASYON KUMESI" if cl["is_manipulation_cluster"] else ""
                print(f"  Kume ID           : {cl['cluster_id']}{tag}")
                print(f"  Kume ort. organik : {cl['mean_organic_score']}")
                if cl.get("top_languages"):
                    print(f"  Baskin diller     : {cl['top_languages']}")
                if cl.get("top_domains"):
                    print(f"  Baskin platformlar: {cl['top_domains']}")

        print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Kullanim:")
        print("  python src/inference.py train          # modeli egit")
        print('  python src/inference.py "metin buraya" # tahmin yap')
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd.lower() == "train":
        train_inference_model()
        return

    text = " ".join(sys.argv[1:])
    print(f'\nMetin: "{text[:120]}{"..." if len(text) > 120 else ""}"')
    print("Model yukleniyor...")

    try:
        pipeline = InferencePipeline()
    except FileNotFoundError as e:
        print(f"\nHata: {e}")
        sys.exit(1)

    result = pipeline.predict(text)
    pipeline.print_result(result)


if __name__ == "__main__":
    main()
