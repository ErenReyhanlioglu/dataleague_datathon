"""
Feature engineering module for Datathon.
Extracts 10 Post-Level and 7 Author-Level features (17 total).
"""

import hashlib
import random
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

# Proje ana dizinini sys.path'e ekleyerek 'src.X' importlarının çalışmasını sağla
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import iter_batches

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def md5_hash(text: str) -> str:
    """Metnin MD5 hash'ini döndürür."""
    if not isinstance(text, str) or not text.strip():
        return ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def kw_fingerprint(kws: str) -> str:
    """Top-8 alfabetik anahtar kelime kombinasyonunun MD5 hash'ini döndürür."""
    if not isinstance(kws, str) or not kws.strip():
        return ""
    words = sorted([w.strip() for w in kws.split(",") if w.strip()])[:8]
    if not words:
        return ""
    return md5_hash(",".join(words))


def build_lookups():
    """
    Tüm veriyi batch'ler halinde okuyup (RAM dostu), metin kopyalarını
    ve anahtar kelime parmak izi eşleşmelerini çıkarır.
    """
    print("Lookup tabloları oluşturuluyor... (Bu işlem bir kere yapılır)")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dup_map = defaultdict(set)
    kw_map = defaultdict(set)

    for batch in tqdm(iter_batches(columns=["original_text", "english_keywords", "author_hash"]), desc="Building Lookups"):
        texts = batch["original_text"].fillna("").tolist()
        kws = batch["english_keywords"].fillna("").tolist()
        authors = batch["author_hash"].fillna("").tolist()

        for t, k, a in zip(texts, kws, authors):
            if not a:  # Yazarı olmayanları çapraz karşılaştırmaya katma
                continue
                
            # MD5 hash of text
            t_hash = md5_hash(t)
            if t_hash:
                dup_map[t_hash].add(a)
                
            # Fingerprint of keywords
            k_hash = kw_fingerprint(k)
            if k_hash:
                kw_map[k_hash].add(a)

    # Dict'leri DataFrame'e dönüştür (sadece 1'den fazla author varsa filtrele)
    print("Lookup DataFrame'leri kaydediliyor...")
    
    dup_records = [{"text_hash": k, "cross_author_dup_count": len(v)} for k, v in dup_map.items() if len(v) > 1]
    dup_df = pd.DataFrame(dup_records)
    if not dup_df.empty:
        dup_df.to_parquet(PROCESSED_DIR / "dup_lookup.parquet")

    kw_records = [{"kw_fingerprint": k, "kw_fingerprint_shared": len(v)} for k, v in kw_map.items() if len(v) > 1]
    kw_df = pd.DataFrame(kw_records)
    if not kw_df.empty:
        kw_df.to_parquet(PROCESSED_DIR / "kw_lookup.parquet")
        
    return dup_df, kw_df


def compute_post_features(df: pd.DataFrame, dup_df: pd.DataFrame = None, kw_df: pd.DataFrame = None) -> pd.DataFrame:
    """10 boyutlu post-level (gönderi bazlı) özellikleri çıkarır."""
    
    # 1. text_len
    df["text_len"] = df["original_text"].fillna("").str.len()
    
    # 2. kw_count
    df["kw_count"] = df["english_keywords"].fillna("").apply(
        lambda x: len([w for w in str(x).split(",") if w.strip()])
    )
    
    # 3. kw_density
    df["kw_density"] = df["kw_count"] / (df["text_len"] + 1)
    
    # 5. sentiment (veri setinden alınır, eksikse 0.0)
    df["sentiment"] = df.get("sentiment", 0.0).fillna(0.0)
    
    # 6. sentiment_extreme
    df["sentiment_extreme"] = (df["sentiment"].abs() > 0.8).astype(int)
    
    # 7. has_author (data_loader zenginleştiriyor, ama garanti altına alalım)
    if "has_author" not in df.columns:
        df["has_author"] = ~df["author_hash"].isin(["", "da39a3ee5e6b4b0d3255bfef95601890afd80709"]).astype(int)
    else:
        df["has_author"] = df["has_author"].astype(int)
        
    # Lookup eşleşmeleri için Hash kolonları
    df["text_hash"] = df["original_text"].apply(md5_hash)
    df["kw_fingerprint_hash"] = df["english_keywords"].apply(kw_fingerprint)
    
    # Join Dup Lookup
    if dup_df is not None and not dup_df.empty:
        df = df.merge(dup_df, on="text_hash", how="left")
    else:
        df["cross_author_dup_count"] = 0
        
    # Join KW Lookup
    if kw_df is not None and not kw_df.empty:
        df = df.merge(kw_df, left_on="kw_fingerprint_hash", right_on="kw_fingerprint", how="left")
    else:
        df["kw_fingerprint_shared"] = 0

    # Eksikleri doldur (0)
    # 8. cross_author_dup_count
    df["cross_author_dup_count"] = df["cross_author_dup_count"].fillna(0).astype(int)
    
    # 9. kw_fingerprint_shared
    df["kw_fingerprint_shared"] = df["kw_fingerprint_shared"].fillna(0).astype(int)
    
    # 10. is_duplicate
    df["is_duplicate"] = (df["cross_author_dup_count"] > 0).astype(int)

    # RAM'i temizle
    df = df.drop(columns=["text_hash", "kw_fingerprint_hash", "kw_fingerprint"], errors="ignore")
    return df


def _fast_jaccard_mean(kw_series: pd.Series) -> float:
    """Bir yazarın kendi metinlerindeki ortalama Jaccard benzerliği (OOM korumalı)."""
    sets = [set(x.split(",")) for x in kw_series.dropna() if isinstance(x, str) and x]
    
    n = len(sets)
    if n < 2:
        return 0.0
        
    # Maksimum 50 gönderiyi karşılaştır. (50*49/2 = ~1225 işlem). 
    if n > 50:
        sets = random.sample(sets, 50)
        n = 50
        
    total_sim = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            intersection = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            if union > 0:
                total_sim += intersection / union
            count += 1
            
    return total_sim / count if count > 0 else 0.0


def compute_author_features(df: pd.DataFrame) -> pd.DataFrame:
    """7 boyutlu author-level özellikleri çıkarır."""
    
    valid_authors = df[df["has_author"] == 1].copy()
    if valid_authors.empty:
        return pd.DataFrame()
        
    # İşlemleri hızlandırmak için zaman serisi olarak sırala
    valid_authors = valid_authors.sort_values(["author_hash", "timestamp"])
    
    # Temel groupby
    grouped = valid_authors.groupby("author_hash")
    
    author_df = pd.DataFrame(index=grouped.groups.keys())
    
    # 1. author_post_count
    author_df["author_post_count"] = grouped.size()
    
    # 2. author_posts_per_day
    min_ts = grouped["timestamp"].min()
    max_ts = grouped["timestamp"].max()
    active_days = (max_ts - min_ts).dt.total_seconds() / 86400.0 + 1.0
    author_df["author_posts_per_day"] = author_df["author_post_count"] / active_days
    
    # 3. author_min_interval_sec
    valid_authors["diff_sec"] = valid_authors.groupby("author_hash")["timestamp"].diff().dt.total_seconds()
    author_df["author_min_interval_sec"] = valid_authors.groupby("author_hash")["diff_sec"].min().fillna(86400)
    
    # 5. author_sentiment_std
    author_df["author_sentiment_std"] = grouped["sentiment"].std().fillna(0.0)

    # 6. author_unique_themes
    author_df["author_unique_themes"] = grouped["primary_theme"].nunique()

    # 7. author_duplicate_ratio
    if "cross_author_dup_count" in valid_authors.columns:
        dup_counts = valid_authors[valid_authors["cross_author_dup_count"] > 1].groupby("author_hash").size()
        author_df["author_duplicate_ratio"] = (dup_counts / author_df["author_post_count"]).fillna(0.0)
    else:
        author_df["author_duplicate_ratio"] = 0.0

    # 8. author_mean_jaccard (Bu işlem nispeten ağırdır)
    tqdm.pandas(desc="Calculating Jaccard Means")
    author_df["author_mean_jaccard"] = grouped["english_keywords"].progress_apply(_fast_jaccard_mean)

    # author_post_count ML feature değil, sadece hesaplama için kullanıldı
    author_df = author_df.drop(columns=["author_post_count"])

    return author_df.reset_index().rename(columns={"index": "author_hash"})


def build_all_features(df: pd.DataFrame, is_inference: bool = False) -> pd.DataFrame:
    """Tüm feature boru hattını yönetir. Çıktı: ML Modeli için hazır DataFrame"""
    
    dup_df, kw_df = None, None
    dup_path, kw_path = PROCESSED_DIR / "dup_lookup.parquet", PROCESSED_DIR / "kw_lookup.parquet"
    
    # Inference anında jüri metni daha önce kullanılmış mı diye bakmak için lookupları her zaman yükle
    if dup_path.exists() and kw_path.exists():
        dup_df = pd.read_parquet(dup_path)
        kw_df = pd.read_parquet(kw_path)
    elif not is_inference:
        dup_df, kw_df = build_lookups()

    df = compute_post_features(df, dup_df, kw_df)
    
    # Eğitim ve Pipeline aşaması ise Yazar (Author) özelliklerini hesapla ve birleştir
    if not is_inference:
        author_df = compute_author_features(df)
        if not author_df.empty:
            author_df.to_parquet(PROCESSED_DIR / "author_features.parquet")
            df = df.merge(author_df, on="author_hash", how="left")
            
    author_cols = [
        "author_posts_per_day", "author_min_interval_sec",
        "author_sentiment_std", "author_unique_themes", "author_duplicate_ratio",
        "author_mean_jaccard"
    ]

    for col in author_cols:
        if col not in df.columns:
            df[col] = np.nan

    if not is_inference:
        medians = author_df[author_cols].median() if not author_df.empty else 0.0
        df[author_cols] = df[author_cols].fillna(medians)
    else:
        df[author_cols] = df[author_cols].fillna(0.0)

    return df

if __name__ == "__main__":
    import time
    from src.data_loader import _enrich
    
    start_time = time.time()
    
    print("1. Lookup tabloları kontrol ediliyor...")
    dup_path, kw_path = PROCESSED_DIR / "dup_lookup.parquet", PROCESSED_DIR / "kw_lookup.parquet"
    if not dup_path.exists() or not kw_path.exists():
        build_lookups()
        
    dup_df = pd.read_parquet(dup_path)
    kw_df = pd.read_parquet(kw_path)
    
    print("2. Veri seti batch'ler halinde okunuyor ve Post-Level özellikler çıkarılıyor (OOM Korumalı)...")
    chunks = []
    for batch in tqdm(iter_batches(), desc="Processing Post-Level"):
        batch = _enrich(batch)
        batch = compute_post_features(batch, dup_df, kw_df)
        
        # OOM Koruması: RAM'i aşırı tüketen ham metin kolonlarını siliyoruz (artık ihtiyaç yok)
        batch = batch.drop(columns=["original_text", "url", "date"], errors="ignore")
        chunks.append(batch)
        
    full_df = pd.concat(chunks, ignore_index=True)
    
    print(f"3. Author-Level özellikler hesaplanıyor ({len(full_df)} satır)...")
    author_df = compute_author_features(full_df)
    
    if not author_df.empty:
        author_df.to_parquet(PROCESSED_DIR / "author_features.parquet")
        full_df = full_df.merge(author_df, on="author_hash", how="left")
        
    author_cols = [
        "author_posts_per_day", "author_min_interval_sec",
        "author_sentiment_std", "author_unique_themes", "author_duplicate_ratio",
        "author_mean_jaccard"
    ]

    for col in author_cols:
        if col not in full_df.columns:
            full_df[col] = np.nan

    medians = author_df[author_cols].median() if not author_df.empty else 0.0
    full_df[author_cols] = full_df[author_cols].fillna(medians)
    
    # RAM ve Disk Tasarrufu: Artık işimiz kalmayan ağır list kolonunu atıyoruz
    full_df = full_df.drop(columns=["english_keywords"], errors="ignore")

    out_path = PROCESSED_DIR / "post_features.parquet"
    print(f"4. Özellikler hesaplandı. {out_path} konumuna kaydediliyor...")
    full_df.to_parquet(out_path)
    
    elapsed = time.time() - start_time
    print(f"İşlem başarıyla tamamlandı! (Süre: {elapsed:.2f} saniye)")
