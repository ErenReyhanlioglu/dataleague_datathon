# ARCHITECTURE.md
# Sosyal Medya Manipülasyon Tespit Sistemi
# DataLeague Datathon — Unsupervised Anomaly Detection
# =====================================================

---

## Proje Yapısı

```
datathon/
├── ARCHITECTURE.md
├── CLAUDE.md
├── pyproject.toml
├── .python-version
│
├── data/
│   ├── raw/
│   │   └── datathonFINAL.parquet          ← gitignore'da
│   └── processed/
│       ├── post_features.parquet          ← (post-level + author-level birleşik) 15 feature, tüm postlar
│       ├── author_features.parquet        ← (author-level) 6 feature author aggregate
│       ├── dup_lookup.parquet             ← metin hash → author count
│       └── kw_lookup.parquet              ← kw fingerprint → author count
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_if_scorer.ipynb
│   ├── 04_manipulation_map.ipynb
│   └── 05_inference.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   ├── inference.py
│   └── dashboard.py
│
├── models/
│   ├── isolation_forest.pkl
│   └── scaler.pkl
│
└── outputs/
    ├── manipulation_map.html
    ├── post_scores.parquet
    ├── author_scores.csv
    └── top_bots.csv
```

---

## Temel Varsayım

Etiket (label) yoktur. Problemin çözümü şu varsayıma dayanır:

- Gerçek kullanıcılar davranışlarında gürültülü, çeşitli ve tutarsızdır.
- Bot ve koordineli hesaplar düzenli, tekrarlı ve senkronize hareket eder.

Bu fark, post ve author seviyesindeki istatistiksel sapmalardan çıkarılır.

---

## Veri Seti — Özet İstatistikler (EDA'dan)

| Metrik | Değer |
|---|---|
| Toplam post | 5,004,813 |
| Unique author | 1,597,838 |
| has_author=True | %58.4 |
| has_author=False | %41.6 (scraper'ın ulaşamadığı hesaplar) |
| Tarih aralığı | 14–24 Kasım 2024 (10 gün) |
| Dominant platform | x.com (%57), reddit (%27), bsky (%10) |
| Dominant dil | İngilizce (%75) |
| Median post/author | 1 (extreme long-tail) |
| Confirmed bot (0a92fab3) | 35,280 post, 62 dil, medyan interval 1s |
| Koordineli metin (full) | 93,481 metin — max 572 farklı author aynı metni paylaşmış |

---

## Veri Seti — Kolon Açıklamaları

| Kolon | Açıklama |
|---|---|
| `original_text` | Ham metin |
| `english_keywords` | Önceden hesaplanmış İngilizce anahtar kelimeler (virgülle ayrılmış) |
| `sentiment` | Önceden hesaplanmış duygu skoru: −1.0 → +1.0 |
| `main_emotion` | Önceden hesaplanmış hakim duygu (26 sınıf) |
| `primary_theme` | Önceden hesaplanmış ana konu (15 sınıf) |
| `language` | ISO dil kodu |
| `url` | Platform/domain |
| `author_hash` | Anonimleştirilmiş kullanıcı kimliği |
| `date` | UTC timestamp |

`sentiment`, `main_emotion`, `primary_theme`, `english_keywords` önceden
hesaplanmış türetilmiş değerlerdir — doğrudan feature olarak kullanılır.

---

## Pipeline Akışı

```
5M Ham Post
      │
      ▼
 data_loader.py
 (dtype optimize, has_author flag, domain parse, timestamp)
      │
      ▼
 features.py
 ┌──────────────────────────────────────────────────────┐
 │ ADIM 1 — Post-Level Feature Extraction (5M post)     │
 │ Her post için hesaplanır, imputation yok             │
 │                                                      │
 │ ADIM 2 — Lookup Tabloları (full scan, bir kez)       │
 │ dup_lookup  : metin MD5 → kaç farklı author          │
 │ kw_lookup   : kw fingerprint → kaç farklı author     │
 │ Her posta JOIN edilir                                │
 │                                                      │
 │ ADIM 3 — Author-Level Aggregate                      │
 │ has_author=True → groupby(author_hash)               │
 │ Her author için istatistik hesaplanır                │
 │                                                      │
 │ ADIM 4 — Join + Imputation                           │
 │ Author feature'ları her posta JOIN edilir            │
 │ has_author=False → SimpleImputer(median)             │
 └──────────────────────────────────────────────────────┘
      │
      ▼
 SimpleImputer(median) → RobustScaler
      │
      ▼
 Isolation Forest
 (15 feature, tüm 5M post)
      │
      ▼
 Clipped Scaling → organiklik skoru [0–1]
 0 = manipülatif, 1 = organik
      │
      ├─────────────────────────┐
      ▼                         ▼
 post_scores.parquet       dashboard.py
 author_scores.csv         (İster 2)
 top_bots.csv
```

---

## Feature Listesi (15 Feature)

POST-LEVEL — 9 feature (gerçek değer, imputation yok):

  text_len                 : orijinal metnin karakter sayısı
                             str.len() ile hesaplanır

  kw_count                 : virgülle ayrılmış keyword sayısı
                             english_keywords.split(",").len() ile hesaplanır

  kw_density               : metindeki keyword yoğunluğu
                             kw_count / (text_len + 1)

  sentiment                : veri setinden direkt alınır [-1, 1]
                             önceden hesaplanmış, dokunulmaz

  sentiment_extreme        : duygu aşırılığı flag'i
                             abs(sentiment) > 0.8 → 1, değilse 0
                             manipülatif içerik genellikle uçlarda

  is_duplicate             : bu metin başka authorlarca da paylaşılmış mı?
                             cross_author_dup_count > 0 → 1, değilse 0
                             count'un yanında binary sinyal olarak güçlü

  has_author               : author bilgisi var mı?
                             author_hash dolu → 1, boş → 0
                             kendisi başlı başına sinyal

  cross_author_dup_count   : bu metnin kaç farklı author tarafından
                             paylaşıldığı — metin hash'i üzerinden hesaplanır
                             §13 bulgusundan: max 572 author aynı metni paylaşmış

  kw_fingerprint_shared    : bu keyword setini kaç farklı author kullanmış
                             top-8 keyword MD5 fingerprint → author sayısı
                             §15 bulgusundan: max 1836 author aynı seti paylaşmış

AUTHOR-LEVEL — 6 feature (has_author=False → median imputation):

  author_posts_per_day     : günlük ortalama post hızı
                             post_count / aktif gün sayısı
                             (max_timestamp - min_timestamp).days + 1

  author_min_interval_sec  : iki post arasındaki minimum süre (saniye)
                             timestamp.diff().min()
                             §12 bulgusundan: 203 author sub-second burst

  author_mean_jaccard      : author'ın kendi postları arasındaki
                             ortalama keyword benzerliği
                             pairwise Jaccard(kw_set_i, kw_set_j).mean()
                             §16 bulgusundan: >0.5 olan 145 author şüpheli

  author_sentiment_std     : author'ın sentiment değişkenliği
                             sentiment.std()
                             çok düşük = hep aynı duygu = bot sinyali

  author_unique_themes     : author'ın kullandığı benzersiz tema sayısı
                             primary_theme.nunique()
                             §8 bulgusundan: 0a92fab3 tüm 15 temayı kullanmış

  author_duplicate_ratio   : author'ın duplicate olan postlarının oranı
                             (cross_author_dup_count > 0 olan post sayısı)
                             / post_count
                             count değil oran — post sayısından bağımsız

---

### Post-Level — 9 Feature
*(5M postun tamamı için, imputation yok)*

| Feature | Açıklama | Türetme |
|---|---|---|
| `text_len` | Karakter sayısı | `str.len()` |
| `kw_count` | Keyword sayısı | `english_keywords.split(",").len()` |
| `kw_density` | Keyword yoğunluğu | `kw_count / (text_len + 1)` |
| `sentiment` | Duygu skoru | Veri setinden direkt |
| `sentiment_extreme` | Aşırı duygu flag'i | `abs(sentiment) > 0.8 → 1` |
| `is_duplicate` | Başka authorlarca paylaşılmış mı? | `cross_author_dup_count > 0 → 1` |
| `has_author` | Author bilgisi var mı? | `author_hash dolu → 1` |
| `cross_author_dup_count` | Bu metni paylaşan farklı author sayısı | `dup_lookup` tablosundan |
| `kw_fingerprint_shared` | Bu keyword setini kullanan farklı author sayısı | `kw_lookup` tablosundan |

! lookup tabloları kayıt edilmeli data/processed altına

### Author-Level — 6 Feature
*(has_author=False → SimpleImputer(median))*

| Feature | Açıklama | Türetme |
|---|---|---|
| `author_posts_per_day` | Günlük ortalama post hızı | `post_count / aktif_gun` |
| `author_min_interval_sec` | İki post arası minimum süre | `timestamp.diff().min()` |
| `author_mean_jaccard` | Kendi postları arasında keyword benzerliği | Pairwise Jaccard ortalaması |
| `author_sentiment_std` | Duygu değişkenliği | `sentiment.std()` |
| `author_unique_themes` | Benzersiz tema sayısı | `primary_theme.nunique()` |
| `author_duplicate_ratio` | Duplicate postların oranı | `dup_post_count / post_count` |

---

## İster 1 — Organiklik Skoru

5M postun tamamı için 15 feature ile IF ile skor üretilir.
Author skoru, o autho'ra ait post skorlarının ortalamasıdır.

```
post_scores.parquet  → her postun organiklik skoru
author_scores.csv    → her author'ın ortalama skoru
top_bots.csv         → en düşük skorlu 1000 hesap
```

---

## İster 2 — Manipülasyon Haritası

IF skorları hazır olunca:

```
post_scores.parquet + orijinal veri
        │
        ▼
dil × platform pivot
ortalama organiklik skoru
        │
        ▼
Plotly ısı haritası → manipulation_map.html
```

Tek görsel, tek dosya. Dash sunucusu gerekmeden açılır.

---

## İster 3 — Inference Pipeline

IF sadece pose text girişine uyum sağlayabilmesi için post_features'tan `has_author` feature'u hariç kalan 8 feature ile tekrardan eğitilir ve bu IF modeli inference için kullanılır. 

```
predict("ham metin")
      │
      ▼
  Post-level feature'lar hesaplanır:
  text_len, kw_count, kw_density,
  sentiment (VADER), sentiment_extreme, is_duplicate,
  cross_author_dup_count (dup_lookup'tan),
  kw_fingerprint_shared (kw_lookup'tan)

      │
      ▼
  scaler.pkl → isolation_forest.pkl
      │
      ▼
  Kural Katmanı
  (deterministik if/else — eşik ihlallerini listeler)
      │
      ▼
  {
    "score": 0.12,
    "verdict": "MANİPÜLATİF",
    "reasons": [
      "572 farklı hesap aynı metni paylaştı",
      "Keyword yoğunluğu: 0.42 (normal ort: 0.08)",
      "Sentiment aşırı uçta: -0.91"
    ]
  }
```

Jüri metni eğitim setinde yoksa `cross_author_dup_count = 0`,
`kw_fingerprint_shared = 0` olur. Reasons listesi bunu açıkça belirtir.
Skor sadece metin örüntüsüne göre hesaplanır.

---

## Kural Katmanı — Çalışma Mantığı

LLM veya doğal dil üretimi yoktur. Tamamen deterministik if/else kuralları:

```python
if cross_author_dup_count > 10:
    reasons.append(f"{cross_author_dup_count} farklı hesap aynı metni paylaştı")

if kw_fingerprint_shared > 100:
    reasons.append(f"Bu keyword seti {kw_fingerprint_shared} hesapta görüldü")

if abs(sentiment) > 0.8:
    reasons.append(f"Sentiment aşırı uçta: {sentiment:.2f}")

if kw_density > eşik:
    reasons.append(f"Keyword yoğunluğu yüksek: {kw_density:.2f}")

if cross_author_dup_count == 0 and kw_fingerprint_shared == 0:
    reasons.append("Eğitim setinde görülmemiş — koordinasyon sinyali yok")
```

---

## Validasyon

- STABİLİTE   → IF'i farklı random seed'lerle çalıştır
               Her seferinde aynı postlar yüksek skor alıyor mu?
               Alıyorsa model tutarlı demektir

- SPOT-CHECK  → En manipülatif işaretlenen 50 hesabı
               elle aç, postlarına bak
               Gerçekten şüpheli görünüyor mu?
               Bu bizim "ground truth olmadan doğrulama" yöntemimiz

---

## Çıktılar

| Çıktı | İster | Açıklama |
|---|---|---|
| `post_scores.parquet` | 1 | 5M postun tamamına organiklik skoru |
| `author_scores.csv` | 1 | Her hesabın ortalama organiklik skoru |
| `top_bots.csv` | 1 | En manipülatif 1000 hesap |
| `manipulation_map.html` | 2 | Dil × platform ısı haritası |
| `predict("metin")` | 3 | Skor + deterministik gerekçe listesi |