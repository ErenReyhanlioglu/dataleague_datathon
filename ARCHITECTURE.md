# ARCHITECTURE.md
# Sosyal Medya Manipülasyon Tespit Sistemi
# DataLeague Datathon — Unsupervised Anomaly Detection
# =====================================================

---

## Proje Yapısı

```
datathon/
├── ARCHITECTURE.md
├── pyproject.toml
├── .python-version
│
├── data/
│   └── datathonFINAL.parquet
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   ├── ensemble.py
│   ├── inference.py
│   └── dashboard.py
│
├── models/
│   ├── isolation_forest.pkl
│   ├── hdbscan_model.pkl
│   └── scaler.pkl
│
└── outputs/
    ├── manipulation_map.html
    ├── author_scores.csv
    └── top_bots.csv
```

---

## Temel Varsayım

Etiket (label) yoktur. Problemin çözümü şu varsayıma dayanır:

- Gerçek kullanıcılar davranışlarında gürültülü, çeşitli ve tutarsızdır.
- Bot ve koordineli hesaplar ise düzenli, tekrarlı ve senkronize hareket eder.

Bu fark, verinin kendi içindeki istatistiksel sapmalardan çıkarılır.

---

## Pipeline Akışı

```
Ham Veri (5M post, post bazlı)
        │
        ▼
   Veri Temizliği
        │
        ▼
  Post-Level Özellikler
  (her post için metin, zaman, duygu bazlı ham özellikler)
        │
        ▼
  Author-Level Aggregation          ← işin özü burada
  (5M post → N unique author)
        │
        ▼
    ┌───┴───┐
    │       │
    ▼       ▼
  Model A  Model B
  (IF)     (HDBSCAN)
    │       │
    └───┬───┘
        │
        ▼
  Kural Katmanı (yorumlanabilirlik)
        │
        ▼
  Organiklik Skoru [0 – 1]
        │
        ▼
  Dashboard + Inference API
```

---

## Katmanların Amacı

### Veri Temizliği
Ham gerçek dünya verisidir. Boş metinler, aralık dışı sentiment değerleri,
parse edilemeyen tarihler ve duplicate postlar bu aşamada temizlenir.
Duplicate tespiti aynı zamanda güçlü bir bot sinyalidir.

### Post-Level Özellikler
Her gönderiden metin uzunluğu, anahtar kelime yoğunluğu, paylaşım saati
gibi ham özellikler çıkarılır. Bu özellikler tek başına anlam taşımaz;
asıl değerleri aggregation sonrasında ortaya çıkar.

### Author-Level Aggregation
Projenin çekirdeği. Her hesabın tüm gönderileri özetlenerek davranışsal
profili çıkarılır: ne hızla paylaşıyor, ne zaman aktif, duygusu ne kadar
değişken, aynı içerikleri tekrar ediyor mu, tek platformda mı hapsolmuş?

### Model A — Isolation Forest
"Bu hesap tek başına anormal mi?" sorusunu yanıtlar.
Bireysel davranışsal sapmaları tespit eder.
Her author için 0-1 arası anomali skoru üretir.
Label gerektirmez, tamamen denetimsizdir.

### Model B — HDBSCAN
"Bu hesaplar birlikte koordineli hareket ediyor mu?" sorusunu yanıtlar.
Isolation Forest'ın göremediği şeyi yakalar: tek başına normal görünen
ama toplu hareket eden hesap ağlarını küme olarak tespit eder.

### Kural Katmanı
Model kararlarını insan diline çevirir.
"Neden manipülatif?" sorusuna kural bazlı, somut gerekçeler üretir.
Jüri sunumu için kritiktir.

### Organiklik Skoru
Üç sinyalin (IF anomali skoru, ağ üyeliği, kural ihlalleri) ağırlıklı
birleşiminden üretilen tek bir değer. 0 manipülatif, 1 organik anlamına gelir.

---

## Validasyon Yaklaşımı

Label olmadığı için klasik train/test accuracy ölçülmez. Bunun yerine:

- **Temporal split:** Model eski veriye fit edilir, yeni veri hiç görülmez.
  Sunum anındaki "unseen" metin bu holdout setinden gelir.
- **Stabilite kontrolü:** Farklı örneklemlerde aynı hesaplar tutarlı
  biçimde anomali olarak işaretleniyor mu?
- **Küme tutarlılığı:** Aynı HDBSCAN kümesindeki hesaplar gerçekten
  aynı içerikleri mi paylaşıyor?
- **Manuel spot-check:** En yüksek bot skoru alan 50 hesabın gönderileri
  elle incelenerek model kararı mantık testine tabi tutulur.

---

## Çıktılar

| Çıktı | Açıklama |
|---|---|
| `author_scores.csv` | Her hesabın organiklik skoru |
| `manipulation_map.html` | Dil × platform ısı haritası, zaman serisi, küme ağı |
| `inference.py → predict()` | Tek metin alır, anlık karar + gerekçe döner |

---