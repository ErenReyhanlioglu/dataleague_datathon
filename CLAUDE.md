# CLAUDE.md
# DataLeague Datathon — Sosyal Medya Manipülasyon Tespit Sistemi
# Unsupervised anomaly detection on 5M social media posts.

---

## Commands

```bash
uv sync                        # bağımlılıkları kur
uv run jupyter notebook        # notebook başlat
uv run python src/features.py  # feature engineering çalıştır
uv run python src/model.py     # modelleri eğit, models/ altına kaydet
uv run python src/dashboard.py # dashboard → localhost:8050
uv run python src/inference.py # canlı inference testi
```

---

## Project Structure

```
data/          → ham parquet (git'e ekleme, .gitignore'da)
notebooks/     → EDA ve deney (01_eda, 02_features, 03_modeling)
src/           → production kod (features, model, ensemble, inference, dashboard)
models/        → eğitilmiş pkl dosyaları (git'e ekleme)
outputs/       → dashboard html, skor csv'leri
```

---

## Architecture

Pipeline sırası: `data_loader → features → model → ensemble → inference / dashboard`

Detay için `ARCHITECTURE.md` dosyasına bak.

---

## Key Conventions

- **Analiz birimi author'dur, post değil.** Model author-level aggregate üzerinde çalışır.
- **Label yoktur.** Tamamen unsupervised. `y_train`, `accuracy`, `f1` kullanma.
- **Veri setindeki sütunlar önceden hesaplanmıştır.** `sentiment`, `main_emotion`,
  `primary_theme`, `english_keywords` ham metin işleme ürünü değil, hazır feature'dır.
  Bu sütunları üretme, doğrudan kullan.
- **Inference ham metin alır.** Jüri author_hash değil metin verecek. `predict()`
  fonksiyonu bu metinden sentiment/keyword/language hesaplayarak çalışır.
  Author-level feature yoktur, bu bilinçli bir kısıt.
- **RAM:** Parquet RAM'de ~4-8 GB açılır. EDA için `load_sample(frac=0.1)` kullan.
  HDBSCAN'i ham post'a değil, author aggregate'e uygula.
- **Model kaydet/yükle:** Sunum sırasında model eğitilmez. `models/*.pkl` yüklenir.

---

## Critical Notes

- `data/` ve `models/` klasörleri `.gitignore`'da olmalı.
- `inference.py` bağımsız çalışabilmeli — sunum anında tek dosya yeterli olmalı.
- Dashboard sunucu gerektirmeden açılabilmesi için `outputs/manipulation_map.html`
  olarak export edilmeli.