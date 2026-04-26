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

Pipeline sırası: `data_loader → features → model → ensemble → inference/dashboard`

Detay için `ARCHITECTURE.md` dosyasına bak.

---

## Key Conventions

- **Analiz birimi author'dur, post değil.** Postlar feature engineering için kullanılır, model author-level aggregate üzerinde çalışır.
- **Label yoktur.** Bu tamamen unsupervised bir problem. Hiçbir yere `y_train`, `accuracy`, `f1` yazma.
- **RAM:** Tam parquet ~4-8 GB açılır. EDA için `load_sample(frac=0.1)` kullan. `data_loader.py`'deki dtype optimizasyonu zorunlu.
- **Model kaydet/yükle:** Sunum sırasında model yeniden eğitilmez. `models/*.pkl` yüklenir.
- **Temporal split:** Holdout set en yeni tarihler (%20). Shuffle yapma, tarih sırasını koru.

---

## Critical Notes

- `data/datathonFINAL.parquet` ve `models/*.pkl` `.gitignore`'a ekli olmalı.
- Inference pipeline (`src/inference.py`) sunum anında tek başına çalışabilir olmalı — dış bağımlılık minimumda tut.
- Jüri unseen metin verecek: `predict("metin")` çağrısı skor + gerekçe dönmeli.
- Dashboard sunum öncesinde `outputs/manipulation_map.html` olarak export edilmeli (Dash sunucusu gerekmeden açılabilsin).