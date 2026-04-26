# DataLeague Datathon — Sosyal Medya Manipülasyon Tespit Sistemi

Unsupervised anomaly detection üzerine kurulu bot ve koordineli hesap tespiti projesi.

## Kurulum

```bash
uv sync
```

## Çalıştırma

```bash
uv run jupyter notebook        # EDA ve deney
uv run python src/features.py  # Feature engineering
uv run python src/model.py     # Model eğitimi
uv run python src/dashboard.py # Dashboard (localhost:8050)
uv run python src/inference.py # Inference testi
```

## Proje Yapısı

- **data/** — Ham parquet dosyası (gitignore'da)
- **notebooks/** — EDA ve deney notebookları
- **src/** — Production kod
- **models/** — Eğitilmiş model dosyaları (gitignore'da)
- **outputs/** — Çıktı dosyaları

## Dokümantasyon

- [ARCHITECTURE.md](ARCHITECTURE.md) — Mimari ve pipeline detayları
- [CLAUDE.md](CLAUDE.md) — Komutlar ve kurallar
