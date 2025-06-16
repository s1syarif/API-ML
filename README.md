# ML2 - Machine Learning API & Utilities

ML2 adalah modul mandiri yang berisi API dan utilitas untuk berbagai kebutuhan machine learning, seperti OCR, rekomendasi gizi, dan prediksi penyakit. ML2 dapat dijalankan secara terpisah dari backend utama.

## Struktur Folder

- `ml_main.py` : Entry point FastAPI untuk seluruh API ML2 (OCR, rekomendasi, dsb)
- `model/`   : Semua file model, scaler, weights, dan mapping
    - `OCR/`      : Model & mapping untuk OCR (misal: `char_to_num.json`, `recognizer_finetuned_weights.h5`)
    - `dieses/`   : Model & scaler untuk prediksi penyakit (misal: `best_multilabel_keras_model.keras`)
    - `recom/`    : Model rekomendasi gizi (misal: `rekomendasi_gizi.joblib`)
- `utils/`   : Semua fungsi utilitas (OCR, rekomendasi, dsb) yang dipanggil dari API utama
- `__init__.py` : Menandai ML2 sebagai package Python

## Cara Menjalankan

1. Pastikan semua dependensi sudah terinstall (lihat requirements di project utama).
2. Jalankan API ML2 dengan FastAPI/uvicorn:
   ```
   uvicorn ML2.ml_main:app --reload
   ```
3. Endpoint utama:
   - `/ocr/` : OCR Gizi
   - `/recommend` : Rekomendasi Gizi
   - `/health-score` : Skor Kesehatan
   - `/predict-dieses` : Prediksi Penyakit

## Catatan
- Semua model dan mapping harus ada di folder `model/`.
- Semua fungsi utilitas ada di `utils/`.
- Hapus file lama/duplikat/deprecated di luar struktur ini.