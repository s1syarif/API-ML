# recog.py - Cleaned and organized

# === Standard Library Imports ===
import os
import sys
import io
import json
import re

# === Third-Party Imports ===
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_ocr
from PIL import Image
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List

# === Path Setup ===
SCRIPT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
OCR_DIR = os.path.join(SCRIPT_DIR, "model", "OCR")
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)
sys.path.append('.')

# === Import Local Modules ===
from ML2.utils.ocr_utils import crop_and_predict_words, extract_nutrition_json
from ML2.utils.rekomendasi_utils import rekomendasi_logic
from ML2.utils.health_score_utils import infer_health_score_custom
from ML2.utils.disease_predict_utils import preprocess_new_data

# === FastAPI App ===
app = FastAPI()

# === Constants and Model Paths ===
WEIGHTS_PATH = os.path.join(OCR_DIR, "recognizer_finetuned_weights.h5")
CHAR2NUM_JSON = os.path.join(OCR_DIR, "char_to_num.json")
NUM2CHAR_JSON = os.path.join(OCR_DIR, "num_to_char.json")
IMG_HEIGHT = 31
IMG_WIDTH = 200
MODEL_PATH_REKOM = os.path.join(SCRIPT_DIR, 'model', 'recom', 'rekomendasi_gizi.joblib')
MODEL_PATH_DIESES = os.path.join(SCRIPT_DIR, 'model', 'dieses', 'best_multilabel_keras_model.keras')
SCALER_PATH = os.path.join(SCRIPT_DIR, 'model', 'dieses', 'scaler.joblib')
MLB_PATH = os.path.join(SCRIPT_DIR, 'model', 'dieses', 'mlb_binarizer.joblib')
FEATURES_PATH = os.path.join(SCRIPT_DIR, 'model', 'dieses', 'feature_columns.joblib')

# === Load OCR Mappings === 
with open(NUM2CHAR_JSON, "r") as f:
    num_to_char = json.load(f)
with open(CHAR2NUM_JSON, "r") as f:
    char_to_num = json.load(f)
num_to_char = {int(k): v for k, v in num_to_char.items()}

# === Load OCR Models ===
recognizer = keras_ocr.recognition.Recognizer()
recognizer.model.load_weights(WEIGHTS_PATH)
detector = keras_ocr.detection.Detector()

# === Load Recommendation Model ===
df_model = joblib.load(MODEL_PATH_REKOM)

# === API Models ===
class KonsumsiRequest(BaseModel):
    konsumsi: Dict[str, float]
    target_harian: Dict[str, float]  # WAJIB dikirim user

# === OCR Endpoint ===
@app.post("/ocr/")
async def ocr_from_image(file: UploadFile = File(...)):
    print("[API CALL] /ocr/ endpoint dipanggil", flush=True)
    print("[INFO] Ada request OCR dari client.", flush=True)
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        texts = crop_and_predict_words(image_np, recognizer, num_to_char, detector, IMG_HEIGHT, IMG_WIDTH)
        print("Hasil OCR:", texts)
        result = extract_nutrition_json(texts)
        print("Hasil ekstraksi OCR:", result)
        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# === Recommendation Endpoint ===
@app.post('/recommend')
def rekomendasi_gizi(data: KonsumsiRequest):
    print("[API CALL] /recommend endpoint dipanggil", flush=True)
    try:
        target_harian = data.target_harian
        konsumsi = data.konsumsi
        hasil = rekomendasi_logic(target_harian, konsumsi, df_model)
        fokus_kurang = hasil['gizi_fokus']
        if fokus_kurang:
            print("[INFO] Gizi yang masih kurang dari 80% target:")
            for k in fokus_kurang:
                print(f"  - {k}")
        else:
            print("[INFO] Semua kebutuhan gizi minimal 80% sudah terpenuhi.")
        return hasil
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'❌ Terjadi kesalahan: {e}')

# === Health Score Section ===
class HealthScoreRequest(BaseModel):
    energi: float
    protein: float
    lemak_total: float
    karbohidrat: float
    serat: float
    gula: float
    garam: float
    target_energi: float
    target_protein: float
    target_lemak_total: float
    target_karbohidrat: float
    target_serat: float
    target_gula: float
    target_garam: float

class HealthScoreResponse(BaseModel):
    score: int

@app.post("/health-score", response_model=HealthScoreResponse)
def get_health_score(req: HealthScoreRequest):
    print("[API CALL] /health-score endpoint dipanggil", flush=True)
    try:
        score = infer_health_score_custom(
            energi=req.energi,
            protein=req.protein,
            lemak_total=req.lemak_total,
            karbohidrat=req.karbohidrat,
            serat=req.serat,
            gula=req.gula,
            garam=req.garam,
            target_energi=req.target_energi,
            target_protein=req.target_protein,
            target_lemak_total=req.target_lemak_total,
            target_karbohidrat=req.target_karbohidrat,
            target_serat=req.target_serat,
            target_gula=req.target_gula,
            target_garam=req.target_garam
        )
        print(f"Input konsumsi: Energi={req.energi}, Protein={req.protein}, Lemak Total={req.lemak_total}, Karbohidrat={req.karbohidrat}, Serat={req.serat}, Gula={req.gula}, Garam={req.garam}")
        print(f"Target harian: Energi={req.target_energi}, Protein={req.target_protein}, Lemak Total={req.target_lemak_total}, Karbohidrat={req.target_karbohidrat}, Serat={req.target_serat}, Gula={req.target_gula}, Garam={req.target_garam}")
        print(f"Skor hasil: {score}")
        return {"score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'❌ Terjadi kesalahan: {e}')

# === Disease Prediction Section ===
model = keras.models.load_model(MODEL_PATH_DIESES)
scaler = joblib.load(SCALER_PATH)
mlb = joblib.load(MLB_PATH)
feature_columns = joblib.load(FEATURES_PATH)

class PredictRequest(BaseModel):
    Ages: List[float]
    Gender: List[str]
    Height: List[float]
    Weight: List[float]
    Protein: List[float]
    Sugar: List[float]
    Sodium: List[float]
    Calories: List[float]
    Carbohydrates: List[float]
    Fiber: List[float]
    Fat: List[float]

class PredictResponse(BaseModel):
    probabilities: List[List[float]]
    binary: List[List[int]]
    labels: List[List[str]]

@app.post("/predict-dieses")
def predict_api(req: PredictRequest):
    print("[API CALL] /predict-dieses endpoint dipanggil", flush=True)
    try:
        df_new = pd.DataFrame(req.dict())
        X_new = preprocess_new_data(df_new, feature_columns, scaler)
        proba = model.predict(X_new)
        y_pred = (proba > 0.5).astype(int)
        labels = mlb.inverse_transform(y_pred)
        print("Label terprediksi:", [list(l) if l else ['Normal'] for l in labels])
        return {"labels": [list(l) if l else ['Normal'] for l in labels]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'❌ Terjadi kesalahan: {e}')