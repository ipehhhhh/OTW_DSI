import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import numpy as np
import pandas as pd 

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="K-Means Clustering API",
    description="API untuk melakukan inferensi K-Means clustering",
    version="1.0.0"
)

# Variabel global untuk menyimpan model dan preprocessor
kmeans_model = None
preprocessor = None

# Tentukan path ke folder models
MODELS_DIR = "models"
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")

# --- Kelas Pydantic untuk Validasi Input Data ---
class DataInput(BaseModel):
    usia: float
    tingkat_variasi_pekerjaan: float
    suku_bunga_euribor_3bln: float
    indeks_kepercayaan_konsumen: float
    gagal_bayar_sebelumnya: str 

# --- Event Handler: Load Model saat Aplikasi Dimulai ---
@app.on_event("startup")
async def load_models():
    global kmeans_model, preprocessor
    try:
        print(f"Mencoba memuat model K-Means dari: {KMEANS_MODEL_PATH}")
        kmeans_model = joblib.load(KMEANS_MODEL_PATH)
        print("Model K-Means berhasil dimuat.")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Model K-Means tidak ditemukan di {KMEANS_MODEL_PATH}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memuat model K-Means: {e}")

    try:
        print(f"Mencoba memuat preprocessor dari: {PREPROCESSOR_PATH}")
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Preprocessor berhasil dimuat.")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Preprocessor tidak ditemukan di {PREPROCESSOR_PATH}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memuat preprocessor: {e}")

# --- Endpoint Utama untuk Inferensi ---
@app.post("/predict_cluster/")
async def predict_cluster(data: DataInput):
    if kmeans_model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model atau Preprocessor belum dimuat. Coba lagi nanti.")

    # Mengambil data dari objek Pydantic dan mengkonversinya ke dictionary
    input_dict = data.model_dump()

    # --- BAGIAN KRUSIAL: Menentukan NAMA KOLOM untuk DataFrame ---
    column_names = [
        "usia",
        "tingkat_variasi_pekerjaan",
        "suku_bunga_euribor_3bln",
        "indeks_kepercayaan_konsumen",
        "gagal_bayar_sebelumnya"
    ]

    # Mengkonversi dictionary input menjadi Pandas DataFrame
    # ColumnTransformer membutuhkan DataFrame jika Anda menentukan kolom dengan string.
    input_df = pd.DataFrame([input_dict], columns=column_names)

    try:
        # Lakukan preprocessing pada DataFrame
        processed_data = preprocessor.transform(input_df) # Menggunakan DataFrame sebagai input

        # Lakukan prediksi cluster dengan model K-Means
        cluster_prediction = kmeans_model.predict(processed_data)[0] # Ambil hasil prediksi pertama

        return {"cluster_id": int(cluster_prediction)} # Pastikan mengembalikan int
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses data atau memprediksi: {e}")

# --- Endpoint Contoh (Opsional) ---
@app.get("/")
async def root():
    return {"message": "Selamat datang di K-Means Clustering API!"}