from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import librosa
import io

app = FastAPI(
    title="Industrial Pump Anomaly Detection API",
    description="API designed to detect mechanical anomalies in water pumps using acoustics analysis."
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
try:
    clf = joblib.load('optimized_isolation_forest.joblib')
except Exception as e:
    raise RuntimeError(f"Failed to load model file 'optimized_isolation_forest.joblib': {str(e)}")

@app.get("/")
def read_root():
    return {"status": "API is running successfully! Send audio files to /predict"}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    # Check file extension
    if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a WAV, MP3, or OGG file.")

    try:
        # 1. Read file bytes
        audio_bytes = await file.read()

        # 2. Load audio with librosa
        SR = 16000
        N_SAMPLES = SR * 10
        
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR)
        y = y[:N_SAMPLES]
        if len(y) < N_SAMPLES:
            y = np.pad(y, (0, N_SAMPLES - len(y)))

        # 3. Extract Advanced DSP Features
        mf = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)
        mf_mean = mf.mean(axis=1)
        mf_std = mf.std(axis=1)

        centroid = librosa.feature.spectral_centroid(y=y, sr=SR)
        centroid_mean = centroid.mean()
        centroid_std = centroid.std()

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR)
        rolloff_mean = rolloff.mean()
        rolloff_std = rolloff.std()

        rms = librosa.feature.rms(y=y)
        rms_mean = rms.mean()
        rms_std = rms.std()

        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = zcr.mean()
        zcr_std = zcr.std()

        # Combine features
        feat_vector = np.r_[
            mf_mean, mf_std,
            centroid_mean, centroid_std,
            rolloff_mean, rolloff_std,
            rms_mean, rms_std,
            zcr_mean, zcr_std
        ].reshape(1, -1)

        # 4. Predict anomaly score
        score = float(-clf.score_samples(feat_vector)[0])
        
        # Detection Threshold
        threshold = -0.35
        verdict = "ABNORMAL" if score > threshold else "NORMAL"

        return {
            "filename": file.filename,
            "anomaly_score": round(score, 5),
            "threshold": threshold,
            "verdict": verdict
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during audio processing: {str(e)}")