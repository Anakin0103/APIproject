import time
import numpy as np
import pandas as pd
import librosa
import joblib
import gradio as gr


FEATURE_CSV = "gtzan_features.csv"
SCALER_PATH = "saved_models/scaler.pkl"
MODEL_PATH = "saved_models/svm.pkl"
LABEL_ENCODER_PATH = "saved_models/label_encoder.pkl"

print("[INIT] Loading artifacts...")
t0 = time.time()

feature_columns = (
    pd.read_csv(FEATURE_CSV)
    .drop(columns=["label", "filename"], errors="ignore")
    .columns
    .tolist()
)

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

print(f"[INIT] Artifacts loaded in {time.time() - t0:.2f} s")
print(f"[INIT] Feature columns: {len(feature_columns)} features")



def extract_features_from_file(
    file_path,
    sr=22050,
    n_mfcc=20,
    n_fft=2048,
    hop_length=512,
    max_duration=10.0,  
):
    t0 = time.time()
    print(f"[FEAT] Loading audio (first {max_duration}s) from {file_path}")


    y, sr = librosa.load(
        file_path,
        sr=sr,
        mono=True,
        duration=max_duration,   
        res_type="kaiser_fast",  
    )

    print(f"[FEAT] Audio length after load: {len(y)/sr:.2f} s")

    feats = {}

    # ---- MFCC ----
    t1 = time.time()
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    for i in range(n_mfcc):
        coeff = mfcc[i]
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(coeff))
        feats[f"mfcc_{i+1}_std"] = float(np.std(coeff))
    print(f"[FEAT] MFCC done in {time.time() - t1:.2f} s")

    # ---- Zero Crossing Rate ----
    t2 = time.time()
    zcr = librosa.feature.zero_crossing_rate(y)
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"] = float(np.std(zcr))
    print(f"[FEAT] ZCR done in {time.time() - t2:.2f} s")

    # ---- Spectral Centroid ----
    t3 = time.time()
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    feats["centroid_mean"] = float(np.mean(centroid))
    feats["centroid_std"] = float(np.std(centroid))
    print(f"[FEAT] Centroid done in {time.time() - t3:.2f} s")

    # ---- Spectral Bandwidth ----
    t4 = time.time()
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    feats["bandwidth_mean"] = float(np.mean(bandwidth))
    feats["bandwidth_std"] = float(np.std(bandwidth))
    print(f"[FEAT] Bandwidth done in {time.time() - t4:.2f} s")

    # ---- Spectral Rolloff ----
    t5 = time.time()
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    feats["rolloff_mean"] = float(np.mean(rolloff))
    feats["rolloff_std"] = float(np.std(rolloff))
    print(f"[FEAT] Rolloff done in {time.time() - t5:.2f} s")

    # ---- RMS ----
    t6 = time.time()
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"] = float(np.std(rms))
    print(f"[FEAT] RMS done in {time.time() - t6:.2f} s")

    # ---- Tempo ----
    t7 = time.time()
    tempo = 0.0

    feats["tempo"] = float(tempo)
    print(f"[FEAT] Tempo done in {time.time() - t7:.4f} s")

    print(f"[FEAT] TOTAL feature extraction time: {time.time() - t0:.2f} s")
    return feats



def predict_genre(audio_file):
    t_total = time.time()

    if audio_file is None:
        return "Please upload an audio file."

    print(f"\n[REQ] New prediction request: {audio_file}")

    feats_dict = extract_features_from_file(audio_file)

    try:
        x = np.array([[feats_dict[col] for col in feature_columns]], dtype=np.float32)
    except KeyError as e:
        print("[ERROR] Missing feature:", e)
        return f"Feature mismatch, missing {e}. Did you train and deploy with the same feature set?"

    x_scaled = scaler.transform(x)

    y_pred = model.predict(x_scaled)
    genre = label_encoder.inverse_transform(y_pred)[0]

    print(f"[REQ] Total processing time: {time.time() - t_total:.2f} s")
    return genre



demo = gr.Interface(
    fn=predict_genre,
    inputs=gr.Audio(type="filepath", label="Upload a song (~10s–30s)"),
    outputs=gr.Label(label="Predicted genre"),
    title="Music Genre Classification (SVM Version)",
    description="This demo uses the first 10 seconds of the uploaded audio and a traditional ML model (SVM).",
)

if __name__ == "__main__":
    demo.launch()
