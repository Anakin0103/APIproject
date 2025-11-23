# app_fast.py

import time
import numpy as np
import pandas as pd
import librosa
import joblib
import gradio as gr

# =========================
# 1. Load artifacts once
# =========================

FEATURE_CSV = "gtzan_features.csv"
SCALER_PATH = "saved_models/scaler.pkl"
MODEL_PATH = "saved_models/svm.pkl"
LABEL_ENCODER_PATH = "saved_models/label_encoder.pkl"

print("[INIT] Loading artifacts...")
t0 = time.time()

# 读取特征列（不包含 label 和 filename）
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


# =========================
# 2. Feature extraction from waveform (no librosa.load!)
# =========================

def extract_features_from_array(
    y,
    sr,
    n_mfcc=20,
    n_fft=2048,
    hop_length=512,
    max_duration=10.0,
):
    """
    从内存中的波形 y, sr 提取特征。
    - 不再调用 librosa.load
    - 只用前 max_duration 秒
    - tempo 特征占位：设为 0.0，不做真实计算
    """
    t0 = time.time()

    # 转成 numpy float32
    y = np.asarray(y, dtype=np.float32)

    # 如果是立体声，转成单声道
    if y.ndim > 1:
        # Gradio 有时返回 shape: (samples, channels)
        y = np.mean(y, axis=1)

    # 只用前 max_duration 秒
    max_samples = int(sr * max_duration)
    if len(y) > max_samples:
        y = y[:max_samples]

    if len(y) == 0:
        # 极端情况：空音频
        raise ValueError("Empty audio array after trimming.")

    print(f"[FEAT] Audio length (array): {len(y)/sr:.2f} s")

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

    # ---- Tempo: 禁用真实计算，只做占位 ----
    t7 = time.time()
    tempo = 0.0  # 占位常数，如果训练时没有 tempo 列，这一项会被忽略
    feats["tempo"] = float(tempo)
    print(f"[FEAT] Tempo (disabled) done in {time.time() - t7:.4f} s")

    print(f"[FEAT] TOTAL feature extraction time: {time.time() - t0:.2f} s")
    return feats


# =========================
# 3. Prediction function for Gradio
# =========================

def predict_genre(audio):
    """
    audio: Gradio Audio(type='numpy') 给的输入。
    在不同版本的 gradio 中，audio 可能是：
      - (sr, y) 的 tuple
      - dict: {'sample_rate': sr, 'data': y}
    这里两种都兼容。
    """
    if audio is None:
        return "Please upload an audio file."

    t_total = time.time()
    print("\n[REQ] New prediction request")

    # 兼容不同 gradio 版本的返回格式
    if isinstance(audio, dict):
        sr = audio.get("sample_rate")
        y = audio.get("data")
    else:
        # 假设是 (sr, y)
        sr, y = audio

    # 1) 特征提取
    t_feat = time.time()
    feats_dict = extract_features_from_array(y, sr)
    print(f"[TIME] Feature extraction: {time.time() - t_feat:.2f} s")

    # 2) 构造特征向量（按训练时列顺序）
    try:
        x = np.array([[feats_dict[col] for col in feature_columns]], dtype=np.float32)
    except KeyError as e:
        print("[ERROR] Missing feature:", e)
        return f"Feature mismatch: missing {e}. Check feature_columns vs. feats_dict."

    # 3) 用 DataFrame 包一下，避免 sklearn 的 feature name warning
    x_df = pd.DataFrame(x, columns=feature_columns)

    t_scaler = time.time()
    x_scaled = scaler.transform(x_df)
    print(f"[TIME] Scaler transform: {time.time() - t_scaler:.4f} s")

    # 4) 预测
    t_pred = time.time()
    y_pred = model.predict(x_scaled)
    print(f"[TIME] Model.predict: {time.time() - t_pred:.4f} s")

    genre = label_encoder.inverse_transform(y_pred)[0]
    print(f"[REQ] Total time: {time.time() - t_total:.2f} s, Result: {genre}")
    return genre


# =========================
# 4. Gradio UI
# =========================

demo = gr.Interface(
    fn=predict_genre,
    inputs=gr.Audio(type="numpy", label="Upload a song (~10s–30s)"),
    outputs=gr.Label(label="Predicted genre"),
    title="Music Genre Classification (Fast SVM Demo)",
    description=(
        "Upload an audio clip. The app extracts MFCC + spectral features "
        "and uses your trained SVM model to predict the genre."
    ),
)

if __name__ == "__main__":
    demo.launch()
