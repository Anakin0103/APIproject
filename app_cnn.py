import time
import numpy as np
import librosa
import gradio as gr
import tensorflow as tf


MODEL_PATH = "saved_models/cnn_genre_model.h5" 
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_DURATION = 10.0
N_FRAMES = 128        

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]



print("[INIT] Loading CNN model...")
t0 = time.time()
model = tf.keras.models.load_model(MODEL_PATH)
print(f"[INIT] Model loaded in {time.time() - t0:.2f} s")
print("[INIT] Model input shape:", model.input_shape)
print("[INIT] Model output shape:", model.output_shape)




def audio_to_logmel(y, sr):
    y = np.asarray(y, dtype=np.float32)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    max_samples = int(sr * MAX_DURATION)
    if len(y) > max_samples:
        y = y[:max_samples]

    if len(y) == 0:
        raise ValueError("Empty audio array after trimming.")

    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    log_S = librosa.power_to_db(S, ref=np.max)


    T = log_S.shape[1]

    if T < N_FRAMES:
        pad_width = N_FRAMES - T
        log_S = np.pad(log_S, ((0, 0), (0, pad_width)), mode="constant")
    elif T > N_FRAMES:
        log_S = log_S[:, :N_FRAMES]

    min_val = log_S.min()
    max_val = log_S.max()
    if max_val > min_val:
        log_S = (log_S - min_val) / (max_val - min_val + 1e-8)
    else:
        log_S = np.zeros_like(log_S)

    return log_S 




def predict_genre_cnn(audio):

    if audio is None:
        return "Please upload an audio file."

    t_total = time.time()
    print("\n[REQ] New CNN prediction request")

    if isinstance(audio, dict):
        sr = audio.get("sample_rate")
        y = audio.get("data")
    else:
        sr, y = audio

    t_feat = time.time()
    logmel = audio_to_logmel(y, sr)  
    print(f"[TIME] Log-mel extraction: {time.time() - t_feat:.2f} s")


    x = logmel[np.newaxis, ..., np.newaxis].astype(np.float32)  

    t_pred = time.time()
    preds = model.predict(x)
    print(f"[TIME] Model.predict: {time.time() - t_pred:.2f} s")

    preds = preds[0]  
    top_idx = int(np.argmax(preds))
    top_genre = GENRES[top_idx]
    top_prob = float(preds[top_idx])

    print(
        f"[REQ] Total time: {time.time() - t_total:.2f} s, "
        f"Result: {top_genre} (p={top_prob:.3f})"
    )

    probs_dict = {g: float(p) for g, p in zip(GENRES, preds)}
    return probs_dict




demo = gr.Interface(
    fn=predict_genre_cnn,
    inputs=gr.Audio(type="numpy", label="Upload a song (~10s–30s)"),
    outputs=gr.Label(num_top_classes=3, label="Predicted genre (CNN)"),
    title="Music Genre Classification (CNN Model)",
    description=(
        "Upload an audio clip. The app computes a log-mel spectrogram and "
        "uses a trained CNN model to predict the music genre."
    ),
)

if __name__ == "__main__":
    demo.launch()
