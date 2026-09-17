import os
import numpy as np
import librosa
import tensorflow as tf


# ==============================
# SETTINGS
# ==============================

MODEL_PATH = "voice_cloning_detection_model.keras"

SAMPLE_RATE = 16000
DURATION = 2
N_SAMPLES = SAMPLE_RATE * DURATION

N_MFCC = 40
MAX_FRAMES = 126

THRESHOLD = 0.4


# ==============================
# LOAD MODEL
# ==============================

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Model input shape:", model.input_shape)


# ==============================
# SAME PREPROCESSING AS TRAINING
# ==============================

def extract_features(audio_path):

    try:
        # Load audio
        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        # Normalize amplitude
        max_amplitude = np.max(np.abs(audio))

        if max_amplitude > 0:
            audio = audio / max_amplitude

        # Make exactly 2 seconds
        if len(audio) < N_SAMPLES:
            audio = np.pad(
                audio,
                (0, N_SAMPLES - len(audio)),
                mode="constant"
            )
        else:
            audio = audio[:N_SAMPLES]

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC
        )

        # Make exactly 126 frames
        if mfcc.shape[1] < MAX_FRAMES:

            mfcc = np.pad(
                mfcc,
                (
                    (0, 0),
                    (0, MAX_FRAMES - mfcc.shape[1])
                ),
                mode="constant"
            )

        else:
            mfcc = mfcc[:, :MAX_FRAMES]

        # Standardize MFCC exactly as training
        mfcc = (
            mfcc - np.mean(mfcc)
        ) / (np.std(mfcc) + 1e-8)

        return mfcc.astype(np.float32)

    except Exception as e:

        print("Error processing audio:")
        print(e)

        return None


# ==============================
# PREDICT AUDIO
# ==============================

def predict_audio(audio_file):

    print("\nTesting file:")
    print(audio_file)

    features = extract_features(audio_file)

    if features is None:
        return

    # Add channel dimension
    X = np.expand_dims(features, axis=-1)

    # Add batch dimension
    X = np.expand_dims(X, axis=0)

    print("Input shape:", X.shape)

    # Prediction
    prediction = model.predict(X, verbose=0)

    score = float(prediction[0][0])

    print("Raw model output:", prediction)
    print("Score:", score)

    # Label mapping from training:
    # 0 = Real Voice
    # 1 = AI Voice

    if score >= THRESHOLD:

        result = "AI-GENERATED VOICE"
        confidence = score * 100

    else:

        result = "REAL VOICE"
        confidence = (1 - score) * 100

    print("\n==============================")
    print("Detection Result :", result)
    print("Detection Score  :", f"{confidence:.2f}%")
    print("==============================")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    audio_path = r"C:\Users\User\Downloads\voice cloning\real_samples/121_121726_000004_000003.wav"

    if not os.path.exists(audio_path):

        print("Audio file not found:")
        print(audio_path)

    else:

        predict_audio(audio_path)