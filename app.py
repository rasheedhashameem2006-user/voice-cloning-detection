import os
import tempfile

import numpy as np
import pandas as pd
import librosa
import streamlit as st
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VoiceGuard AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    /* Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 16px;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] {
        background-color: #111827;
        border: 1px dashed #475569;
        border-radius: 12px;
        padding: 10px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #ef4444;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 22px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #dc2626;
        color: white;
    }

    /* Cards */
    .custom-card {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #94a3b8;
    }

    .result-ai {
        background-color: #3f1d24;
        border-left: 5px solid #ef4444;
        border-radius: 10px;
        padding: 20px;
    }

    .result-real {
        background-color: #12352d;
        border-left: 5px solid #22c55e;
        border-radius: 10px;
        padding: 20px;
    }

    .small-text {
        color: #94a3b8;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL AND AUDIO CONFIGURATION
# ============================================================

MODEL_PATH = "voice_cloning_detection_model.keras"

SAMPLE_RATE = 16000
DURATION = 2
N_SAMPLES = SAMPLE_RATE * DURATION
N_MFCC = 40
MAX_FRAMES = 126

THRESHOLD = 0.4


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_detection_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(MODEL_PATH)


model = load_detection_model()


# ============================================================
# AUDIO PREPROCESSING
# MUST MATCH TRAINING PREPROCESSING
# ============================================================

def preprocess_audio(audio_file):

    # Load audio
    audio, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Normalize audio
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    # Pad or crop audio to exactly 2 seconds
    if len(audio) < N_SAMPLES:
        audio = np.pad(
            audio,
            (0, N_SAMPLES - len(audio))
        )
    else:
        audio = audio[:N_SAMPLES]

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    # Pad or crop MFCC frames
    if mfcc.shape[1] < MAX_FRAMES:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, MAX_FRAMES - mfcc.shape[1]))
        )
    else:
        mfcc = mfcc[:, :MAX_FRAMES]

    # Standardization
    mfcc = (
        mfcc - np.mean(mfcc)
    ) / (
        np.std(mfcc) + 1e-8
    )

    # CNN input shape:
    # (1, 40, 126, 1)
    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)

    return mfcc


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_voice(audio_file):

    features = preprocess_audio(audio_file)

    prediction = model.predict(
        features,
        verbose=0
    )

    score = float(np.squeeze(prediction))

    if score >= THRESHOLD:
        result = "AI-Generated Voice"
        confidence = score * 100
    else:
        result = "Real Voice"
        confidence = (1 - score) * 100

    return result, score, confidence


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:20px 0;">
            <div style="font-size:60px;">🎙️</div>
            <h2>VoiceGuard AI</h2>
            <p style="color:#94a3b8;">
                Intelligent Voice Security
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Select Page",
        [
            "Dashboard",
            "Detection History",
            "Model Information"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")

    if model is not None:
        st.success("Model Loaded")
    else:
        st.error("Model Not Found")

    st.caption("VoiceGuard AI v1.0")


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "Dashboard":

    # --------------------------------------------------------
    # HERO SECTION
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="custom-card">
            <div class="hero-title">
                Voice Cloning Detection
            </div>
            <div class="hero-subtitle">
                AI-powered audio analysis for detecting real and
                synthetic voices.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total_detections = len(st.session_state.history)

    ai_count = sum(
        1 for item in st.session_state.history
        if item["Result"] == "AI-Generated Voice"
    )

    real_count = sum(
        1 for item in st.session_state.history
        if item["Result"] == "Real Voice"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Detections",
            total_detections
        )

    with col2:
        st.metric(
            "AI-Generated Voices",
            ai_count
        )

    with col3:
        st.metric(
            "Real Voices",
            real_count
        )

    with col4:
        st.metric(
            "Detection Threshold",
            THRESHOLD
        )

    st.markdown("")

    # --------------------------------------------------------
    # UPLOAD SECTION
    # --------------------------------------------------------

    st.markdown("## Analyze Audio")

    st.markdown(
        """
        <div class="custom-card">
            <p class="small-text">
                Upload a WAV, MP3, or other supported audio file.
                The system will analyze the voice and classify it.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=[
            "wav",
            "mp3",
            "ogg",
            "flac",
            "m4a"
        ]
    )

    if uploaded_file is not None:

        st.audio(
            uploaded_file,
            format="audio/wav"
        )

        st.markdown("")

        analyze_button = st.button(
            "Analyze Voice",
            use_container_width=True
        )

        if analyze_button:

            if model is None:

                st.error(
                    f"Model file not found: {MODEL_PATH}"
                )

            else:

                with st.spinner("Analyzing audio..."):

                    try:

                        # Save uploaded audio temporarily
                        suffix = os.path.splitext(
                            uploaded_file.name
                        )[1]

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=suffix
                        ) as temp_audio:

                            temp_audio.write(
                                uploaded_file.getbuffer()
                            )

                            temp_audio_path = temp_audio.name

                        # Predict
                        result, score, confidence = predict_voice(
                            temp_audio_path
                        )

                        # Delete temporary file
                        os.remove(temp_audio_path)

                        # Store result
                        st.session_state.history.append({
                            "File": uploaded_file.name,
                            "Result": result,
                            "Score": round(score, 4),
                            "Confidence": round(confidence, 2)
                        })

                        st.session_state.total_detections += 1

                        # ------------------------------------------------
                        # RESULT DISPLAY
                        # ------------------------------------------------

                        st.markdown("## Detection Result")

                        if result == "AI-Generated Voice":

                            st.markdown(
                                f"""
                                <div class="result-ai">
                                    <h2>⚠️ AI-Generated Voice Detected</h2>
                                    <p>
                                        The model classified this audio
                                        as potentially synthetic or
                                        AI-generated.
                                    </p>
                                    <h3>Confidence: {confidence:.2f}%</h3>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        else:

                            st.markdown(
                                f"""
                                <div class="result-real">
                                    <h2>✓ Real Voice Detected</h2>
                                    <p>
                                        The model classified this audio
                                        as a natural human voice.
                                    </p>
                                    <h3>Confidence: {confidence:.2f}%</h3>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        # ------------------------------------------------
                        # PROBABILITY DETAILS
                        # ------------------------------------------------

                        st.markdown("### Detection Score")

                        score_col1, score_col2 = st.columns(2)

                        with score_col1:
                            st.metric(
                                "AI Voice Score",
                                f"{score:.4f}"
                            )

                        with score_col2:
                            st.metric(
                                "Confidence",
                                f"{confidence:.2f}%"
                            )

                        st.progress(
                            min(max(confidence / 100, 0.0), 1.0)
                        )

                    except Exception as e:

                        st.error(
                            f"Prediction failed: {str(e)}"
                        )


    # ========================================================
    # ANALYTICS SECTION
    # ========================================================

    st.markdown("---")

    st.markdown("## Detection Analytics")

    if len(st.session_state.history) > 0:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        # ----------------------------------------------------
        # GRAPH 1: DETECTION DISTRIBUTION
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Detection Distribution")

            result_counts = (
                history_df["Result"]
                .value_counts()
                .reindex(
                    [
                        "Real Voice",
                        "AI-Generated Voice"
                    ],
                    fill_value=0
                )
                .reset_index()
            )

            result_counts.columns = [
                "Result",
                "Count"
            ]

            st.bar_chart(
                result_counts,
                x="Result",
                y="Count",
                use_container_width=True
            )

        # ----------------------------------------------------
        # GRAPH 2: CONFIDENCE SCORE HISTORY
        # ----------------------------------------------------

        with col2:

            st.markdown("### Confidence Score History")

            confidence_df = history_df[
                ["Confidence"]
            ].copy()

            confidence_df["Detection"] = range(
                1,
                len(confidence_df) + 1
            )

            confidence_df = confidence_df.set_index(
                "Detection"
            )

            st.line_chart(
                confidence_df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # GRAPH 3: AI VS REAL PERCENTAGE
        # ----------------------------------------------------

        st.markdown("### Classification Overview")

        pie_data = (
            history_df["Result"]
            .value_counts()
            .reindex(
                [
                    "Real Voice",
                    "AI-Generated Voice"
                ],
                fill_value=0
            )
        )

        st.bar_chart(
            pie_data,
            use_container_width=True
        )

        # ----------------------------------------------------
        # HISTORY TABLE
        # ----------------------------------------------------

        st.markdown("### Recent Detection History")

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Upload and analyze audio files to generate analytics."
        )


# ============================================================
# DETECTION HISTORY PAGE
# ============================================================

elif page == "Detection History":

    st.markdown("## Detection History")

    if len(st.session_state.history) == 0:

        st.info(
            "No detection history available yet."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Files",
                len(history_df)
            )

        with col2:
            st.metric(
                "AI-Generated",
                (history_df["Result"] == "AI-Generated Voice").sum()
            )

        with col3:
            st.metric(
                "Real Voice",
                (history_df["Result"] == "Real Voice").sum()
            )

        st.markdown("### Confidence Scores")

        confidence_df = history_df[
            ["Confidence"]
        ].copy()

        confidence_df["Detection"] = range(
            1,
            len(confidence_df) + 1
        )

        confidence_df = confidence_df.set_index(
            "Detection"
        )

        st.line_chart(
            confidence_df,
            use_container_width=True
        )

        if st.button("Clear Detection History"):

            st.session_state.history = []

            st.rerun()


# ============================================================
# MODEL INFORMATION PAGE
# ============================================================

elif page == "Model Information":

    st.markdown("## Model Information")

    st.markdown(
        """
        <div class="custom-card">
            <h3>Voice Cloning Detection Model</h3>
            <p class="small-text">
                This system uses a convolutional neural network
                trained on MFCC audio features to classify voices
                as real or AI-generated.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Audio Configuration")

        st.write("Sample Rate: 16,000 Hz")
        st.write("Audio Duration: 2 seconds")
        st.write("MFCC Features: 40")
        st.write("Maximum MFCC Frames: 126")

    with col2:

        st.markdown("### Classification Configuration")

        st.write("Real Voice Label: 0")
        st.write("AI-Generated Voice Label: 1")
        st.write(f"Classification Threshold: {THRESHOLD}")
        st.write("Input Shape: (40, 126, 1)")

    st.markdown("### Processing Pipeline")

    st.write("""
    1. Audio file is uploaded.
    2. Audio is converted to mono format.
    3. Sampling rate is standardized to 16 kHz.
    4. Audio is padded or cropped to 2 seconds.
    5. MFCC features are extracted.
    6. Features are padded or cropped to 126 frames.
    7. Features are standardized.
    8. The CNN model predicts the voice class.
    9. The result and confidence score are displayed.
    """)

    if model is not None:

        st.success("Detection model is loaded and ready.")

        with st.expander("View Model Summary"):

            summary_lines = []

            model.summary(
                print_fn=lambda x: summary_lines.append(x)
            )

            st.code(
                "\n".join(summary_lines)
            )

    else:

        st.error(
            f"Model file not found: {MODEL_PATH}"
        )