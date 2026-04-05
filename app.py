import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import speech_recognition as sr
import os
import tempfile
import time
import soundfile as sf
from audio_recorder_streamlit import audio_recorder

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Vocalytics | Voice Analyzer",
    page_icon="🎙️",
    layout="wide"
)

# ------------------ UI STYLE ------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 50% 0%, #1f1437 0%, #0F0A1A 60%);
}
.main-title {
    background: linear-gradient(135deg, #A855F7, #EC4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    text-align: center;
    font-weight: 800;
}
.sub-title {
    text-align: center;
    color: #94A3B8;
    margin-bottom: 20px;
}
.glass {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<h1 class="main-title">Vocalytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI Voice Analysis System</p>', unsafe_allow_html=True)

# ------------------ FUNCTIONS ------------------

def pitch_category(hz):
    if hz == 0:
        return "No Voice"
    elif hz < 130:
        return "Deep"
    elif hz < 180:
        return "Male"
    elif hz < 300:
        return "Female"
    else:
        return "High"

def analyze_audio(file_path):
    y, sr_rate = librosa.load(file_path, sr=None)

    # Remove silence
    y = librosa.effects.trim(y)[0]

    duration = librosa.get_duration(y=y, sr=sr_rate)

    # Pitch detection
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )

    f0_clean = f0[~np.isnan(f0)]
    pitch = np.percentile(f0_clean, 50) if len(f0_clean) > 0 else 0

    category = pitch_category(pitch)

    # Speech Recognition
    recognizer = sr.Recognizer()
    text = ""

    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language="en-IN")
    except:
        text = "Speech not recognized"

    wc = len(text.split())
    wpm = round(wc / (duration / 60), 1) if duration > 0 else 0

    # Energy
    energy = np.mean(librosa.feature.rms(y=y))

    return pitch, category, text, wc, wpm, duration, y, sr_rate, f0, energy

def waveform(y, sr):
    t = np.linspace(0, len(y)/sr, len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines'))
    fig.update_layout(template="plotly_dark", height=250)
    return fig

def pitch_plot(f0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=f0, mode='lines+markers'))
    fig.update_layout(template="plotly_dark", height=250)
    return fig

# ------------------ INPUT ------------------
st.markdown("### 🎙️ Record or Upload")

tab1, tab2 = st.tabs(["Record", "Upload"])

audio_path = None

with tab1:
    audio_bytes = audio_recorder()
    if audio_bytes:
        st.audio(audio_bytes)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            audio_path = f.name

with tab2:
    file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg", "m4a"])
    if file:
        st.audio(file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(file.getvalue())
            audio_path = f.name

# ------------------ ANALYSIS ------------------
if audio_path:
    st.markdown("---")
    with st.spinner("Analyzing..."):

        pitch, category, text, wc, wpm, duration, y, sr_rate, f0, energy = analyze_audio(audio_path)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Pitch", f"{pitch:.1f} Hz")
        c2.metric("Words", wc)
        c3.metric("WPM", wpm)
        c4.metric("Duration", f"{duration:.2f}s")

        st.markdown("### 🎯 Voice Type")
        st.success(category)

        st.markdown("### 📝 Transcript")
        st.code(text)

        st.markdown("### ⚡ Energy Level")
        st.info(f"{energy:.5f}")

        st.plotly_chart(waveform(y, sr_rate), use_container_width=True)
        st.plotly_chart(pitch_plot(f0), use_container_width=True)

    # Cleanup
    os.remove(audio_path)
