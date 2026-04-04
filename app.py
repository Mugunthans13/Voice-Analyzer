import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import speech_recognition as sr
import os
import tempfile
import time
from io import BytesIO
from audio_recorder_streamlit import audio_recorder

# Design configuration
st.set_page_config(
    page_title="Vocalytics | Advanced Voice Analysis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium UI
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1f1437 0%, #0F0A1A 60%);
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Main title */
    .main-title {
        background: linear-gradient(135deg, #A855F7 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0rem;
    }
    .sub-title {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: rgba(30, 21, 51, 0.6);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: rgba(168, 85, 247, 0.5);
    }
    div[data-testid="metric-container"] label {
        color: #A855F7 !important;
        font-weight: 600;
        font-size: 1rem;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-size: 2.5rem;
        font-weight: 700;
    }

    /* Card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    
    /* Voice category badges */
    .badge {
        display: inline-block;
        padding: 0.5em 1em;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9em;
        letter-spacing: 0.025em;
        text-transform: uppercase;
    }
    .badge-deep { background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.5); }
    .badge-male { background: rgba(34, 197, 94, 0.2); color: #4ADE80; border: 1px solid rgba(34, 197, 94, 0.5); }
    .badge-female { background: rgba(234, 179, 8, 0.2); color: #FACC15; border: 1px solid rgba(234, 179, 8, 0.5); }
    .badge-high { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.5); }
    .badge-none { background: rgba(100, 116, 139, 0.2); color: #94A3B8; border: 1px solid rgba(100, 116, 139, 0.5); }
    
    /* Text transcript box */
    .transcript-box {
        background: rgba(0, 0, 0, 0.3);
        border-left: 4px solid #A855F7;
        padding: 20px;
        border-radius: 0 8px 8px 0;
        font-size: 1.1rem;
        color: #E2E8F0;
        line-height: 1.6;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

def header():
    st.markdown('<h1 class="main-title">Vocalytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Advanced AI-Powered Voice & Speech Analysis Engine</p>', unsafe_allow_html=True)

def pitch_category(hz):
    if hz == 0:
        return "No Voice", "badge-none", "🌑"
    elif hz < 130:
        return "Deep Bass", "badge-deep", "🔵"
    elif hz < 180:
        return "Baritone / Low", "badge-male", "🟢"
    elif hz < 300:
        return "Tenor / Mid", "badge-female", "🟡"
    else:
        return "Treble / High", "badge-high", "🔴"

def analyze_audio(file_path):
    st.toast("Loading audio...", icon="🎵")
    # Load audio
    y, sr_rate = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr_rate)
    
    st.toast("Extracting pitch data...", icon="📊")
    # Pitch analysis
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
    )
    f0_filtered = f0[~np.isnan(f0)]
    median_pitch = np.median(f0_filtered) if len(f0_filtered) > 0 else 0
    
    category, badge_class, icon = pitch_category(median_pitch)
    
    st.toast("Transcribing speech...", icon="🗣️")
    # Transcription
    recognizer = sr.Recognizer()
    text = ""
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = "(No identifiable speech detected)"
    except Exception as e:
        text = f"(Transcription error: {str(e)})"
        
    wc = len(text.split())
    wpm = round(wc / (duration / 60), 1) if duration > 0 else 0
    
    return {
        "duration": duration,
        "pitch": median_pitch,
        "category": category,
        "badge_class": badge_class,
        "icon": icon,
        "text": text,
        "wc": wc,
        "wpm": wpm,
        "y": y,
        "sr": sr_rate,
        "f0": f0,
        "time_axis": librosa.times_like(f0, sr=sr_rate)
    }

def create_waveform_plot(y, sr):
    time = np.linspace(0, len(y)/sr, num=len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time[::100], # Downsample for plotting performance
        y=y[::100],
        mode='lines',
        line=dict(color='rgba(168, 85, 247, 0.7)', width=1),
        fill='tozeroy',
        fillcolor='rgba(168, 85, 247, 0.2)'
    ))
    fig.update_layout(
        title="Audio Waveform",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=250,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude"
    )
    return fig

def create_pitch_plot(time_axis, f0):
    fig = go.Figure()
    # Replace nans with None for Plotly
    f0_plot = [f if not np.isnan(f) else None for f in f0]
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=f0_plot,
        mode='markers+lines',
        marker=dict(color='#10B981', size=4),
        line=dict(color='rgba(16, 185, 129, 0.3)', width=1),
    ))
    fig.update_layout(
        title="Pitch Contour (F0)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=250,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)"
    )
    return fig

def main():
    header()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    audio_path = None
    
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        tabs = st.tabs(["🎙️ Record Live", "📁 Upload File"])
        
        with tabs[0]:
            st.write("Click the mic to start recording. Click again to stop.")
            audio_bytes = audio_recorder(
                text="",
                recording_color="#EC4899",
                neutral_color="#64748B",
                icon_name="microphone",
                icon_size="3x"
            )
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_bytes)
                    audio_path = tmp.name
                    
        with tabs[1]:
            uploaded_file = st.file_uploader("Drop your audio here", type=['wav', 'mp3', 'ogg', 'm4a'])
            if uploaded_file is not None:
                st.audio(uploaded_file)
                # Save uploaded file
                suffix = f".{uploaded_file.name.split('.')[-1]}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    audio_path = tmp.name
                    
        st.markdown("</div>", unsafe_allow_html=True)

    if audio_path:
        st.markdown("---")
        
        with st.spinner("Analyzing voice profile..."):
            try:
                start_time = time.time()
                r = analyze_audio(audio_path)
                process_time = time.time() - start_time
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Median Pitch", f"{r['pitch']:.1f} Hz")
                m2.metric("Word Count", str(r['wc']))
                m3.metric("Speaking Rate", f"{r['wpm']} WPM")
                m4.metric("Duration", f"{r['duration']:.2f} s")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Main Analysis Area
                l_col, r_col = st.columns([1, 1])
                
                with l_col:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.markdown("<h3>🎙️ Voice Profile</h3>", unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="margin: 20px 0;">
                        <span style="color: #94A3B8; font-size: 1.1rem;">Acoustic Classification:</span><br><br>
                        <span class="badge {r['badge_class']}">{r['icon']} {r['category']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<h3>📝 Transcript</h3>", unsafe_allow_html=True)
                    st.markdown(f"<div class='transcript-box'>{r['text']}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with r_col:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.markdown("<h3>📈 Acoustic Visualization</h3>", unsafe_allow_html=True)
                    # Waveform
                    st.plotly_chart(create_waveform_plot(r['y'], r['sr']), use_container_width=True)
                    # Pitch Contour
                    st.plotly_chart(create_pitch_plot(r['time_axis'], r['f0']), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                st.caption(f"Analysis completed in {process_time:.2f}s")
                
            except Exception as e:
                st.error(f"Error analyzing audio. Note: For accurate transcriptions, consider uploading clear English speech. Error Details: {e}")
            finally:
                # Cleanup temp file
                if os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except:
                        pass

if __name__ == "__main__":
    main()
