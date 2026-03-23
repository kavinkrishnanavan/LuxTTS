import streamlit as st
import soundfile as sf
import torch
import tempfile
import time
from zipvoice.luxvoice import LuxTTS

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

rms = 0.02
t_shift = 0.7
num_steps = 8          # 🔥 increase this
speed = 0.8
ref_duration = 10000    # 🔥 reduce this
# -----------------------------
# Load model (only once)
# -----------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return LuxTTS('YatharthS/LuxTTS', device=device)

lux_tts = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🎤 LuxTTS Voice Cloning App")

def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"

def _write_reference_audio_to_temp(file_obj) -> str:
    suffix = ""
    name = getattr(file_obj, "name", None)
    if isinstance(name, str) and "." in name:
        suffix = "." + name.rsplit(".", 1)[-1]
    if suffix.lower() not in {".wav", ".mp3"}:
        suffix = ".wav"

    if hasattr(file_obj, "getvalue"):
        data = file_obj.getvalue()
    else:
        data = file_obj.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name

st.subheader("Reference audio")
recorded_audio = None
if hasattr(st, "audio_input"):
    recorded_audio = st.audio_input("Record reference audio")

uploaded_file = st.file_uploader("Or upload reference audio (wav/mp3)", type=["wav", "mp3"])

st.subheader("Text")
text = st.text_area(
    "Enter text to generate",
    value="Hello, what is one plus one?",
    height=160,
    placeholder="Type or paste your text here...",
)

# Advanced settings
with st.expander("⚙️ Advanced Settings"):
    rms = st.slider("RMS (volume)", 0.001, 0.05, 0.01)
    t_shift = st.slider("t_shift", 0.1, 1.5, 0.9)
    num_steps = st.slider("num_steps", 1, 10, 4)
    speed = st.slider("speed", 0.5, 2.0, 1.0)
    ref_duration = st.slider("reference duration", 1000, 20000, 10000)

ref_audio = recorded_audio or uploaded_file

if ref_audio and st.button("Generate Speech"):
    with st.spinner("Generating voice..."):
        start_time = time.perf_counter()
        progress = st.progress(0)
        status = st.empty()

        def update_progress(pct: int, label: str, eta_seconds: float | None = None) -> None:
            elapsed = time.perf_counter() - start_time
            parts = [label, f"Elapsed: {_format_duration(elapsed)}"]
            if eta_seconds is not None:
                parts.append(f"ETA: {_format_duration(eta_seconds)}")
            status.text(" | ".join(parts))
            progress.progress(min(100, max(0, int(pct))))

        update_progress(5, "Saving reference audio")
        prompt_audio_path = _write_reference_audio_to_temp(ref_audio)

        # Encode prompt
        update_progress(25, "Encoding reference audio")
        t0 = time.perf_counter()
        encoded_prompt = lux_tts.encode_prompt(
            prompt_audio_path,
            duration=ref_duration,
            rms=rms
        )
        encode_s = time.perf_counter() - t0

        # Generate speech
        update_progress(55, "Generating speech", eta_seconds=max(1.0, encode_s * 1.8))
        t1 = time.perf_counter()
        final_wav = lux_tts.generate_speech(
            text,
            encoded_prompt,
            num_steps=num_steps,
            t_shift=t_shift,
            speed=speed,
            return_smooth=False
        )
        gen_s = time.perf_counter() - t1
        update_progress(90, "Finalizing output", eta_seconds=max(1.0, gen_s * 0.2))

        # Convert to numpy
        final_wav = final_wav.numpy().squeeze()

        # Save output
        output_path = "output.wav"
        sf.write(output_path, final_wav, 48000)
        update_progress(100, "Done", eta_seconds=0)

        st.success("✅ Done!")

        # Play audio
        st.audio(output_path)

        # Download button
        with open(output_path, "rb") as f:
            st.download_button("⬇ Download Audio", f, file_name="output.wav")
