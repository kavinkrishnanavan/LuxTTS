import streamlit as st
import soundfile as sf
import torch
import tempfile
from zipvoice.luxvoice import LuxTTS

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

rms = 0.02
t_shift = 0.7
num_steps = 8          # 🔥 increase this
speed = 1.0
ref_duration = 4000    # 🔥 reduce this
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

uploaded_file = st.file_uploader("Upload reference audio (wav/mp3)", type=["wav", "mp3"])
text = st.text_input("Enter text to generate", "Hello, what is one plus one?")

# Advanced settings
with st.expander("⚙️ Advanced Settings"):
    rms = st.slider("RMS (volume)", 0.001, 0.05, 0.01)
    t_shift = st.slider("t_shift", 0.1, 1.5, 0.9)
    num_steps = st.slider("num_steps", 1, 10, 4)
    speed = st.slider("speed", 0.5, 2.0, 1.0)
    ref_duration = st.slider("reference duration", 1000, 20000, 10000)

if uploaded_file and st.button("Generate Speech"):
    with st.spinner("Generating voice..."):
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            prompt_audio_path = tmp.name

        # Encode prompt
        encoded_prompt = lux_tts.encode_prompt(
            prompt_audio_path,
            duration=ref_duration,
            rms=rms
        )

        # Generate speech
        final_wav = lux_tts.generate_speech(
            text,
            encoded_prompt,
            num_steps=num_steps,
            t_shift=t_shift,
            speed=speed,
            return_smooth=False
        )

        # Convert to numpy
        final_wav = final_wav.numpy().squeeze()

        # Save output
        output_path = "output.wav"
        sf.write(output_path, final_wav, 48000)

        st.success("✅ Done!")

        # Play audio
        st.audio(output_path)

        # Download button
        with open(output_path, "rb") as f:
            st.download_button("⬇ Download Audio", f, file_name="output.wav")