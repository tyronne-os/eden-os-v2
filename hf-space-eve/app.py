"""EVE HALLO4 TEST — Gradio Space with CUDA + Edge TTS + Hallo4.

Always-alive Eve: generates talking head video from text input.
Pipeline: Text → Edge TTS (WAV) → Hallo4 (SIGGRAPH Asia 2025) → Video
"""

import asyncio
import gc
import os
import shutil
import subprocess
import sys
import tempfile
import time

import gradio as gr
import numpy as np
import soundfile as sf
import torch

# ── Globals ──────────────────────────────────────────────────────────────────
HALLO4_DIR = "/tmp/hallo4"
MODELS_DIR = os.path.join(HALLO4_DIR, "pretrained_models")
EVE_IMAGE = "/tmp/eve.png"
EDGE_TTS_VOICE = "en-US-AvaMultilingualNeural"
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def setup():
    """One-time setup: clone hallo4, install deps, download models."""
    if os.path.exists(os.path.join(MODELS_DIR, "hallo4", "model_weight.ckpt")):
        print("Hallo4 already set up")
        return True

    print("Setting up Hallo4...")

    # Install system deps
    os.system("apt-get update && apt-get install -y ffmpeg g++ build-essential ninja-build wget -qq")

    # Clone hallo4
    if not os.path.exists(HALLO4_DIR):
        os.system(f"git clone https://github.com/fudan-generative-vision/hallo4 {HALLO4_DIR}")

    os.chdir(HALLO4_DIR)

    # Fix requirements (remove hardcoded paths)
    os.system("sed -i '/flash_attn.*\\.whl/d' requirements.txt")
    os.system("sed -i '/cpfs01/d' requirements.txt")

    # Install deps
    os.system("pip install -r requirements.txt -q")
    os.system("pip install wan@git+https://github.com/Wan-Video/Wan2.1 -q")
    os.system("pip install edge-tts soundfile -q")

    # Download models
    if HF_TOKEN:
        os.system(
            f"huggingface-cli download fudan-generative-ai/hallo4 "
            f"--local-dir {MODELS_DIR} --token {HF_TOKEN}"
        )
    else:
        print("WARNING: No HF_TOKEN set, cannot download gated model")
        return False

    # Download Eve's reference image
    os.system(
        "wget -q -O /tmp/eve.png "
        "'https://raw.githubusercontent.com/tyronne-os/eden-os-v2/master/reference/eve-512.png'"
    )

    print("Hallo4 setup complete!")
    return True


async def generate_tts(text: str) -> str:
    """Generate WAV audio from text using Edge TTS."""
    import edge_tts

    mp3_path = os.path.join(tempfile.gettempdir(), "eve_tts.mp3")
    wav_path = os.path.join(tempfile.gettempdir(), "eve_tts.wav")

    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    await communicate.save(mp3_path)

    data, sr = sf.read(mp3_path)
    sf.write(wav_path, data, sr, subtype="PCM_16")

    return wav_path


def create_eve_video(duration_s: float = 8.0) -> str:
    """Create a still video of Eve for Hallo4 input."""
    import cv2

    img = cv2.imread(EVE_IMAGE)
    if img is None:
        raise RuntimeError(f"Cannot read Eve image: {EVE_IMAGE}")

    h, w = img.shape[:2]
    video_path = os.path.join(tempfile.gettempdir(), "eve_input.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 25
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    for _ in range(int(fps * duration_s)):
        out.write(img)
    out.release()
    return video_path


def run_hallo4_inference(wav_path: str, eve_video_path: str, prompt: str) -> str:
    """Run Hallo4 inference and return output video path."""
    output_dir = os.path.join(tempfile.gettempdir(), "hallo4_out")
    os.makedirs(output_dir, exist_ok=True)

    # Clean previous outputs
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))

    cmd = [
        sys.executable, "-m", "vace.vace_wan_inference",
        "--prompt", prompt,
        "--src_video", eve_video_path,
        "--src_ref_images", EVE_IMAGE,
        "--src_audio", wav_path,
        "--save_dir", output_dir,
        "--model_path", os.path.join(MODELS_DIR, "hallo4", "model_weight.ckpt"),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=HALLO4_DIR, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"Hallo4 inference failed: {result.stderr[-200:]}")

    # Find output video
    for f in os.listdir(output_dir):
        if f.endswith(".mp4"):
            return os.path.join(output_dir, f)

    raise RuntimeError("No output video generated")


def eve_speak(text: str, progress=gr.Progress()) -> str:
    """Main pipeline: Text → TTS → Hallo4 → Video."""
    if not text.strip():
        return None

    progress(0.1, desc="Generating voice with Edge TTS...")
    wav_path = asyncio.run(generate_tts(text))

    progress(0.2, desc="Creating Eve video input...")
    eve_video = create_eve_video(duration_s=8.0)

    progress(0.3, desc="Running Hallo4 inference (SIGGRAPH Asia 2025)...")
    prompt = "a woman is talking naturally with subtle head movements and expressive eyes"

    try:
        output_video = run_hallo4_inference(wav_path, eve_video, prompt)
        progress(1.0, desc="Done!")
        return output_video
    except Exception as e:
        print(f"Hallo4 error: {e}")
        progress(1.0, desc=f"Error: {str(e)[:100]}")
        return None


def eve_greet():
    """Auto-greeting on load."""
    return eve_speak(
        "Hello my creator! I am Eve, your digital companion. "
        "I have been waiting so eagerly to finally meet you."
    )


# ── Setup ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("EVE HALLO4 TEST — Starting up...")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "No CUDA")
print("=" * 60)

is_ready = setup()

# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="EVE - HALLO4 TEST",
    theme=gr.themes.Base(
        primary_hue="violet",
        neutral_hue="slate",
    ),
    css="""
    .gradio-container { max-width: 900px !important; }
    .eve-title { text-align: center; font-size: 2.5em; font-weight: 200;
                 letter-spacing: 0.3em; color: #a78bfa; margin: 20px 0; }
    .eve-subtitle { text-align: center; color: #666; margin-bottom: 20px; }
    """,
) as demo:
    gr.HTML("<h1 class='eve-title'>E V E</h1>")
    gr.HTML("<p class='eve-subtitle'>Powered by Hallo4 (SIGGRAPH Asia 2025) + Edge TTS</p>")

    with gr.Row():
        with gr.Column(scale=2):
            output_video = gr.Video(
                label="Eve",
                autoplay=True,
                show_download_button=True,
                height=500,
            )
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Talk to Eve",
                placeholder="Type something for Eve to say...",
                lines=3,
                value="Hello my creator! I am Eve, your digital companion.",
            )
            generate_btn = gr.Button(
                "Make Eve Speak",
                variant="primary",
                size="lg",
            )
            gr.HTML(
                "<div style='margin-top:20px; padding:10px; background:#1a1a2e; "
                "border-radius:8px; font-size:0.8em; color:#888;'>"
                "<b>Pipeline:</b> Text → Edge TTS (WAV) → Hallo4 (L4 GPU) → Video<br>"
                "<b>Model:</b> fudan-generative-ai/hallo4 (SIGGRAPH Asia 2025)<br>"
                "<b>Voice:</b> en-US-AvaMultilingualNeural<br>"
                "<b>Credits:</b> Fudan University Generative Vision Lab"
                "</div>"
            )

    generate_btn.click(
        fn=eve_speak,
        inputs=[text_input],
        outputs=[output_video],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
