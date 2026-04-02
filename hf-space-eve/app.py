"""EVE — Talking Avatar Demo.

Pipeline: Text → Edge TTS (WAV) → Wav2Lip (HF ZeroGPU) → Animated Video

Uses the proven Wav2Lip pipeline for fast lip-sync animation.
Hallo4 (SIGGRAPH Asia 2025) available via separate L40S GPU job.
"""

import asyncio
import os
import tempfile

import cv2
import gradio as gr
import numpy as np
import soundfile as sf


EDGE_TTS_VOICE = "en-US-AvaMultilingualNeural"


async def generate_tts(text: str) -> str:
    """Text → WAV via Edge TTS."""
    import edge_tts

    mp3_path = os.path.join(tempfile.gettempdir(), "eve_tts.mp3")
    wav_path = os.path.join(tempfile.gettempdir(), "eve_tts.wav")

    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    await communicate.save(mp3_path)

    data, sr = sf.read(mp3_path)
    sf.write(wav_path, data, sr, subtype="PCM_16")
    return wav_path


def animate_with_wav2lip(image_path: str, wav_path: str) -> str | None:
    """Image + WAV → animated video via Wav2Lip HF Space."""
    from gradio_client import Client, handle_file

    client = Client("pragnakalp/Wav2lip-ZeroGPU")
    result = client.predict(
        input_image=handle_file(image_path),
        input_audio=handle_file(wav_path),
        api_name="/run_infrence",
    )

    video_path = result.get("video", result) if isinstance(result, dict) else result
    if video_path and os.path.exists(video_path):
        return video_path
    return None


def eve_speak(text: str, image, progress=gr.Progress()) -> str | None:
    """Main pipeline: Text → TTS → Wav2Lip → Video."""
    if not text.strip():
        return None

    # Save uploaded image
    if image is not None:
        img_path = os.path.join(tempfile.gettempdir(), "eve_ref.png")
        if isinstance(image, np.ndarray):
            img = cv2.resize(image, (512, 512))
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        elif isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.resize(img, (512, 512))
            cv2.imwrite(img_path, img)
    else:
        # Use default Eve
        img_path = os.path.join(os.path.dirname(__file__), "eve-512.png")
        if not os.path.exists(img_path):
            return None

    progress(0.2, desc="Generating voice...")
    wav_path = asyncio.run(generate_tts(text))

    progress(0.4, desc="Animating face with Wav2Lip...")
    try:
        video = animate_with_wav2lip(img_path, wav_path)
        if video:
            progress(1.0, desc="Done!")
            return video
    except Exception as e:
        progress(1.0, desc=f"Error: {str(e)[:80]}")
        print(f"Wav2Lip error: {e}")

    return None


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="EVE - Talking Avatar",
    theme=gr.themes.Base(primary_hue="violet", neutral_hue="slate"),
    css="""
    .eve-title { text-align: center; font-size: 2.5em; font-weight: 200;
                 letter-spacing: 0.3em; color: #a78bfa; margin: 20px 0; }
    .eve-sub { text-align: center; color: #666; margin-bottom: 20px; }
    """,
) as demo:
    gr.HTML("<h1 class='eve-title'>E V E</h1>")
    gr.HTML("<p class='eve-sub'>Audio-driven talking avatar | Edge TTS + Wav2Lip</p>")

    with gr.Row():
        with gr.Column(scale=2):
            output_video = gr.Video(label="Eve", autoplay=True, height=500)
        with gr.Column(scale=1):
            ref_image = gr.Image(
                label="Reference Face (or use default Eve)",
                type="numpy",
                value=os.path.join(os.path.dirname(__file__), "eve-512.png")
                if os.path.exists(os.path.join(os.path.dirname(__file__), "eve-512.png"))
                else None,
            )
            text_input = gr.Textbox(
                label="Talk to Eve",
                placeholder="Type something for Eve to say...",
                lines=3,
                value="Hello! I am Eve, your digital companion. I am so happy to meet you!",
            )
            generate_btn = gr.Button("Make Eve Speak", variant="primary", size="lg")
            gr.HTML(
                "<div style='margin-top:15px;padding:10px;background:#1a1a2e;"
                "border-radius:8px;font-size:0.8em;color:#888;'>"
                "<b>Pipeline:</b> Text → Edge TTS → Wav2Lip (GPU) → Video<br>"
                "<b>Voice:</b> en-US-AvaMultilingualNeural<br>"
                "<b>Credits:</b> Wav2Lip (Rudrabha et al.), "
                "Hallo4 (Fudan Generative Vision, SIGGRAPH Asia 2025)"
                "</div>"
            )

    generate_btn.click(
        fn=eve_speak,
        inputs=[text_input, ref_image],
        outputs=[output_video],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
