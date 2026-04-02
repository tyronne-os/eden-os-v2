---
title: EVE - Talking Avatar
emoji: 👩
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.23.0
app_file: app.py
pinned: true
license: mit
---

# EVE - Talking Avatar

Audio-driven talking avatar powered by **Wav2Lip** + **Edge TTS**.

## Pipeline
Text → Edge TTS (WAV) → Wav2Lip (HF ZeroGPU) → Animated Video

## Credits
- **Wav2Lip**: Rudrabha et al. (audio-driven lip sync)
- **Hallo4**: Fudan University Generative Vision Lab (SIGGRAPH Asia 2025)
- **Edge TTS**: Microsoft (en-US-AvaMultilingualNeural)
