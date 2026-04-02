#!/bin/bash
export CUDA_HOME=/usr/local/cuda
. /opt/venv/bin/activate 2>/dev/null || true
cd /tmp/hallo4
python -m vace.vace_wan_inference     --prompt "$PROMPT"     --src_video "$SRC_VIDEO"     --src_ref_images "$SRC_IMAGE"     --src_audio "$SRC_AUDIO"     --save_dir "$OUTPUT_DIR"     --model_path pretrained_models/hallo4/model_weight.ckpt
