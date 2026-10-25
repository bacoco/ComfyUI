FastH3 Omni ComfyUI workflow tested on NVIDIA DGX Spark

Import workflow.json into ComfyUI. It uses only native nodes.

Required files:
models/diffusion_models/fasth3_omniref_qad_nvfp4_comfy.safetensors
models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
models/vae/minimax_h3_video_vae_fp16.safetensors
models/vae/minimax_h3_audio_vae_fp32.safetensors

Official model sources:
https://huggingface.co/FastVideo/FastH3-Omni-NVFP4-QAD-Comfy
https://huggingface.co/FastVideo/FastH3-Omni-Comfy

Provide an image in the LoadImage node. The local test used the release image:
https://haoailab.com/blogs/fastvideo-omniref-update/media/inputs/hf_character_action_reference.png
saved as input/fasth3_reference.png. Change the prompt to match your image.

Settings: 608 x 352, 73 frames, 24 fps, 8 steps, guidance 1,
video/audio shift 12/3, Euler/simple, seed 0.
The workflow saves an MP4 with synchronized stereo audio.

Measured on one GB10 on 2026-10-25: 28.66 s server execution,
30.04 s client wall time, including model loading but excluding server startup.
The MP4 was fully decoded and its dimensions, frame count, and non-silent
stereo audio verified. This is one measurement, not a general benchmark.

This branch fixes PDD head detection and initialization for this checkpoint.
The repository-root start-test.sh, run-test.py and TEST-FASTH3.txt are local
Spark test tools with machine-specific paths and service names. They require
the existing local environment; they are not portable installation scripts.
Model weights, generated videos and local test evidence are excluded from git.
