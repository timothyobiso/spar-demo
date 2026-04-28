"""Modal deployment wrapper.

Deploy:
    modal deploy modal_app.py

Serve ephemerally (auto-stops when you quit):
    modal serve modal_app.py

The probes.pkl at ./results/qwen3.5-4b/probes.pkl is shipped with the image.
HF model weights are cached in a persistent Modal Volume so cold starts only
pay the download once.
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("spar-demo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gradio>=4.44.0",
        "torch>=2.3",
        "transformers>=4.44",
        "accelerate>=0.33",
        "numpy>=1.26",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "fastapi",
    )
    .add_local_dir(
        Path(__file__).parent,
        remote_path="/root/spar_demo",
        ignore=[".venv/*", "__pycache__/*", ".gradio/*", "*.pyc"],
    )
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=3600,
    scaledown_window=300,
    max_containers=1,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def gradio_app():
    import os
    import sys

    sys.path.insert(0, "/root/spar_demo")
    os.chdir("/root/spar_demo")

    import gradio as gr
    from fastapi import FastAPI

    from app import build_ui
    from steering import engine_from_env

    engine = engine_from_env()
    demo = build_ui(engine).queue()

    fastapi_app = FastAPI()
    return gr.mount_gradio_app(fastapi_app, demo, path="/")
