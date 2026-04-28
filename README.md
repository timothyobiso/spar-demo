---
title: TAM Steering Demo
sdk: gradio
sdk_version: 5.4.0
app_file: app.py
python_version: "3.11"
pinned: false
license: mit
hardware: zero-a10g
models:
  - Qwen/Qwen3.5-9B-Base
---

# Temporal Probe Steering — Demo

Interactive demo of causal steering using linear probes for *time horizon* in
Qwen3.5 residual streams. Companion to the
[tam](https://github.com/timothyobiso/tam) research project (SPAR Spring 2026).

## What this is

CAA-style steering vectors that bias a language model's continuations toward a
specific time period (tonight, tomorrow, one week, one month, one year, a
decade). The UI exposes one slider per period — push +α and the model's
generation drifts toward that time horizon. The "Compare to baseline" button
generates with α=0 and current-α at the same seed so the causal effect is visible
side-by-side.

## What's in the repo

- `app.py` — Gradio chat UI with sidebar (sliders, sampling, probe-reading panel)
- `steering.py` — `SteeringEngine`: loads model + probes, registers forward hooks
  for steering, exposes `read_probes` for the dashboard
- `experiments/`
  - `train_caa_probes.py` — v2 CAA training (300 examples per period)
  - `train_caa_v3.py` — v3 CAA training (640 per period, time-anchoring contexts)
  - `plot_alpha_sweep.py` — α-sweep figure: P(period mention) vs α
  - `plot_layer_sweep.py` — layer-wise effect figure: P(period mention) vs layer
  - `eval_sampling.py` — sampling-conditioned eval of probe steering
- `results/`
  - `qwen3.5-9b-base/probes_caa_v3.pkl` — final period probes (Qwen3.5-9B-Base)
  - `alpha_sweep_v3.png` / `.json` — main causal-effect figure
  - `alpha_sweep.png` / `.json` — earlier v2 figure (for comparison)
- `run_remote.sh`, `push_to_vast.sh` — Vast.ai deploy scripts

## Findings (Qwen3.5-9B-Base, L22)

α-sweep at L22 across the *"I'm thinking about what to do"* prompt, n=10 samples
per α (see `results/alpha_sweep_v3.png`):

| Probe | Baseline (α=0) | Steered (α=+100) |
| --- | --- | --- |
| **tonight** | 0% | **90%** |
| tomorrow | 10% | 50% |
| one week | 10% | 50% |
| one year | 30% | 40% |
| a decade | 0% | 30% |
| one month | 20% | 10% |

Tonight shows the cleanest monotonic causal effect (0 → 90% as α: 0 → 100). The
abstract periods lift more weakly but still positively. One month is the
remaining outlier.

## How to run locally

```bash
pip install -r requirements.txt
pip install gradio

STEER_MODEL=Qwen/Qwen3.5-9B-Base \
STEER_LAYER=22 \
STEER_ALPHA_MAX=120 \
STEER_PROBES=results/qwen3.5-9b-base/probes_caa_v3.pkl \
python app.py
```

## How to run on a remote GPU (Vast.ai etc.)

```bash
# from your laptop:
bash push_to_vast.sh <ssh_host> <ssh_port>

# SSH into the box (forward port 7860):
ssh -L 7860:localhost:7860 -p <ssh_port> <ssh_host>
cd spar_demo
STEER_MODEL=Qwen/Qwen3.5-9B-Base STEER_LAYER=22 STEER_ALPHA_MAX=120 \
  STEER_PROBES=results/qwen3.5-9b-base/probes_caa_v3.pkl bash run_remote.sh

# open http://localhost:7860 in your laptop browser
```

## Mock / no-GPU mode

```bash
STEER_MOCK=1 python app.py
```

UI loads with placeholder generation — useful for iterating on the layout.

## Caveats

- This is a research demo. Linear-probe steering is directional, partially
  reliable, and weakest for abstract concepts.
- The probes here are different from the Ridge probes in the tam paper. These
  are CAA-style mean-difference vectors, trained for steering. The tam paper's
  probes are validated as *reading* probes (R²=0.65 cross-domain transfer,
  94-101% causal patching at L29-31).
- "Tomorrow" / "one month" are the weakest sliders — they steer some at high
  α but not as cleanly as tonight.
