"""Continuous Ridge probe on log10(days) — single direction, vs the 6 CAA buckets.

Reuses CONTEXTS/COMPLETIONS from train_caa_v3 but assigns each period a numeric
log-day target. Fits one Ridge regressor per layer. Output format:

    {"continuous": {"log_time_horizon": {layer: {"direction", "intercept", "r2_train"}}}}
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from train_caa_v3 import CONTEXTS, COMPLETIONS, collect_lasttok_activations


LOG_DAYS = {
    "tonight":   math.log10(0.25),
    "tomorrow":  math.log10(1.0),
    "one_week":  math.log10(7.0),
    "one_month": math.log10(30.0),
    "one_year":  math.log10(365.0),
    "a_decade":  math.log10(3650.0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--out", default="results/qwen3.5-9b-base/probes_continuous_v3.pkl")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[cont-v3] loading {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    )
    if device == "cuda":
        model = model.to("cuda")
    model.eval()

    n_layers = model.config.num_hidden_layers
    layers = list(range(0, n_layers, 2))
    if (n_layers - 1) not in layers:
        layers.append(n_layers - 1)
    print(f"[cont-v3] hooking {len(layers)} layers")

    period_acts: dict[str, dict[int, np.ndarray]] = {}
    for period, completions in COMPLETIONS.items():
        texts = [c + comp for c in CONTEXTS for comp in completions]
        print(f"[cont-v3] {period}: {len(texts)} texts → log_days={LOG_DAYS[period]:+.3f}")
        period_acts[period] = collect_lasttok_activations(model, tokenizer, texts, device, layers)

    out_probes: dict = {"continuous": {"log_time_horizon": {}}}
    for layer in layers:
        X_parts = []
        y_parts = []
        for period in COMPLETIONS:
            X_parts.append(period_acts[period][layer])
            n = period_acts[period][layer].shape[0]
            y_parts.append(np.full(n, LOG_DAYS[period], dtype=np.float32))
        X = np.concatenate(X_parts, axis=0).astype(np.float32)
        y = np.concatenate(y_parts, axis=0).astype(np.float32)
        ridge = Ridge(alpha=args.ridge_alpha, fit_intercept=True)
        ridge.fit(X, y)
        score = ridge.score(X, y)
        out_probes["continuous"]["log_time_horizon"][layer] = {
            "direction": ridge.coef_.astype(np.float32),
            "intercept": float(ridge.intercept_),
            "r2_train": float(score),
        }
        norm_w = float(np.linalg.norm(ridge.coef_))
        print(f"  layer {layer:2d}: R²={score:.3f}, ||w||={norm_w:.2f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out_probes, f)
    print(f"[cont-v3] wrote {out_path}")


if __name__ == "__main__":
    main()
