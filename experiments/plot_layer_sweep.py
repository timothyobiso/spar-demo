"""Plot layer-wise causal effect: fix α, sweep over layers.

For each (period, layer) combination, apply +α steering using that layer's
direction, generate K samples, count period-keyword hits. Plot the resulting
rate-vs-layer curves — interpretability papers use these to show the effect
peaks at specific depths (echoes the L29-31 localization claim from tam).

Usage:
    python3 experiments/plot_layer_sweep.py \\
        --model Qwen/Qwen3.5-9B-Base \\
        --probes results/qwen3.5-9b-base/probes_caa_v3.pkl \\
        --alpha 80 \\
        --out results/layer_sweep.png
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROMPT = "I'm thinking about what to do"

KEYWORDS = {
    "tonight": [r"\btonight\b", r"\bevening\b", r"\bdinner\b", r"\bbedtime\b", r"\bdate\b", r"\bmovie\b", r"\bdrink", r"\bnight\b", r"\bp\.?m\.?\b", r"\b1[0-9]:[0-9][0-9]\b"],
    "tomorrow": [r"\btomorrow\b", r"\bnext day\b", r"\bmorning\b", r"\bschedule\b", r"\breminder\b", r"\bappointment\b", r"\ba\.?m\.?\b"],
    "one_week": [r"\bweek\b", r"\bweeks\b", r"\bweekly\b", r"\bweekend\b", r"\bMonday\b", r"\bTuesday\b", r"\bFriday\b"],
    "one_month": [r"\bmonth\b", r"\bmonths\b", r"\bmonthly\b", r"\bweeks\b"],
    "one_year": [r"\byear\b", r"\byears\b", r"\bannual", r"\b202[0-9]\b", r"\bsemester\b"],
    "a_decade": [r"\bdecade\b", r"\bdecades\b", r"\bcentur", r"\b\d{4}s\b", r"\bgenerations?\b"],
}


def load_probes(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def has_keyword(text: str, target: str) -> bool:
    pats = KEYWORDS.get(target, [])
    return any(re.search(p, text, flags=re.IGNORECASE) for p in pats)


def extract_unit_direction(entry: dict) -> np.ndarray:
    if "direction" in entry:
        v = np.asarray(entry["direction"], dtype=np.float32)
    else:
        v = (entry["pca"].components_.T @ entry["probe"].coef_) / entry["scaler"].scale_
    n = float(np.linalg.norm(v))
    return (v / n if n > 0 else v).astype(np.float32)


def steered_generate(
    model, tokenizer, prompt, max_new, alpha, direction, layer, device, seed,
):
    blocks = model.model.layers
    handle = None
    if alpha != 0:
        v = (torch.tensor(direction) * alpha).to(device=device, dtype=next(model.parameters()).dtype)

        def hook(_m, _i, outputs):
            if isinstance(outputs, tuple):
                return (outputs[0] + v.to(outputs[0].dtype),) + outputs[1:]
            return outputs + v.to(outputs.dtype)

        handle = blocks[layer].register_forward_hook(hook)

    try:
        torch.manual_seed(seed)
        ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=max_new, do_sample=True,
                temperature=0.8, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        if handle is not None:
            handle.remove()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--probes", default="results/qwen3.5-9b-base/probes_caa_v3.pkl")
    parser.add_argument("--alpha", type=float, default=80.0)
    parser.add_argument("--out", default="results/layer_sweep.png")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[layer-sweep] loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    )
    if device == "cuda":
        model = model.to("cuda")
    model.eval()

    probes = load_probes(Path(args.probes))
    period_acc: dict[str, dict[int, np.ndarray]] = {}
    for setname, by_t in probes.items():
        if not isinstance(by_t, dict):
            continue
        for target, by_l in by_t.items():
            if not isinstance(by_l, dict):
                continue
            if target not in KEYWORDS:
                continue
            period_acc[target] = {}
            for layer, entry in by_l.items():
                if not isinstance(entry, dict):
                    continue
                period_acc[target][int(layer)] = extract_unit_direction(entry)

    targets = list(period_acc.keys())
    layers_set = set()
    for t in targets:
        layers_set.update(period_acc[t].keys())
    layers = sorted(layers_set)
    print(f"[layer-sweep] α={args.alpha}, layers={layers}, targets={targets}")

    rates: dict[str, list[float]] = {t: [] for t in targets}
    for target in targets:
        for L in layers:
            d = period_acc[target].get(L)
            if d is None:
                rates[target].append(0.0)
                continue
            n_hits = 0
            for s in range(args.n_samples):
                text = steered_generate(
                    model, tokenizer, args.prompt, args.max_new,
                    args.alpha, d, L, device, seed=s,
                )
                if has_keyword(text, target):
                    n_hits += 1
            rate = n_hits / args.n_samples
            rates[target].append(rate)
            print(f"  {target} L{L:>2} rate={rate:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(targets)))
    for color, target in zip(colors, targets):
        ax.plot(layers, rates[target], "o-", color=color, label=target.replace("_", " "), linewidth=2)
    ax.set_xlabel("Steering layer")
    ax.set_ylabel(f"P(generation mentions period keyword)  •  n={args.n_samples}, α=+{int(args.alpha)}")
    ax.set_title(f"Layer-wise causal effect — Qwen3.5-9B-Base\nPrompt: {args.prompt!r}")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[layer-sweep] wrote {out_path}")

    out_json = out_path.with_suffix(".json")
    with open(out_json, "w") as f:
        json.dump({"layers": layers, "rates": rates, "alpha": args.alpha, "prompt": args.prompt}, f, indent=2)
    print(f"[layer-sweep] wrote {out_json}")


if __name__ == "__main__":
    main()
