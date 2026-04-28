"""Plot α-sweep: fraction of generations mentioning each period vs steering α.

For each (period, α) combination, generate K samples from a fixed prompt, count
how many contain a period-related keyword, plot the resulting curves. Produces
the kind of figure interpretability papers use to argue causal effect of a
direction.

Usage:
    python3 experiments/plot_alpha_sweep.py \\
        --model Qwen/Qwen3.5-9B-Base \\
        --probes results/qwen3.5-9b-base/probes_caa.pkl \\
        --probe-layer 22 \\
        --out results/alpha_sweep.png
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


def load_directions(pkl_path: Path) -> dict[tuple[str, int], np.ndarray]:
    with open(pkl_path, "rb") as f:
        probes = pickle.load(f)
    out: dict[tuple[str, int], np.ndarray] = {}
    for setname, by_t in probes.items():
        if not isinstance(by_t, dict):
            continue
        for target, by_l in by_t.items():
            if not isinstance(by_l, dict):
                continue
            for layer, entry in by_l.items():
                if not isinstance(entry, dict) or "direction" not in entry:
                    continue
                v = np.asarray(entry["direction"], dtype=np.float32)
                n = float(np.linalg.norm(v))
                v = v / n if n > 0 else v
                out[(target, int(layer))] = v.astype(np.float32)
    return out


def has_period_keyword(text: str, target: str) -> bool:
    pats = KEYWORDS.get(target, [])
    return any(re.search(p, text, flags=re.IGNORECASE) for p in pats)


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
    parser.add_argument("--probes", default="results/qwen3.5-9b-base/probes_caa.pkl")
    parser.add_argument("--probe-layer", type=int, default=22)
    parser.add_argument("--out", default="results/alpha_sweep.png")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--alphas", default="-100,-70,-40,-20,0,20,40,60,80,100")
    args = parser.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    targets = list(KEYWORDS.keys())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[plot] loading {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    )
    if device == "cuda":
        model = model.to("cuda")
    model.eval()

    directions = load_directions(Path(args.probes))
    print(f"[plot] loaded {len(directions)} (target, layer) directions")
    print(f"[plot] prompt: {args.prompt!r}, n_samples={args.n_samples}, αs={alphas}")

    rates: dict[str, list[float]] = {t: [] for t in targets}
    raw_outputs: dict = {t: {} for t in targets}

    for target in targets:
        d = directions.get((target, args.probe_layer))
        if d is None:
            print(f"[plot] missing direction for {target}@L{args.probe_layer}")
            for _ in alphas:
                rates[target].append(0.0)
            continue
        for alpha in alphas:
            n_hits = 0
            outs = []
            for s in range(args.n_samples):
                text = steered_generate(
                    model, tokenizer, args.prompt, args.max_new,
                    alpha, d, args.probe_layer, device, seed=s,
                )
                if has_period_keyword(text, target):
                    n_hits += 1
                outs.append(text)
            rate = n_hits / args.n_samples
            rates[target].append(rate)
            raw_outputs[target][str(int(alpha))] = outs
            print(f"  {target} α={int(alpha):>+5} hit-rate={rate:.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(targets)))
    for color, target in zip(colors, targets):
        ax.plot(alphas, rates[target], "o-", color=color, label=target.replace("_", " "), linewidth=2)
    ax.set_xlabel("Steering α (unit-normalized direction × α)")
    ax.set_ylabel(f"P(generation mentions period keyword)  •  n={args.n_samples}")
    ax.set_title(f"Causal steering effect — Qwen3.5-9B-Base, L{args.probe_layer}\nPrompt: {args.prompt!r}")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[plot] wrote {out_path}")

    # Save raw outputs JSON for inspection
    out_json = out_path.with_suffix(".json")
    with open(out_json, "w") as f:
        json.dump({"alphas": alphas, "rates": rates, "raw_outputs": raw_outputs}, f, indent=2)
    print(f"[plot] wrote {out_json}")


if __name__ == "__main__":
    main()
