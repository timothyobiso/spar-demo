"""Proper CAA period steering vectors from contrastive prompt-completion pairs.

For each period, build texts of the form `prompt + period_completion`. Capture
the residual-stream activation at the LAST token of the completion at every
(even) layer. Steering direction = mean(period acts) − mean(other-periods acts).

Larger dataset version: 30 prompts × 10 completions × 6 periods = 1800 examples.
Completions are 3-7 tokens to push the activation past pure-token-embedding into
the period concept (period word + thematic continuation).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "The event will happen",
    "We're going to start",
    "I'll see you",
    "She arrives",
    "They're meeting",
    "He plans to leave",
    "I'm getting ready",
    "We're heading out",
    "She'll be back",
    "He'll be home",
    "I expect to finish",
    "It will be done",
    "We'll wrap up",
    "I'll have it ready",
    "The meeting is set for",
    "The deadline is",
    "The trip is scheduled",
    "We're flying out",
    "She'll arrive",
    "He's leaving",
    "I'll do it",
    "We'll finish the project",
    "The package arrives",
    "Plan to do it",
    "Book the appointment for",
    "I'll be there",
    "She'll come over",
    "Pick me up",
    "Call me",
    "Let's catch up",
]

COMPLETIONS: dict[str, list[str]] = {
    "tonight": [
        " tonight after work",
        " tonight at nine",
        " tonight before bed",
        " tonight at the bar",
        " tonight after dinner",
        " late tonight",
        " tonight at the show",
        " tonight when the kids sleep",
        " tonight before the movie",
        " tonight around midnight",
    ],
    "tomorrow": [
        " tomorrow morning at eight",
        " tomorrow afternoon",
        " tomorrow at the office",
        " tomorrow before lunch",
        " tomorrow on the way home",
        " tomorrow at three",
        " tomorrow first thing",
        " tomorrow during the meeting",
        " tomorrow after the gym",
        " tomorrow night",
    ],
    "one_week": [
        " next Monday morning",
        " in seven days",
        " by next Friday",
        " a week from today",
        " next week at the conference",
        " in five business days",
        " by the end of next week",
        " next week during the trip",
        " a week from now",
        " sometime next week",
    ],
    "one_month": [
        " in about a month",
        " thirty days from now",
        " by the end of next month",
        " next month during the festival",
        " in four weeks",
        " a month from today",
        " around this time next month",
        " by mid next month",
        " in roughly a month",
        " next month at the conference",
    ],
    "one_year": [
        " by next April",
        " a year from now",
        " in twelve months",
        " next year at this time",
        " by my next birthday",
        " in around a year",
        " next year before graduation",
        " sometime within the next year",
        " a year from today",
        " by the end of next year",
    ],
    "a_decade": [
        " in ten years",
        " a decade from now",
        " by 2035",
        " in the next decade",
        " ten years from today",
        " sometime in the 2030s",
        " when this technology matures over the decade",
        " by the time my kids graduate college",
        " at some point in the next decade or two",
        " over the coming decade",
    ],
}


def collect_lasttok_activations(
    model, tokenizer, texts: list[str], device: str, layers: list[int]
) -> dict[int, np.ndarray]:
    blocks = model.model.layers
    captures: dict[int, list[np.ndarray]] = {l: [] for l in layers}

    def make_hook(L: int):
        def hook(_m, _i, outputs):
            x = outputs[0] if isinstance(outputs, tuple) else outputs
            v = x[:, -1, :].squeeze(0).detach().to(torch.float32).cpu().numpy()
            captures[L].append(v)
        return hook

    handles = [blocks[l].register_forward_hook(make_hook(l)) for l in layers]
    try:
        for i, t in enumerate(texts):
            ids = tokenizer(t, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**ids)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(texts)}")
    finally:
        for h in handles:
            h.remove()
    return {l: np.stack(captures[l], axis=0) for l in layers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--out", default="results/qwen3.5-4b/probes_caa.pkl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[caa] loading {args.model} on {device} ({dtype})")
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
    print(f"[caa] hooking layers: {layers}")
    print(f"[caa] {len(PROMPTS)} prompts × {len(next(iter(COMPLETIONS.values())))} completions per period × {len(COMPLETIONS)} periods")

    period_acts: dict[str, dict[int, np.ndarray]] = {}
    for period, completions in COMPLETIONS.items():
        texts = [p + c for p in PROMPTS for c in completions]
        print(f"[caa] {period}: {len(texts)} pairs")
        period_acts[period] = collect_lasttok_activations(model, tokenizer, texts, device, layers)

    out_probes: dict = {"period_caa": {}}
    for period in COMPLETIONS:
        out_probes["period_caa"][period] = {}
        others = [n for n in COMPLETIONS if n != period]
        for l in layers:
            mean_p = period_acts[period][l].mean(axis=0)
            mean_o = np.concatenate([period_acts[n][l] for n in others], axis=0).mean(axis=0)
            diff = mean_p - mean_o
            out_probes["period_caa"][period][l] = {
                "direction": diff.astype(np.float32),
            }
        norms = [
            float(np.linalg.norm(out_probes["period_caa"][period][l]["direction"]))
            for l in layers
        ]
        print(f"  {period}: norms min={min(norms):.2f} max={max(norms):.2f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out_probes, f)
    print(f"[caa] wrote {out_path}")


if __name__ == "__main__":
    main()
