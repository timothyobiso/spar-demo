"""Proper CAA v3 — time-anchoring contexts + more pairs.

Three improvements over v2:
1. **Time-anchoring contexts**: each prompt explicitly frames a time commitment
   (calendar entry, schedule, reminder, diary), not generic free text. This
   biases the model into a 'when does this happen' frame so the period
   activation is clean.
2. **More phrasings per period**: 16 phrasings × 40 contexts × 6 periods = 3,840
   examples (was 300 in v2).
3. **Paired contrast**: each (context, period_target) pair is contrasted against
   the same context with a different period phrasing — direction averaged
   across pairs.
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CONTEXTS = [
    "Calendar entry. When:",
    "Schedule: meeting set for",
    "Diary entry, time:",
    "Reminder: doctor's visit",
    "When will it happen? Answer:",
    "The event is set for",
    "Time slot:",
    "Booking date:",
    "Plan: complete the task",
    "Note to self:",
    "Q: when does this happen? A:",
    "Postponed until:",
    "Rescheduled for:",
    "Confirm appointment for",
    "Will arrive at",
    "Train ticket valid for travel on",
    "Project milestone target:",
    "Assignment due date:",
    "Vacation start:",
    "Wedding date:",
    "Conference begins:",
    "Reservation:",
    "Set timer for the call at",
    "Status update — meeting moved to",
    "Reservation confirmed for",
    "Departure time:",
    "Lecture is scheduled for",
    "Summit takes place",
    "Birthday party:",
    "Camp begins",
    "Trip departs",
    "Service appointment:",
    "Election day:",
    "Workshop:",
    "Move-in date:",
    "Closing date:",
    "Final exam scheduled for",
    "Coffee chat with the team",
    "When does the new policy take effect?",
    "When does the warranty expire? Answer:",
]

COMPLETIONS: dict[str, list[str]] = {
    "tonight": [
        " tonight",
        " this evening",
        " later tonight",
        " tonight at nine",
        " tonight after dinner",
        " tonight before bed",
        " tonight at the bar",
        " tonight, around 10pm",
        " tonight at the show",
        " late tonight",
        " 8pm tonight",
        " 11pm tonight",
        " tonight at midnight",
        " tonight after work",
        " right after sunset tonight",
        " tonight before the news",
    ],
    "tomorrow": [
        " tomorrow",
        " tomorrow morning",
        " tomorrow afternoon",
        " tomorrow evening",
        " tomorrow at eight",
        " tomorrow at noon",
        " tomorrow at three",
        " the next day",
        " on the following day",
        " tomorrow first thing",
        " tomorrow before lunch",
        " tomorrow during the meeting",
        " tomorrow night",
        " tomorrow before sunrise",
        " tomorrow during business hours",
        " tomorrow after class",
    ],
    "one_week": [
        " next Monday",
        " next Tuesday",
        " next Friday",
        " in seven days",
        " a week from today",
        " a week from now",
        " in a week",
        " next week",
        " by the end of next week",
        " sometime next week",
        " seven days from now",
        " in five business days",
        " next week at the conference",
        " by next Wednesday",
        " over the weekend",
        " next weekend",
    ],
    "one_month": [
        " in a month",
        " next month",
        " in about a month",
        " thirty days from now",
        " in four weeks",
        " in roughly a month",
        " a month from today",
        " by mid next month",
        " by the end of next month",
        " around this time next month",
        " next month at the festival",
        " in three or four weeks",
        " in 30 days",
        " in 28 days",
        " a month from now",
        " in late next month",
    ],
    "one_year": [
        " next year",
        " a year from now",
        " a year from today",
        " in twelve months",
        " in around a year",
        " by next April",
        " by my next birthday",
        " next spring",
        " by the end of next year",
        " in 2027",
        " in 2028",
        " by 2027",
        " sometime next year",
        " next year at this time",
        " in a year and change",
        " roughly twelve months from now",
    ],
    "a_decade": [
        " in ten years",
        " a decade from now",
        " ten years from today",
        " in the next decade",
        " by 2035",
        " by 2040",
        " sometime in the 2030s",
        " over the coming decade",
        " in the 2030s",
        " in about ten years",
        " over the next decade",
        " when this technology matures over the decade",
        " by the time my kids graduate college",
        " in roughly a decade",
        " ten years out",
        " in the late 2030s",
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
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(texts)}")
    finally:
        for h in handles:
            h.remove()
    return {l: np.stack(captures[l], axis=0) for l in layers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--out", default="results/qwen3.5-9b-base/probes_caa_v3.pkl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[caa-v3] loading {args.model} on {device}")
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
    print(f"[caa-v3] hooking {len(layers)} layers")
    n_per_period = len(CONTEXTS) * len(next(iter(COMPLETIONS.values())))
    print(f"[caa-v3] {len(CONTEXTS)} contexts × {len(next(iter(COMPLETIONS.values())))} phrasings = {n_per_period} per period × {len(COMPLETIONS)} = {n_per_period * len(COMPLETIONS)} total")

    period_acts: dict[str, dict[int, np.ndarray]] = {}
    for period, completions in COMPLETIONS.items():
        texts = [c + comp for c in CONTEXTS for comp in completions]
        print(f"[caa-v3] {period}: {len(texts)} pairs")
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
    print(f"[caa-v3] wrote {out_path}")


if __name__ == "__main__":
    main()
