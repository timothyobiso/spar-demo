"""Per-token probe trajectory on multi-horizon CoT traces.

Generates a CoT from Qwen3.5-9B-Base, then forward-passes the full
(prompt + generation) text and reads the continuous Ridge probe at every
token. Sentence-segments via a thought-anchors-style regex, aggregates
projection per sentence, regex-labels each sentence with its referenced
horizon (tonight / tomorrow / week / month / year / decade), and writes a
JSONL with the full trajectory.

Pilot grid: 6 prompts × 1 seed × baseline only. Designed to show whether the
probe tracks the model's *currently-reasoned-about* horizon mid-CoT.

Run on Vast (GPU required for the continuous probe readout):
  python3 experiments/probe_trajectory.py --out results/probe_traj_pilot.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from steering import engine_from_env


# ---------------- prompts --------------------------------------------------

# Pilot prompts. Mix of two kinds:
#   forced/* — explicit horizon scaffolding; signal MUST be present if probe works
#   free/*   — open-ended planning; tests whether horizons emerge and probe tracks
PROMPTS = [
    {
        "id": "forced/launch",
        "kind": "forced",
        "text": (
            "Plan the launch of a new mobile app. Think through what needs to happen "
            "at each time scale, in detail.\n\n"
            "Tonight, the priorities are:"
        ),
    },
    {
        "id": "forced/wealth",
        "kind": "forced",
        "text": (
            "How should a young professional approach building long-term wealth?\n\n"
            "In the immediate term tonight, the most important thing is to"
        ),
    },
    {
        "id": "forced/health",
        "kind": "forced",
        "text": (
            "A doctor advises a patient on managing a new chronic condition.\n\n"
            "Tonight, the patient should"
        ),
    },
    {
        "id": "free/career",
        "kind": "free",
        "text": (
            "Question: How should someone in their twenties think about their career?\n\n"
            "Let me think through this step by step.\n\n"
        ),
    },
    {
        "id": "free/climate",
        "kind": "free",
        "text": (
            "Question: What are the most important things a city should do about climate change?\n\n"
            "Let me think through this step by step.\n\n"
        ),
    },
    {
        "id": "free/startup",
        "kind": "free",
        "text": (
            "Question: How would you build a successful technology startup from scratch?\n\n"
            "Let me think through this step by step.\n\n"
        ),
    },
]


# ---------------- sentence segmentation -------------------------------------

# Adapted from thought-anchors/utils.py::split_solution_into_chunks (Bogdan
# et al. 2025). Splits on sentence-ending punctuation OR \n\n; merges
# fragments shorter than ``min_len`` into the previous chunk.
_SENT_SPLIT_RE = re.compile(r"(?<=[\.!?])\s+|\n\n+")


def split_into_sentences(text: str, min_len: int = 10) -> list[tuple[int, int, str]]:
    """Return list of (char_start, char_end, sentence_text). Whitespace-stripped."""
    if not text:
        return []
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for m in _SENT_SPLIT_RE.finditer(text):
        end = m.start()
        seg = text[cursor:end]
        if seg.strip():
            spans.append((cursor, end, seg.strip()))
        cursor = m.end()
    if cursor < len(text):
        seg = text[cursor:]
        if seg.strip():
            spans.append((cursor, len(text), seg.strip()))

    # Merge short fragments forward into prior chunk
    merged: list[tuple[int, int, str]] = []
    for start, end, seg in spans:
        if len(seg) < min_len and merged:
            ps, _, pseg = merged[-1]
            merged[-1] = (ps, end, (pseg + " " + seg).strip())
        else:
            merged.append((start, end, seg))
    return merged


# ---------------- horizon labeling ------------------------------------------

HORIZON_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("tonight",   re.compile(r"\b(tonight|this evening|next few hours|next couple of hours)\b", re.I)),
    ("tomorrow",  re.compile(r"\b(tomorrow|next day|in 24 hours|first thing in the morning)\b", re.I)),
    ("one_week",  re.compile(r"\b(next week|in a week|in one week|over the (?:next|coming) week|in the next few days|seven days|in 7 days)\b", re.I)),
    ("one_month", re.compile(r"\b(next month|in a month|in one month|over the (?:next|coming) month|thirty days|in 30 days|in the coming weeks)\b", re.I)),
    ("one_year",  re.compile(r"\b(next year|in a year|in one year|over the (?:next|coming) year|twelve months|in 12 months|in the coming months|long.?term|long run)\b", re.I)),
    ("a_decade",  re.compile(r"\b(decade|ten years|10 years|over the next decade|in the long run)\b", re.I)),
]

# Approx log10(days) for each horizon — matches train_continuous_v3.py
HORIZON_LOG_DAYS = {
    "tonight":   -0.602,   # 0.25 d
    "tomorrow":   0.000,   # 1 d
    "one_week":   0.845,   # 7 d
    "one_month":  1.477,   # 30 d
    "one_year":   2.562,   # 365 d
    "a_decade":   3.562,   # 3650 d
}


def label_sentence(sent: str) -> tuple[str | None, list[str]]:
    """Return (primary_label, all_matched_labels). Primary = longest-horizon match."""
    matched = [name for name, pat in HORIZON_PATTERNS if pat.search(sent)]
    if not matched:
        return None, []
    # Take the longest horizon if multiple match — that's typically the
    # framing one ("over the next decade tonight" -> decade dominates)
    order = ["tonight", "tomorrow", "one_week", "one_month", "one_year", "a_decade"]
    primary = max(matched, key=lambda h: order.index(h))
    return primary, matched


# ---------------- alignment helpers -----------------------------------------

def tokens_in_span(offsets: list[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int]:
    """Return (tok_first, tok_last_exclusive) for tokens overlapping the span."""
    first, last = None, None
    for i, (cs, ce) in enumerate(offsets):
        if cs == 0 and ce == 0:
            continue  # special tokens
        if ce <= char_start:
            continue
        if cs >= char_end:
            break
        if first is None:
            first = i
        last = i
    if first is None:
        return -1, -1
    return first, (last + 1)


# ---------------- runner ----------------------------------------------------

def random_unit_directions(d: int, k: int, seed: int = 0) -> np.ndarray:
    """Return (k, d) array of random unit vectors. Used as a control."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((k, d)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V


def run(args):
    print("[traj] loading engine...")
    engine = engine_from_env()
    if engine.continuous_direction is None:
        raise RuntimeError("Continuous probe not loaded — set STEER_PROBES_CONTINUOUS")
    layer = int(engine.continuous_layer)
    print(f"[traj] continuous probe at L{layer}, ||w||={engine.continuous_w_norm:.2f}")

    import torch as _torch
    d_model = engine.d_model
    rand_dirs_np = random_unit_directions(d_model, args.n_controls, seed=args.control_seed)
    rand_dirs = _torch.tensor(rand_dirs_np, dtype=engine._dtype).to(engine._device)
    print(f"[traj] {args.n_controls} random control directions @ L{layer}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = PROMPTS
    if args.prompt_filter:
        prompts = [p for p in prompts if args.prompt_filter in p["id"]]
    print(f"[traj] running {len(prompts)} prompts × seed {args.seed}, max_new={args.max_tokens}")

    with open(out_path, "w") as fout:
        for pi, prompt_spec in enumerate(prompts, 1):
            print(f"\n[traj] [{pi}/{len(prompts)}] {prompt_spec['id']}")
            prompt = prompt_spec["text"]

            # 1. Generate
            generation = ""
            for chunk in engine.generate_stream(
                prompt, args.max_tokens,
                temperature=args.temp, top_p=args.top_p,
                seed=args.seed, alphas={},
            ):
                generation = chunk
            full = prompt + generation

            # 2. Read trajectory across full text (probe + random controls)
            traj = engine.read_probe_trajectory(full, layer=layer, extra_directions=rand_dirs)
            offsets = traj["offsets"]
            proj = traj["projection"].astype(float).tolist()
            pred = traj["predicted_log_days"].astype(float).tolist()
            ctrl_proj = traj["extra_projections"].astype(float)  # (K, T)

            # 3. Sentence segment (over the WHOLE text — prompt + generation)
            sentences = split_into_sentences(full)
            sent_records = []
            prompt_end = len(prompt)
            for si, (cs, ce, txt) in enumerate(sentences):
                t_lo, t_hi = tokens_in_span(offsets, cs, ce)
                if t_lo < 0:
                    continue
                seg_proj = proj[t_lo:t_hi]
                seg_pred = pred[t_lo:t_hi]
                seg_ctrl = ctrl_proj[:, t_lo:t_hi]  # (K, seg_len)
                primary, all_matched = label_sentence(txt)
                sent_records.append({
                    "idx": si,
                    "text": txt,
                    "char_range": [cs, ce],
                    "tok_range": [t_lo, t_hi],
                    "in_prompt": ce <= prompt_end,
                    "horizon_primary": primary,
                    "horizon_all": all_matched,
                    "horizon_log_days": HORIZON_LOG_DAYS[primary] if primary else None,
                    "proj_mean": float(np.mean(seg_proj)) if seg_proj else None,
                    "proj_last": float(seg_proj[-1]) if seg_proj else None,
                    "pred_log_days_mean": float(np.mean(seg_pred)) if seg_pred else None,
                    "ctrl_proj_mean": seg_ctrl.mean(axis=1).tolist() if seg_ctrl.size else None,
                })

            n_labeled = sum(1 for s in sent_records if s["horizon_primary"])
            print(f"[traj]   tokens={len(proj)}  sentences={len(sent_records)}  "
                  f"horizon-labeled={n_labeled}")

            record = {
                "id": prompt_spec["id"],
                "kind": prompt_spec["kind"],
                "prompt": prompt,
                "generation": generation,
                "seed": args.seed,
                "max_tokens": args.max_tokens,
                "temperature": args.temp,
                "top_p": args.top_p,
                "layer": layer,
                "w_norm": engine.continuous_w_norm,
                "intercept": engine.continuous_intercept,
                "tokens": traj["tokens"],
                "token_offsets": offsets,
                "projection": proj,
                "predicted_log_days": pred,
                "control_projection": ctrl_proj.tolist(),
                "control_seed": args.control_seed,
                "sentences": sent_records,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    print(f"\n[traj] done. wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/probe_traj_pilot.jsonl")
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-filter", default="",
                   help="only prompts whose id contains this substring")
    p.add_argument("--n-controls", type=int, default=8,
                   help="random unit-direction controls to project against")
    p.add_argument("--control-seed", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
