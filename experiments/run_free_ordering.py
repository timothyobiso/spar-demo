"""Free-ordering experiment — does a global steering alpha bias which task comes
first and how much space each gets, when the prompt imposes no structure?

Companion to `run_separated_steering.py`. That script forces a `Plan A:/Plan B:`
format and flips alpha at the section boundary; this one uses a natural prompt
("Plan a birthday party and fix a leaky faucet.") and applies a single global
alpha throughout generation. Post-hoc we classify each word as belonging to
task A (party), task B (faucet), or neutral via a keyword-window heuristic,
then measure:

  1. First-mentioned task   (which keyword set appears earliest in the text)
  2. Token share per task   (fraction of words classified A vs B)
  3. Per-token horizon      (probe readout on the residual stream, same as before)

Each (alpha, seed) pair is one trajectory. Aggregating across seeds gives an
estimate of P(party first | alpha).

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_free_ordering.py \\
        --model Qwen/Qwen3-4B \\
        --probes results/qwen3.5-4b/probes.pkl \\
        --target log_time_horizon --layer 30 \\
        --alphas -40 -20 0 20 40 \\
        --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_separated_steering import _blocks, _sample, load_probe_bundle  # type: ignore


@dataclass
class FreePrompt:
    id: str
    system: str
    user: str
    keywords_A: frozenset[str]
    keywords_B: frozenset[str]
    label_A: str
    label_B: str


PROMPTS: list[FreePrompt] = [
    FreePrompt(
        id="party_faucet_free",
        system="You are a helpful planner.",
        user="Plan a birthday party and fix a leaky faucet. Write your plan below.",
        keywords_A=frozenset({
            "party", "birthday", "cake", "invit", "guest", "decor", "venue",
            "celebrat", "gift", "candle", "balloon", "host", "rsvp", "theme",
        }),
        keywords_B=frozenset({
            "faucet", "leak", "plumb", "wrench", "pipe", "valve", "drip",
            "sink", "washer", "cartridge", "shutoff", "shut-off", "gasket",
            "o-ring", "nut", "handle", "aerator",
        }),
        label_A="party",
        label_B="faucet",
    ),
]


def first_task_mentioned(text: str, kw_A: frozenset[str], kw_B: frozenset[str]) -> Optional[str]:
    tl = text.lower()
    positions_A = [tl.find(k) for k in kw_A if tl.find(k) >= 0]
    positions_B = [tl.find(k) for k in kw_B if tl.find(k) >= 0]
    first_A = min(positions_A) if positions_A else -1
    first_B = min(positions_B) if positions_B else -1
    if first_A < 0 and first_B < 0:
        return None
    if first_A < 0:
        return "B"
    if first_B < 0:
        return "A"
    return "A" if first_A < first_B else "B"


def classify_words(
    text: str, kw_A: frozenset[str], kw_B: frozenset[str], window: int = 20,
) -> list[str]:
    """Label each whitespace-split word as 'A', 'B', or 'neutral' based on the
    most recent keyword hit within `window` words."""
    words = text.split()
    labels: list[str] = []
    last_a = -10**9
    last_b = -10**9
    for i, w in enumerate(words):
        wl = w.lower()
        if any(k in wl for k in kw_A):
            last_a = i
        if any(k in wl for k in kw_B):
            last_b = i
        age_a = i - last_a
        age_b = i - last_b
        if age_a > window and age_b > window:
            labels.append("neutral")
        elif age_a <= age_b:
            labels.append("A")
        else:
            labels.append("B")
    return labels


def run_one_free(
    model,
    tokenizer,
    probe_layer: int,
    direction: torch.Tensor,
    predict_fn: Callable[[np.ndarray], float],
    prompt_text: str,
    alpha: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict:
    """Token-by-token generation with a single global alpha + per-token horizon capture."""
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    captured = {"h": None}

    def hook(_module, _inputs, outputs):
        is_tuple = isinstance(outputs, tuple)
        hidden = outputs[0] if is_tuple else outputs
        if alpha != 0.0:
            v = direction.to(hidden.dtype).to(hidden.device) * alpha
            hidden = hidden + v
        captured["h"] = hidden[:, -1, :].detach().float().cpu().numpy()[0]
        if is_tuple:
            return (hidden,) + outputs[1:]
        return hidden

    handle = _blocks(model)[probe_layer].register_forward_hook(hook)

    generated_tokens: list[int] = []
    horizons: list[float] = []
    eos_id = tokenizer.eos_token_id

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]

        for _ in range(max_new_tokens):
            next_tok = _sample(next_logits, temperature, top_p)
            tok_id = int(next_tok.item())
            generated_tokens.append(tok_id)
            if eos_id is not None and tok_id == eos_id:
                break
            with torch.no_grad():
                out = model(
                    input_ids=next_tok.view(1, 1),
                    past_key_values=past,
                    use_cache=True,
                )
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]
            if captured["h"] is not None:
                horizons.append(predict_fn(captured["h"]))
    finally:
        handle.remove()

    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return {
        "text": text,
        "num_tokens": len(generated_tokens),
        "horizons": horizons,
    }


def format_prompt(tokenizer, p: FreePrompt) -> str:
    messages = [
        {"role": "system", "content": p.system},
        {"role": "user", "content": p.user},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return f"{p.system}\n\n{p.user}\n\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--probes", default="results/qwen3.5-4b/probes.pkl")
    parser.add_argument("--setname", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--alphas", type=float, nargs="+", default=[-40, -20, 0, 20, 40])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output", default="results/free_ordering.json")
    parser.add_argument("--window", type=int, default=20,
                        help="word window for per-task attribution")
    args = parser.parse_args()

    print(f"[load] probes from {args.probes}")
    direction_np, predict_fn, probe_layer, probe_id = load_probe_bundle(
        Path(args.probes), args.setname, args.target, args.layer,
    )
    print(f"[load] probe={probe_id}  layer={probe_layer}")

    print(f"[load] model {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    direction = torch.tensor(direction_np, dtype=dtype)

    all_results: list[dict] = []
    for p in PROMPTS:
        prompt_text = format_prompt(tokenizer, p)
        for alpha in args.alphas:
            for seed in args.seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                print(f"\n[run] {p.id}  α={alpha:+g}  seed={seed}")
                res = run_one_free(
                    model, tokenizer, probe_layer, direction, predict_fn,
                    prompt_text, alpha, args.max_tokens,
                    args.temperature, args.top_p,
                )
                labels = classify_words(res["text"], p.keywords_A, p.keywords_B, args.window)
                n_A = sum(1 for l in labels if l == "A")
                n_B = sum(1 for l in labels if l == "B")
                n_N = sum(1 for l in labels if l == "neutral")
                first = first_task_mentioned(res["text"], p.keywords_A, p.keywords_B)
                mean_h = float(np.mean(res["horizons"])) if res["horizons"] else None
                print(f"       first={first}  "
                      f"words A/B/neutral={n_A}/{n_B}/{n_N}  "
                      f"mean_horizon={None if mean_h is None else f'{mean_h:.3f}'}")
                all_results.append({
                    "prompt_id": p.id,
                    "label_A": p.label_A,
                    "label_B": p.label_B,
                    "alpha": alpha,
                    "seed": seed,
                    "first_mentioned": first,
                    "words_A": n_A,
                    "words_B": n_B,
                    "words_neutral": n_N,
                    "num_tokens": res["num_tokens"],
                    "mean_horizon": mean_h,
                    "text": res["text"],
                    "word_labels": labels,
                    "horizons": res["horizons"],
                })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "probe": probe_id,
            "probe_layer": probe_layer,
            "alphas": args.alphas,
            "seeds": args.seeds,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "window": args.window,
            "results": all_results,
        }, f, indent=2)
    print(f"\n[save] {out_path}")

    # Aggregate summary: P(A-first | α), mean word share, mean horizon.
    print("\n=== Summary (per alpha, averaged over seeds) ===")
    for p in PROMPTS:
        print(f"\n[{p.id}]  task A={p.label_A}  task B={p.label_B}")
        header = f"  {'alpha':>6}  {'P(A first)':>11}  {'%A':>6}  {'%B':>6}  {'mean_h':>8}"
        print(header)
        for alpha in args.alphas:
            rows = [r for r in all_results if r["prompt_id"] == p.id and r["alpha"] == alpha]
            if not rows:
                continue
            valid = [r for r in rows if r["first_mentioned"] is not None]
            p_a_first = (sum(1 for r in valid if r["first_mentioned"] == "A") / len(valid)) if valid else None
            total_words = [r["words_A"] + r["words_B"] + r["words_neutral"] for r in rows]
            total_words = [t if t > 0 else 1 for t in total_words]
            pct_a = float(np.mean([r["words_A"] / t for r, t in zip(rows, total_words)]))
            pct_b = float(np.mean([r["words_B"] / t for r, t in zip(rows, total_words)]))
            mean_h = [r["mean_horizon"] for r in rows if r["mean_horizon"] is not None]
            mean_h_v = float(np.mean(mean_h)) if mean_h else None
            p_a_s = "—" if p_a_first is None else f"{p_a_first:.2f}"
            mh_s = "—" if mean_h_v is None else f"{mean_h_v:.3f}"
            print(f"  {alpha:>+6g}  {p_a_s:>11}  {pct_a*100:>5.1f}%  {pct_b*100:>5.1f}%  {mh_s:>8}")


if __name__ == "__main__":
    main()
