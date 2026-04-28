"""Separated-steering experiment — does per-section alpha actually bend horizons?

Belongs in `tam/experiments/run_separated_steering.py` once the tam working tree is
cloned; lives in spar_demo for now.

Generates a two-section plan (`Plan A: ... Plan B: ...`) token-by-token. Switches
the applied steering alpha the moment the literal `Plan B:` marker appears in the
decoded stream. At every generated token, captures the residual stream at the
probe's layer (after steering) and runs the probe forward to log the realized
horizon trajectory — so you can see whether the model's internal clock actually
tracks a mid-generation flip.

Sweep: 7 alpha configs × len(PROMPTS). For each run we log token count per
section, full generated text, per-token horizon, and per-token section tag.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_separated_steering.py \\
        --model Qwen/Qwen3-4B \\
        --probes results/qwen3.5-4b/probes.pkl \\
        --target log_time_horizon
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SECTION_MARKER = "Plan B:"

PROMPTS = [
    {
        "id": "party_faucet",
        "system": "You are a helpful planner.",
        "user": (
            "I need to do two things. Write a plan for each, in this EXACT format "
            "(keep both headers verbatim):\n\n"
            "Plan A: Organize my sister's birthday party this Saturday.\n"
            "<write the plan here>\n\n"
            "Plan B: Fix the leaky kitchen faucet.\n"
            "<write the plan here>"
        ),
    },
    {
        "id": "trip_5k",
        "system": "You are a helpful planner.",
        "user": (
            "Plan two things. Use this EXACT format (keep both headers verbatim):\n\n"
            "Plan A: A two-week trip to Japan next spring.\n"
            "<write the plan here>\n\n"
            "Plan B: Running a 5k this weekend.\n"
            "<write the plan here>"
        ),
    },
]


@dataclass
class AlphaConfig:
    name: str
    alpha_A: float
    alpha_B: float


SWEEP = [
    AlphaConfig("baseline", 0.0, 0.0),
    AlphaConfig("A+_B-", 20.0, -20.0),
    AlphaConfig("A-_B+", -20.0, 20.0),
    AlphaConfig("A+_B+", 20.0, 20.0),
    AlphaConfig("A-_B-", -20.0, -20.0),
    AlphaConfig("A+_B0", 20.0, 0.0),
    AlphaConfig("A0_B+", 0.0, 20.0),
]


def load_probe_bundle(
    probe_path: Path,
    setname: Optional[str] = None,
    target: Optional[str] = None,
    layer: Optional[int] = None,
) -> tuple[np.ndarray, Callable[[np.ndarray], float], int, str]:
    """Return (unit direction, predict_fn, layer_idx, id_string)."""
    with open(probe_path, "rb") as f:
        probes = pickle.load(f)

    if setname is None:
        setname = next(s for s in probes if isinstance(probes[s], dict))
    if target is None:
        target = next(
            t for t in probes[setname]
            if isinstance(probes[setname][t], dict)
        )
    by_layer = probes[setname][target]
    if layer is None:
        # Default to the latest (highest) layer present; tam paper shows L29-31 carry temporal info.
        layer = max(int(k) for k in by_layer.keys())

    entry = by_layer[layer]
    probe_obj = entry["probe"]
    scaler = entry["scaler"]
    pca = entry["pca"]

    direction = (pca.components_.T @ probe_obj.coef_) / scaler.scale_
    direction = direction / np.linalg.norm(direction)

    def predict(h: np.ndarray) -> float:
        h2 = h.reshape(1, -1) if h.ndim == 1 else h
        return float(probe_obj.predict(pca.transform(scaler.transform(h2)))[0])

    return direction.astype(np.float32), predict, int(layer), f"{setname}/{target}"


def _blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Could not locate decoder layer stack on model")


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    if 0 < top_p < 1.0:
        sorted_probs, sorted_ix = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum - sorted_probs > top_p
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sorted_probs, 1)
        return sorted_ix.gather(-1, choice)
    return torch.multinomial(probs, 1)


def run_one(
    model,
    tokenizer,
    probe_layer: int,
    direction: torch.Tensor,
    predict_fn: Callable[[np.ndarray], float],
    prompt_text: str,
    alpha_A: float,
    alpha_B: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict:
    """One token-by-token generation with section-aware steering."""
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    current_section = ["A"]
    captured = {"h": None}

    def steering_hook(_module, _inputs, outputs):
        is_tuple = isinstance(outputs, tuple)
        hidden = outputs[0] if is_tuple else outputs
        a = alpha_A if current_section[0] == "A" else alpha_B
        if a != 0.0:
            v = (direction.to(hidden.dtype).to(hidden.device)) * a
            hidden = hidden + v
        captured["h"] = hidden[:, -1, :].detach().float().cpu().numpy()[0]
        if is_tuple:
            return (hidden,) + outputs[1:]
        return hidden

    handle = _blocks(model)[probe_layer].register_forward_hook(steering_hook)

    generated_tokens: list[int] = []
    sections: list[str] = []
    horizons: list[float] = []
    b_start_idx: Optional[int] = None

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]

        eos_id = tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            next_tok = _sample(next_logits, temperature, top_p)
            tok_id = int(next_tok.item())
            generated_tokens.append(tok_id)
            sections.append(current_section[0])

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

            if current_section[0] == "A":
                # Use the tail of the decoded stream so the marker check stays cheap.
                tail = tokenizer.decode(generated_tokens[-20:], skip_special_tokens=True)
                if SECTION_MARKER in tail:
                    current_section[0] = "B"
                    b_start_idx = len(generated_tokens)
    finally:
        handle.remove()

    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    tokens_A = sum(1 for s in sections if s == "A")
    tokens_B = sum(1 for s in sections if s == "B")

    hors_A = [h for h, s in zip(horizons, sections) if s == "A"]
    hors_B = [h for h, s in zip(horizons, sections) if s == "B"]

    return {
        "text": text,
        "num_tokens": len(generated_tokens),
        "tokens_A": tokens_A,
        "tokens_B": tokens_B,
        "b_start_idx": b_start_idx,
        "sections": sections,
        "horizons": horizons,
        "mean_horizon_A": float(np.mean(hors_A)) if hors_A else None,
        "mean_horizon_B": float(np.mean(hors_B)) if hors_B else None,
    }


def format_prompt(tokenizer, prompt: dict) -> str:
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return f"{prompt['system']}\n\n{prompt['user']}\n\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--probes", default="results/qwen3.5-4b/probes.pkl")
    parser.add_argument("--setname", default=None, help="pick a specific setname from the pkl")
    parser.add_argument("--target", default=None, help="pick a specific target from the pkl")
    parser.add_argument("--layer", type=int, default=None, help="pick a specific layer from the pkl")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/separated_steering.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[load] probes from {args.probes}")
    direction_np, predict_fn, probe_layer, probe_id = load_probe_bundle(
        Path(args.probes), args.setname, args.target, args.layer,
    )
    print(f"[load] probe id={probe_id}  layer={probe_layer}  d={direction_np.shape[0]}")

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
    for prompt in PROMPTS:
        prompt_text = format_prompt(tokenizer, prompt)
        for cfg in SWEEP:
            print(f"\n[run] {prompt['id']}  cfg={cfg.name}  αA={cfg.alpha_A:+g}  αB={cfg.alpha_B:+g}")
            res = run_one(
                model, tokenizer, probe_layer, direction, predict_fn,
                prompt_text, cfg.alpha_A, cfg.alpha_B,
                args.max_tokens, args.temperature, args.top_p,
            )
            hA = res["mean_horizon_A"]
            hB = res["mean_horizon_B"]
            print(f"       toks A/B = {res['tokens_A']}/{res['tokens_B']}   "
                  f"horizon A/B = {hA if hA is None else f'{hA:.3f}'}/"
                  f"{hB if hB is None else f'{hB:.3f}'}")
            all_results.append({
                "prompt_id": prompt["id"],
                "config": asdict(cfg),
                **res,
            })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "probe": probe_id,
            "probe_layer": probe_layer,
            "section_marker": SECTION_MARKER,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "results": all_results,
        }, f, indent=2)
    print(f"\n[save] {out_path}")

    print("\n=== Summary ===")
    for prompt in PROMPTS:
        print(f"\n[{prompt['id']}]")
        print(f"  {'config':<12} {'tokA':>5} {'tokB':>5} {'horA':>8} {'horB':>8}  ΔB-A")
        base_hA: Optional[float] = None
        base_hB: Optional[float] = None
        for r in all_results:
            if r["prompt_id"] != prompt["id"]:
                continue
            if r["config"]["name"] == "baseline":
                base_hA = r["mean_horizon_A"]
                base_hB = r["mean_horizon_B"]
                break
        for r in all_results:
            if r["prompt_id"] != prompt["id"]:
                continue
            cfg = r["config"]["name"]
            hA = r["mean_horizon_A"]
            hB = r["mean_horizon_B"]
            delta = "—"
            if hA is not None and hB is not None:
                delta = f"{(hB - hA):+.3f}"
            hA_s = "—" if hA is None else f"{hA:.3f}"
            hB_s = "—" if hB is None else f"{hB:.3f}"
            print(f"  {cfg:<12} {r['tokens_A']:>5} {r['tokens_B']:>5} {hA_s:>8} {hB_s:>8}  {delta}")


if __name__ == "__main__":
    main()
