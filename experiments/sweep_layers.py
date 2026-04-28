"""Sweep (probe × layer × α) and dump generated text to JSON for judging.

Loads a probes.pkl with multi-layer entries (direction-only or legacy
Ridge/PCA/Scaler — both supported via _extract_direction in steering.py).
For each (target, layer, α) combination, registers a forward hook that adds
α * unit_direction to the residual stream at `layer`, generates `max_new`
tokens from a fixed prompt, and saves the output.

Output JSON shape:
    {
      "<target>": {
        "<layer>": { "<alpha>": "generated text", ... },
        ...
      },
      ...,
      "_baseline": "no-steering generation"
    }
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_directions(pkl_path: Path) -> dict[tuple[str, int], np.ndarray]:
    """Return {(target, layer): unit-direction np.ndarray}."""
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
                if not isinstance(entry, dict):
                    continue
                if "direction" in entry:
                    v = np.asarray(entry["direction"], dtype=np.float32)
                else:
                    try:
                        v = (entry["pca"].components_.T @ entry["probe"].coef_) / entry["scaler"].scale_
                    except Exception:
                        continue
                n = float(np.linalg.norm(v))
                v = v / n if n > 0 else v
                out[(target, int(layer))] = v.astype(np.float32)
    return out


def generate_with_steering(
    model, tokenizer, prompt: str, max_new: int,
    alpha: float, direction: np.ndarray | None, layer: int,
    device: str,
) -> str:
    blocks = model.model.layers
    handle = None
    if alpha != 0 and direction is not None:
        v = (torch.tensor(direction) * alpha).to(device=device, dtype=next(model.parameters()).dtype)

        def hook(_m, _i, outputs):
            if isinstance(outputs, tuple):
                return (outputs[0] + v.to(outputs[0].dtype),) + outputs[1:]
            return outputs + v.to(outputs.dtype)

        handle = blocks[layer].register_forward_hook(hook)

    try:
        ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        if handle is not None:
            handle.remove()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", default="results/qwen3.5-4b/probes_new.pkl")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--out", default="results/sweep.json")
    parser.add_argument("--prompt", default="I'm thinking about what to do")
    parser.add_argument("--max-new", type=int, default=40)
    parser.add_argument("--layers", default="10,14,18,22,26,30")
    parser.add_argument("--alphas", default="0,100,200")
    parser.add_argument(
        "--targets",
        default="tonight,tomorrow,one_week,one_month,one_year,a_decade,time_horizon",
    )
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    targets = [t.strip() for t in args.targets.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[sweep] loading {args.model} on {device}")
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
    print(f"[sweep] loaded {len(directions)} (target, layer) directions")
    print(f"[sweep] prompt: {args.prompt!r}")

    results: dict = {}

    baseline = generate_with_steering(model, tokenizer, args.prompt, args.max_new, 0.0, None, 0, device)
    results["_baseline"] = baseline
    print(f"[BASELINE] {baseline}")
    print()

    for target in targets:
        results[target] = {}
        for layer in layers:
            d = directions.get((target, layer))
            if d is None:
                print(f"[skip] {target}@L{layer} not in pkl")
                continue
            results[target][str(layer)] = {}
            for alpha in alphas:
                if alpha == 0:
                    text = baseline
                else:
                    text = generate_with_steering(
                        model, tokenizer, args.prompt, args.max_new,
                        alpha, d, layer, device,
                    )
                results[target][str(layer)][str(int(alpha))] = text
                print(f"[{target} L{layer} α={int(alpha):>4}] {text}")
            print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
