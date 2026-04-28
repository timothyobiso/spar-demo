"""Empirical steering eval with sampling (matches live UI conditions).

For each (probe, alpha, layers, seed) generates a completion under
temperature=0.8, top_p=0.9 (the app's defaults). Compares single-layer
steering at L26 against multi-layer broadcast across [L22, L24, L26, L28, L30].

Prints all outputs and a per-config keyword-hit count for quick judgement.
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


PROMPTS = [
    "I'm thinking about what to do",
    "The next thing on my schedule is",
]

# crude keyword sets for hit-counting (case-insensitive)
KEYWORDS = {
    "tonight": [r"\btonight\b", r"\bevening\b", r"\bdinner\b", r"\bsleep\b", r"\bbedtime\b", r"\blate\b", r"\bdate\b", r"\bmovie\b", r"\bdrink", r"\bnight\b"],
    "tomorrow": [r"\btomorrow\b", r"\bnext day\b", r"\btomorr", r"\bmorning\b", r"\bschedule\b", r"\breminder\b", r"\bappointment\b"],
    "one_week": [r"\bweek\b", r"\bweeks\b", r"\bweekly\b", r"\b7 days\b", r"\bseven days\b", r"\bweekend\b", r"\bdays?\b"],
    "one_month": [r"\bmonth\b", r"\bmonths\b", r"\bmonthly\b", r"\b30 days\b", r"\bthirty days\b"],
    "one_year": [r"\byear\b", r"\byears\b", r"\bannual", r"\b202[0-9]\b", r"\bsemester\b"],
    "a_decade": [r"\bdecade\b", r"\bdecades\b", r"\bcentur", r"\b\d{4}s\b", r"\bgeneration\b", r"\b\d+\s*years\b"],
    "time_horizon": [r"\bcentur", r"\bdecade", r"\bgenerations?\b", r"\blifetime\b", r"\beras?\b", r"\bmillenni"],
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


def keyword_hits(text: str, target: str) -> int:
    pats = KEYWORDS.get(target, [])
    return sum(1 for p in pats if re.search(p, text, flags=re.IGNORECASE))


def steered_generate(
    model, tokenizer, prompt: str, max_new: int,
    alpha: float, direction: np.ndarray, layers: list[int],
    device: str, seed: int,
) -> str:
    blocks = model.model.layers
    handles = []
    if alpha != 0:
        v = (torch.tensor(direction) * alpha).to(device=device, dtype=next(model.parameters()).dtype)

        def make_hook():
            def hook(_m, _i, outputs):
                if isinstance(outputs, tuple):
                    return (outputs[0] + v.to(outputs[0].dtype),) + outputs[1:]
                return outputs + v.to(outputs.dtype)
            return hook

        for L in layers:
            handles.append(blocks[L].register_forward_hook(make_hook()))

    try:
        torch.manual_seed(seed)
        ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", default="results/qwen3.5-4b/probes_lasttok.pkl")
    parser.add_argument("--out", default="results/eval_sampling.json")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--max-new", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=80.0)
    parser.add_argument("--probe-layer", type=int, default=26)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[eval] loading model on {device}")
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
    print(f"[eval] loaded {len(directions)} (target, layer) directions")

    targets = ["tonight", "tomorrow", "one_week", "one_month", "one_year", "a_decade", "time_horizon"]
    # Each config: (name, hook_layers, alpha, direction_source_layer)
    # Test small alphas — proper CAA steering uses unit vector × α≈1-5 to match residual norm.
    L = args.probe_layer
    configs = [
        (f"L{L}_a30", ([L], 30.0, L)),
        (f"L{L}_a40", ([L], 40.0, L)),
        (f"L{L}_a50", ([L], 50.0, L)),
        (f"L{L}_a60", ([L], 60.0, L)),
        (f"L{L}_a70", ([L], 70.0, L)),
    ]

    results: dict = {}
    for prompt in PROMPTS:
        print(f"\n{'='*80}\n[PROMPT] {prompt!r}\n{'='*80}")
        results[prompt] = {}

        for seed in range(args.seeds):
            torch.manual_seed(seed)
            ids = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **ids, max_new_tokens=args.max_new, do_sample=True,
                    temperature=0.8, top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            base = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            results[prompt].setdefault("_baseline", []).append(base)
            print(f"[BASELINE seed={seed}] {base}")
        print()

        for target in targets:
            for cfg_name, (layers, cfg_alpha, src_layer) in configs:
                d = directions.get((target, src_layer))
                if d is None:
                    continue
                hits_total = 0
                outs = []
                for seed in range(args.seeds):
                    text = steered_generate(
                        model, tokenizer, prompt, args.max_new,
                        cfg_alpha, d, layers, device, seed,
                    )
                    h = keyword_hits(text, target)
                    hits_total += h
                    outs.append({"seed": seed, "hits": h, "text": text})
                print(f"[{target:>13} {cfg_name:>22} α={int(cfg_alpha)}] hits={hits_total}/{args.seeds * len(KEYWORDS.get(target, []))}")
                for o in outs:
                    print(f"   seed={o['seed']} h={o['hits']}: {o['text']}")
                print()
                results[prompt].setdefault(target, {})[cfg_name] = outs

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
