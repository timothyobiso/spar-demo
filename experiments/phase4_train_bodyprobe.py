"""Phase 4 step 1: train body-distribution Ridge probes at multiple layers
and save their directions to a pkl, ready for use as steering vectors.

Reads anchored CoT JSONL (from probe_trajectory.py output), forward-passes
each (prompt + generation), captures residuals at multiple layers at each
generated sentence's final token, trains Ridge per layer, saves to pkl.

The output pkl has the same structure as probes_continuous_v3.pkl:
    {"continuous": {"log_time_horizon": {layer: {"direction", "intercept", "r2_train"}}}}
so it can be loaded via SteeringEngine.load_continuous_probe().

Run on Vast (GPU required):
  STEER_LAYER=22 python3 experiments/phase4_train_bodyprobe.py \\
      --in results/probe_traj_anchored.jsonl \\
      --out results/qwen3.5-9b-base/body_probe_multilayer.pkl \\
      --layers 16,18,20,22,24,26,28
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent.parent))
from steering import engine_from_env


HORIZON_LOG_DAYS = {
    "tonight":   -0.602, "tomorrow":   0.000, "one_week":   0.845,
    "one_month":  1.477, "one_year":   2.562, "a_decade":   3.562,
}


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def collect_residuals(records, engine, layers: list[int]):
    """Forward-pass each (prompt + generation) once with hooks on all layers.
    Capture residuals at each generated sentence's final token.

    Returns:
        X: dict[layer -> (N, d) array]
        y: (N,) array of anchor log_days
        has_word: (N,) bool array
    """
    import torch as _torch
    blocks = engine._blocks()
    captures: dict[int, _torch.Tensor] = {}

    def make_hook(L: int):
        def hook(_m, _i, outputs):
            x = outputs[0] if isinstance(outputs, tuple) else outputs
            captures[L] = x.squeeze(0).detach()
        return hook

    handles = [blocks[L].register_forward_hook(make_hook(L)) for L in layers]

    X_lists: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    y_list: list[float] = []
    has_word_list: list[bool] = []
    anchor_list: list[str] = []

    try:
        for ri, r in enumerate(records):
            anchor = r.get("anchor_horizon")
            if anchor not in HORIZON_LOG_DAYS:
                continue
            full = r["prompt"] + r["generation"]
            input_device = next(engine.model.parameters()).device
            enc = engine.tokenizer(full, return_tensors="pt").to(input_device)
            with _torch.no_grad():
                engine.model(**enc)

            # Re-tokenize to get offsets (encoder doesn't keep them on the GPU enc)
            enc2 = engine.tokenizer(full, return_offsets_mapping=True)
            offsets = enc2["offset_mapping"]
            T = enc["input_ids"].shape[1]

            for s in r.get("sentences", []):
                if s.get("in_prompt"):
                    continue
                cs, ce = s["char_range"]
                tok_last = -1
                for ti, (a, b) in enumerate(offsets):
                    if a == 0 and b == 0:
                        continue
                    if a >= ce:
                        break
                    if b <= ce:
                        tok_last = ti
                if tok_last < 0 or tok_last >= T:
                    continue
                for L in layers:
                    res = captures[L].float().cpu().numpy()
                    X_lists[L].append(res[tok_last])
                y_list.append(HORIZON_LOG_DAYS[anchor])
                has_word_list.append(bool(s.get("horizon_regex")))
                anchor_list.append(anchor)

            if (ri + 1) % 20 == 0:
                print(f"[p4-train] forward {ri + 1}/{len(records)}  "
                      f"sentences so far: {len(y_list)}")
    finally:
        for h in handles:
            h.remove()

    X = {L: np.stack(X_lists[L], axis=0).astype(np.float32) for L in layers}
    y = np.array(y_list, dtype=np.float32)
    has_word = np.array(has_word_list, dtype=bool)
    return X, y, has_word, np.array(anchor_list)


def fit_one(X, y, alpha: float = 1.0, holdout: float = 0.2,
            seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int((1 - holdout) * len(X))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X[train_idx], y[train_idx])
    r2_train = float(ridge.score(X[train_idx], y[train_idx]))
    r2_test  = float(ridge.score(X[test_idx],  y[test_idx]))
    return {
        "direction": ridge.coef_.astype(np.float32),
        "intercept": float(ridge.intercept_),
        "r2_train": r2_train,
        "r2_test":  r2_test,
        "n_train":  int(n_train),
        "n_test":   int(len(X) - n_train),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="results/probe_traj_anchored.jsonl",
                   help="anchored CoT JSONL (with sentences and anchor_horizon)")
    p.add_argument("--out", default="results/qwen3.5-9b-base/body_probe_multilayer.pkl")
    p.add_argument("--layers", default="16,18,20,22,24,26,28,30,31",
                   help="comma-separated layers to train probes at")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    layers = sorted({int(L) for L in args.layers.split(",")})
    print(f"[p4-train] target layers: {layers}")

    print("[p4-train] loading engine...")
    engine = engine_from_env()

    records = load_jsonl(Path(args.inp))
    print(f"[p4-train] {len(records)} anchored records from {args.inp}")

    print(f"[p4-train] forward-passing all records, capturing residuals at "
          f"{len(layers)} layers per sentence-final token...")
    X_by_layer, y, has_word, anchors = collect_residuals(records, engine, layers)
    print(f"[p4-train] {len(y)} sentences, "
          f"with-word={int(has_word.sum())}, without={int((~has_word).sum())}")

    out = {"continuous": {"log_time_horizon": {}}}
    out_diag = {}  # also save per-split diagnostics
    for L in layers:
        X = X_by_layer[L]
        # Train on all sentences (most data, simplest probe)
        all_fit = fit_one(X, y, alpha=args.alpha, seed=args.seed)
        # Diagnostic: also fit on without-word only
        Xn, yn = X[~has_word], y[~has_word]
        without_fit = fit_one(Xn, yn, alpha=args.alpha, seed=args.seed) if len(Xn) >= 20 else None
        # Save the all-sentence direction as the canonical probe (most data)
        out["continuous"]["log_time_horizon"][L] = {
            "direction": all_fit["direction"],
            "intercept": all_fit["intercept"],
            "r2_train":  all_fit["r2_train"],
            "r2_test":   all_fit["r2_test"],
            "trained_on": "all body-sentence residuals",
            "n_train":   all_fit["n_train"],
            "n_test":    all_fit["n_test"],
            "alpha":     args.alpha,
        }
        norm = float(np.linalg.norm(all_fit["direction"]))
        wo_msg = (f", without-word: R²_test={without_fit['r2_test']:+.3f}"
                  if without_fit else "")
        print(f"[p4-train] L{L:>2d}: ||w||={norm:.2f}, "
              f"all-sentence R²_test={all_fit['r2_test']:+.3f}{wo_msg}")
        out_diag[L] = {
            "all": all_fit,
            "without_word": without_fit,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    diag_path = out_path.with_suffix(".diag.pkl")
    with open(diag_path, "wb") as f:
        pickle.dump(out_diag, f)
    print(f"\n[p4-train] wrote probe pkl: {out_path}")
    print(f"[p4-train] wrote diagnostics: {diag_path}")


if __name__ == "__main__":
    main()
