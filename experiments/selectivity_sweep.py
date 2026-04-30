"""Phase 1: layer-sweep Hewitt-Liang selectivity + lexical baseline.

For each layer 0..N:
  1. Train Ridge probe on residuals → log_days. Held-out R².
  2. Train CONTROL probes with permuted labels (10 seeds). Held-out R².
  3. Selectivity = probe_R² - mean(control_R²).

Plus a lexical baseline:
  4. TfidfVectorizer + Ridge on the raw text → log_days. Held-out R².
     This is the "best you can do from words alone" upper bound.

The probe is non-trivially semantic only if probe_R² > control_R² + lex_R².

Run on Vast (GPU required for residual extraction):
  python3 experiments/selectivity_sweep.py \\
      --out results/selectivity_sweep.npz \\
      --plot-out results/probe_traj_anchored_L22_figs/selectivity_sweep.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).parent))
from train_caa_v3 import CONTEXTS, COMPLETIONS, collect_lasttok_activations
from train_continuous_v3 import LOG_DAYS


def build_dataset() -> tuple[list[str], np.ndarray]:
    texts, labels = [], []
    for period, completions in COMPLETIONS.items():
        for c in CONTEXTS:
            for comp in completions:
                texts.append(c + comp)
                labels.append(LOG_DAYS[period])
    return texts, np.array(labels, dtype=np.float32)


def fit_score(X_train, y_train, X_test, y_test, alpha: float) -> float:
    return float(Ridge(alpha=alpha).fit(X_train, y_train).score(X_test, y_test))


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[sel] building dataset...")
    texts, y = build_dataset()
    print(f"[sel] {len(texts)} texts × {len(set(y))} unique log_days targets")

    print(f"[sel] loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    )
    if device == "cuda":
        model = model.to("cuda")
    model.eval()

    n_layers = model.config.num_hidden_layers
    if args.layers:
        layers = sorted({int(l) for l in args.layers.split(",")})
    else:
        layers = list(range(0, n_layers, 2))
        if (n_layers - 1) not in layers:
            layers.append(n_layers - 1)
    print(f"[sel] sweeping {len(layers)} layers: {layers}")

    print("[sel] extracting residuals (one forward pass per text, cached)...")
    acts = collect_lasttok_activations(model, tokenizer, texts, device, layers)
    # Free GPU
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Train/test split (80/20, deterministic)
    rng = np.random.default_rng(args.split_seed)
    idx = rng.permutation(len(texts))
    n_train = int(0.8 * len(texts))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    y_train, y_test = y[train_idx], y[test_idx]
    train_texts = [texts[i] for i in train_idx]
    test_texts  = [texts[i] for i in test_idx]
    print(f"[sel] split: {n_train} train / {len(test_idx)} test")

    # ----- Lexical baseline (text → log_days, no model) -----
    lex_pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2000),
        Ridge(alpha=args.alpha),
    )
    lex_pipeline.fit(train_texts, y_train)
    lex_r2 = lex_pipeline.score(test_texts, y_test)
    print(f"[sel] lexical baseline (TfidfVec ngram=(1,2) + Ridge): held-out R²={lex_r2:.3f}")

    # ----- Per-layer probe + control sweep -----
    n_ctrl = args.n_control_seeds
    probe_r2 = []
    ctrl_r2_mean, ctrl_r2_std = [], []
    for layer in layers:
        X = acts[layer]
        X_train, X_test = X[train_idx], X[test_idx]

        r2_real = fit_score(X_train, y_train, X_test, y_test, args.alpha)

        ctrl_scores = []
        for s in range(n_ctrl):
            rng_c = np.random.default_rng(s + 1)
            y_perm = rng_c.permutation(y).astype(np.float32)
            yp_train, yp_test = y_perm[train_idx], y_perm[test_idx]
            r2_perm = fit_score(X_train, yp_train, X_test, yp_test, args.alpha)
            ctrl_scores.append(r2_perm)

        probe_r2.append(r2_real)
        ctrl_r2_mean.append(float(np.mean(ctrl_scores)))
        ctrl_r2_std.append(float(np.std(ctrl_scores)))
        print(f"[sel] L{layer:>2d}: probe R²={r2_real:+.3f}  "
              f"ctrl R²={ctrl_r2_mean[-1]:+.3f}±{ctrl_r2_std[-1]:.3f}  "
              f"selectivity={r2_real - ctrl_r2_mean[-1]:+.3f}  "
              f"(lex baseline={lex_r2:+.3f})")

    probe_r2 = np.array(probe_r2)
    ctrl_r2_mean = np.array(ctrl_r2_mean)
    ctrl_r2_std = np.array(ctrl_r2_std)
    selectivity = probe_r2 - ctrl_r2_mean

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             layers=np.array(layers),
             probe_r2=probe_r2,
             ctrl_r2_mean=ctrl_r2_mean,
             ctrl_r2_std=ctrl_r2_std,
             selectivity=selectivity,
             lex_r2=lex_r2)
    print(f"[sel] saved {out_path}")

    # ----- Plot -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(layers, probe_r2, color="#1F6FEB", linewidth=2.4, marker="o",
             markersize=8, label="probe R² (held-out)")
    ax1.fill_between(layers, ctrl_r2_mean - ctrl_r2_std, ctrl_r2_mean + ctrl_r2_std,
                     color="gray", alpha=0.25, label=f"control ±1σ ({n_ctrl} permuted-label fits)")
    ax1.plot(layers, ctrl_r2_mean, color="gray", linewidth=1.0)
    ax1.axhline(lex_r2, color="#dc2626", linestyle="--", linewidth=2.0,
                label=f"lexical baseline R²={lex_r2:.3f}")
    ax1.axhline(0, color="black", linewidth=0.3, alpha=0.4)
    ax1.set_xlabel("layer")
    ax1.set_ylabel("held-out R² (log_days target)")
    ax1.set_title("Probe vs control vs lexical baseline by layer")
    ax1.legend(loc="best", fontsize=9)

    ax2.plot(layers, selectivity, color="#1F6FEB", linewidth=2.4, marker="o",
             markersize=8, label="selectivity (probe − control)")
    ax2.axhline(lex_r2 - 0, color="#dc2626", linestyle="--", linewidth=2.0,
                label=f"lexical baseline R² (target to beat)")
    ax2.axhline(0, color="black", linewidth=0.3, alpha=0.4)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("selectivity (R² advantage over permuted-label control)")
    ax2.set_title("Layer-wise selectivity — where does horizon information live?")
    ax2.legend(loc="best", fontsize=9)

    plt.tight_layout()
    Path(args.plot_out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.plot_out, dpi=150)
    plt.close(fig)
    print(f"[sel] wrote {args.plot_out}")

    # Where does the probe most cleanly beat the lexical baseline?
    margin = probe_r2 - lex_r2
    best_idx = int(np.argmax(margin))
    print()
    print(f"[sel] best layer (probe vs lex): L{layers[best_idx]} "
          f"(probe {probe_r2[best_idx]:.3f} vs lex {lex_r2:.3f}, "
          f"margin {margin[best_idx]:+.3f})")
    best_sel_idx = int(np.argmax(selectivity))
    print(f"[sel] best layer (selectivity):  L{layers[best_sel_idx]} "
          f"(sel {selectivity[best_sel_idx]:+.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    p.add_argument("--out", default="results/selectivity_sweep.npz")
    p.add_argument("--plot-out", default="results/probe_traj_anchored_L22_figs/selectivity_sweep.png")
    p.add_argument("--layers", default="",
                   help="comma-separated layer indices (default: all even + last)")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Ridge regularization")
    p.add_argument("--n-control-seeds", type=int, default=10,
                   help="number of permuted-label control fits per layer")
    p.add_argument("--split-seed", type=int, default=42)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
