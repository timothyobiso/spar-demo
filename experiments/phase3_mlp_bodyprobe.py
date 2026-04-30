"""Phase 3: non-linear (MLP) probe + body-only probe at L22.

Two questions:
  (a) Does a non-linear probe recover horizon information that the linear
      Ridge probe missed?
  (b) Trained directly on body-only sentences (the non-lexical distribution),
      does any probe recover horizon better than the original training-distribution
      probe does?

Pipeline:
  1. Load anchored L22 JSONL.
  2. Forward-pass each record once; capture L22 residuals at every token.
  3. For each generated sentence, slice the residual at its last token.
  4. Split sentences into with-word vs without-word.
  5. Train Linear (Ridge) and MLP probes on each split, held-out R².

Run on Vast:
  STEER_LAYER=22 python3 experiments/phase3_mlp_bodyprobe.py \\
      --in results/probe_traj_anchored_L22.jsonl \\
      --out-dir results/probe_traj_anchored_L22_figs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))
from steering import engine_from_env


HORIZON_ORDER = ["tonight", "tomorrow", "one_week", "one_month", "one_year", "a_decade"]
HORIZON_LOG_DAYS = {
    "tonight":   -0.602, "tomorrow":   0.000, "one_week":   0.845,
    "one_month":  1.477, "one_year":   2.562, "a_decade":   3.562,
}


def load(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def extract_residuals(records, engine, layer):
    """For each labeled generated sentence: forward pass, capture L22 residual
    at the sentence's last token. Returns X (n, d), y (n,), has_word (n,)."""
    import torch as _torch
    blocks = engine._blocks()

    X_list, y_list, has_word_list, anchor_list = [], [], [], []
    captures = {}

    def hook(_m, _i, outputs):
        x = outputs[0] if isinstance(outputs, tuple) else outputs
        captures["res"] = x.squeeze(0).detach()  # (T, d)

    h = blocks[layer].register_forward_hook(hook)
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
            res = captures["res"].float().cpu().numpy()  # (T, d)

            # Map char offsets to tokens via tokenizer offsets (re-tokenize for offsets)
            enc2 = engine.tokenizer(full, return_offsets_mapping=True)
            offsets = enc2["offset_mapping"]

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
                if tok_last < 0 or tok_last >= res.shape[0]:
                    continue
                X_list.append(res[tok_last])
                y_list.append(HORIZON_LOG_DAYS[anchor])
                has_word_list.append(bool(s.get("horizon_regex")))
                anchor_list.append(anchor)
            if (ri + 1) % 20 == 0:
                print(f"[p3] forwarded {ri + 1}/{len(records)} records, {len(X_list)} sentences so far")
    finally:
        h.remove()

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    has_word = np.array(has_word_list, dtype=bool)
    anchors = np.array(anchor_list)
    return X, y, has_word, anchors


def fit_eval(X_train, y_train, X_test, y_test, kind: str, alpha: float = 1.0):
    if kind == "linear":
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    elif kind == "mlp":
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256, 64),
                         activation="relu",
                         max_iter=400,
                         alpha=1e-3,
                         random_state=0,
                         early_stopping=True,
                         n_iter_no_change=20),
        )
    else:
        raise ValueError(kind)
    model.fit(X_train, y_train)
    return float(model.score(X_test, y_test))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="results/probe_traj_anchored_L22.jsonl")
    p.add_argument("--out-dir", default="results/probe_traj_anchored_L22_figs")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--split-seed", type=int, default=42)
    args = p.parse_args()

    print("[p3] loading engine...")
    engine = engine_from_env()
    layer = engine.continuous_layer or 22
    print(f"[p3] using layer L{layer}")

    records = load(Path(args.inp))
    print(f"[p3] {len(records)} anchored records")

    X, y, has_word, anchors = extract_residuals(records, engine, layer)
    print(f"[p3] extracted {len(X)} sentence residuals "
          f"(with-word={has_word.sum()}, without-word={(~has_word).sum()})")

    rng = np.random.default_rng(args.split_seed)
    idx = rng.permutation(len(X))
    n_train = int(0.8 * len(X))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    # ----- All sentences (mix of with/without) -----
    print()
    print("[p3] training probes on ALL sentences:")
    r2_lin_all = fit_eval(X[train_idx], y[train_idx], X[test_idx], y[test_idx], "linear", args.alpha)
    r2_mlp_all = fit_eval(X[train_idx], y[train_idx], X[test_idx], y[test_idx], "mlp")
    print(f"[p3]   Linear: R²={r2_lin_all:+.3f}")
    print(f"[p3]   MLP   : R²={r2_mlp_all:+.3f}")

    # ----- With-word only (lexical-rich distribution) -----
    print("[p3] training probes on WITH-WORD sentences only:")
    Xw, yw = X[has_word], y[has_word]
    if len(Xw) >= 20:
        idxw = rng.permutation(len(Xw))
        ntw = int(0.8 * len(Xw))
        r2_lin_w = fit_eval(Xw[idxw[:ntw]], yw[idxw[:ntw]],
                            Xw[idxw[ntw:]], yw[idxw[ntw:]], "linear", args.alpha)
        r2_mlp_w = fit_eval(Xw[idxw[:ntw]], yw[idxw[:ntw]],
                            Xw[idxw[ntw:]], yw[idxw[ntw:]], "mlp")
        print(f"[p3]   Linear: R²={r2_lin_w:+.3f}")
        print(f"[p3]   MLP   : R²={r2_mlp_w:+.3f}")
    else:
        r2_lin_w = r2_mlp_w = float("nan")

    # ----- Without-word only (non-lexical distribution — the key test) -----
    print("[p3] training probes on WITHOUT-WORD sentences only:")
    Xn, yn = X[~has_word], y[~has_word]
    if len(Xn) >= 20:
        idxn = rng.permutation(len(Xn))
        ntn = int(0.8 * len(Xn))
        r2_lin_n = fit_eval(Xn[idxn[:ntn]], yn[idxn[:ntn]],
                            Xn[idxn[ntn:]], yn[idxn[ntn:]], "linear", args.alpha)
        r2_mlp_n = fit_eval(Xn[idxn[:ntn]], yn[idxn[:ntn]],
                            Xn[idxn[ntn:]], yn[idxn[ntn:]], "mlp")
        print(f"[p3]   Linear: R²={r2_lin_n:+.3f}")
        print(f"[p3]   MLP   : R²={r2_mlp_n:+.3f}")
    else:
        r2_lin_n = r2_mlp_n = float("nan")

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(9, 5))
    conditions = ["all\n(mixed)", "with-word\nonly", "without-word\nonly"]
    x = np.arange(len(conditions))
    width = 0.36
    lin_vals = [r2_lin_all, r2_lin_w, r2_lin_n]
    mlp_vals = [r2_mlp_all, r2_mlp_w, r2_mlp_n]
    ax.bar(x - width / 2, lin_vals, width, color="#1F6FEB",
           label="Linear (Ridge) probe", edgecolor="white")
    ax.bar(x + width / 2, mlp_vals, width, color="#dc2626",
           label="MLP (256→64) probe",   edgecolor="white")
    for xi, v in enumerate(lin_vals):
        if not np.isnan(v):
            ax.text(xi - width / 2, v + 0.01, f"{v:+.2f}", ha="center", va="bottom", fontsize=8)
    for xi, v in enumerate(mlp_vals):
        if not np.isnan(v):
            ax.text(xi + width / 2, v + 0.01, f"{v:+.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel("held-out R² (predicting anchor log_days from L22 residual)")
    ax.set_title(f"Phase 3: linear vs non-linear probe at L{layer}\n"
                 f"on body-sentence residuals, split by horizon-word presence")
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    out = Path(args.out_dir) / "phase3_mlp_bodyprobe.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n[p3] wrote {out}")
    print()
    print("[p3] interpretation:")
    print("[p3]   if MLP_without > Linear_without: non-linearity recovers non-lexical horizon")
    print("[p3]   if Linear_without ≈ Linear_all: lexical content was a small part of all-data signal")


if __name__ == "__main__":
    main()
