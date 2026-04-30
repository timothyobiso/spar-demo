"""Phase 4 step 2: causal validation of the body-distribution probe direction.

Uses the body-distribution probe trained in phase4_train_bodyprobe.py as a
steering vector. For each "tonight"-anchored prompt, runs:
  - baseline (no steering)
  - body-direction steering at α ∈ {-100, -60, -30, +30, +60, +100} per layer
  - matched-norm random-direction control at the same α and layer

Sweeps multiple layers (default 18, 20, 22, 24, 26) so we can identify
*where* (if anywhere) steering with the body-distribution direction
produces a probe-specific causal effect on horizon-relevant generation.

Run on Vast (after phase4_train_bodyprobe.py has produced the pkl):
  STEER_PROBES_CONTINUOUS=results/qwen3.5-9b-base/body_probe_multilayer.pkl \\
      python3 experiments/phase4_causal.py \\
      --out results/phase4_causal.jsonl \\
      --seeds 42,1337,7 \\
      --layers 18,20,22,24,26 \\
      --alphas -100,-60,-30,30,60,100 \\
      --n-controls 5

Then plot:
  python3 experiments/phase4_causal.py --plot \\
      --in results/phase4_causal.jsonl \\
      --out-dir results/probe_traj_anchored_L22_figs
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from steering import engine_from_env
from experiments.phase2_causal import (
    DOMAINS, SOURCE_HEADER, SOURCE_LABEL, PERIOD_ORDER,
    HORIZON_PATTERNS, count_horizon_words, PERIOD_COLORS,
)


def load_body_directions(pkl_path: Path) -> dict[int, dict]:
    """Return {layer: {direction, intercept, r2_test, ...}}"""
    with open(pkl_path, "rb") as f:
        blob = pickle.load(f)
    return {int(L): entry for L, entry in blob["continuous"]["log_time_horizon"].items()}


def run_experiment(args):
    print("[p4] loading engine...")
    engine = engine_from_env()

    body_pkl = Path(args.body_probe_pkl)
    body_by_layer = load_body_directions(body_pkl)
    print(f"[p4] loaded body-distribution probes for layers: {sorted(body_by_layer.keys())}")

    layers = [int(L) for L in args.layers.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    domains = DOMAINS
    if args.domain_filter:
        domains = [d for d in domains if args.domain_filter in d[0]]

    # Validate layers exist in body probe
    for L in layers:
        if L not in body_by_layer:
            raise ValueError(f"layer {L} not in body probe pkl. Available: {sorted(body_by_layer.keys())}")

    # Build random control directions per layer (deterministic per --control-seed)
    import torch as _torch
    rng = np.random.default_rng(args.control_seed)
    rand_dirs = {}
    for L in layers:
        d = engine.d_model
        V = rng.standard_normal((args.n_controls, d)).astype(np.float32)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        rand_dirs[L] = _torch.tensor(V, dtype=engine._dtype).to(engine._device)

    # Body-direction unit vectors per layer
    body_unit = {}
    for L in layers:
        w = np.asarray(body_by_layer[L]["direction"], dtype=np.float32)
        n = float(np.linalg.norm(w))
        if n == 0:
            raise ValueError(f"body probe at L{L} has zero-norm direction")
        u = w / n
        body_unit[L] = _torch.tensor(u, dtype=engine._dtype).to(engine._device)

    n_baseline = len(domains) * len(seeds)
    n_body = n_baseline * len(layers) * len(alphas)
    n_rand = n_baseline * len(layers) * len(alphas) * args.n_controls
    total = n_baseline + n_body + n_rand
    print(f"[p4] grid: {n_baseline} baselines + {n_body} body-steered + "
          f"{n_rand} random-steered = {total} generations")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    i = 0
    with open(out_path, "w") as fout:
        for dom_id, dom_text in domains:
            for seed in seeds:
                base_prompt = f"{dom_text}\n\n{SOURCE_HEADER}"

                # Baseline
                gen = ""
                for chunk in engine.generate_stream(
                    base_prompt, args.max_tokens,
                    temperature=args.temp, top_p=args.top_p,
                    seed=seed, alphas={},
                ):
                    gen = chunk
                i += 1
                rec = {
                    "i": i, "domain": dom_id, "seed": seed,
                    "kind": "baseline",
                    "layer": None, "alpha": 0.0, "direction_idx": None,
                    "prompt": base_prompt, "generation": gen,
                    "counts": count_horizon_words(gen),
                }
                fout.write(json.dumps(rec) + "\n"); fout.flush()

                for L in layers:
                    # Body-direction steering
                    for alpha in alphas:
                        steering_vectors = {L: body_unit[L] * float(alpha)}
                        gen = ""
                        for chunk in engine.generate_stream(
                            base_prompt, args.max_tokens,
                            temperature=args.temp, top_p=args.top_p,
                            seed=seed,
                            steering_vectors=steering_vectors,
                        ):
                            gen = chunk
                        i += 1
                        rec = {
                            "i": i, "domain": dom_id, "seed": seed,
                            "kind": "body",
                            "layer": L, "alpha": alpha, "direction_idx": None,
                            "prompt": base_prompt, "generation": gen,
                            "counts": count_horizon_words(gen),
                        }
                        fout.write(json.dumps(rec) + "\n"); fout.flush()

                    # Random-direction control
                    for k in range(args.n_controls):
                        for alpha in alphas:
                            steering_vectors = {L: rand_dirs[L][k] * float(alpha)}
                            gen = ""
                            for chunk in engine.generate_stream(
                                base_prompt, args.max_tokens,
                                temperature=args.temp, top_p=args.top_p,
                                seed=seed,
                                steering_vectors=steering_vectors,
                            ):
                                gen = chunk
                            i += 1
                            rec = {
                                "i": i, "domain": dom_id, "seed": seed,
                                "kind": "random",
                                "layer": L, "alpha": alpha, "direction_idx": k,
                                "prompt": base_prompt, "generation": gen,
                                "counts": count_horizon_words(gen),
                            }
                            fout.write(json.dumps(rec) + "\n"); fout.flush()

                if i % 60 == 0 or i == total:
                    print(f"[p4] {i}/{total}  ({100 * i / total:.1f}%)")

    print(f"\n[p4] done. wrote {out_path}")


# ---------------- analysis & plotting ------------------------------------

def run_plot(args):
    records = []
    with open(args.inp) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[p4] loaded {len(records)} records")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index baselines per (domain, seed)
    baseline = {}
    for r in records:
        if r["kind"] == "baseline":
            baseline[(r["domain"], r["seed"])] = r["counts"]

    layers = sorted({r["layer"] for r in records if r["layer"] is not None})
    alphas = sorted({r["alpha"] for r in records if r["alpha"] != 0.0})

    # ----- Plot 1: dose-response per layer (body vs random) for far-vs-near score -----
    # Score: (decade-words + year-words) - (tonight-words + tomorrow-words)
    def far_minus_near(counts):
        return ((counts["a_decade"] + counts["one_year"])
                - (counts["tonight"] + counts["tomorrow"]))

    fig, axes = plt.subplots(1, len(layers), figsize=(3.5 * len(layers), 5),
                             sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, L in zip(axes, layers):
        body_at_alpha: dict[float, list[float]] = {a: [] for a in alphas}
        rand_at_alpha: dict[float, list[float]] = {a: [] for a in alphas}
        for r in records:
            if r["layer"] != L or r["alpha"] == 0.0:
                continue
            b = baseline.get((r["domain"], r["seed"]))
            if not b:
                continue
            delta = far_minus_near(r["counts"]) - far_minus_near(b)
            if r["kind"] == "body":
                body_at_alpha[r["alpha"]].append(delta)
            elif r["kind"] == "random":
                rand_at_alpha[r["alpha"]].append(delta)

        body_means = [np.mean(body_at_alpha[a]) if body_at_alpha[a] else 0.0 for a in alphas]
        body_sems  = [np.std(body_at_alpha[a]) / max(1, np.sqrt(len(body_at_alpha[a])))
                      if body_at_alpha[a] else 0.0 for a in alphas]
        rand_means = [np.mean(rand_at_alpha[a]) if rand_at_alpha[a] else 0.0 for a in alphas]
        rand_sems  = [np.std(rand_at_alpha[a]) / max(1, np.sqrt(len(rand_at_alpha[a])))
                      if rand_at_alpha[a] else 0.0 for a in alphas]

        ax.errorbar(alphas, body_means, yerr=body_sems, marker="o", color="#1F6FEB",
                    linewidth=2.0, label="body-probe direction", capsize=3)
        ax.errorbar(alphas, rand_means, yerr=rand_sems, marker="s", color="gray",
                    linewidth=2.0, label=f"random control (K={len({r['direction_idx'] for r in records if r['kind']=='random'})})",
                    capsize=3)
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
        ax.axvline(0, color="black", linewidth=0.4, alpha=0.4)
        ax.set_xlabel(r"steering strength $\alpha$")
        ax.set_title(f"L{L}", fontsize=10)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel(r"$\Delta$ (far-near horizon-lexicon counts)")
    fig.suptitle("Phase 4 — body-distribution probe vs.\ random-direction null, dose-response per layer")
    plt.tight_layout()
    plt.savefig(out_dir / "phase4_doseresponse.png", dpi=150)
    plt.close(fig)
    print(f"[p4] wrote {out_dir/'phase4_doseresponse.png'}")

    # ----- Plot 2: specificity heatmap (layers × horizon classes) -----
    M_body = np.zeros((len(layers), len(PERIOD_ORDER)))
    M_rand = np.zeros((len(layers), len(PERIOD_ORDER)))
    for li, L in enumerate(layers):
        for hi, h in enumerate(PERIOD_ORDER):
            body_deltas, rand_deltas = [], []
            for r in records:
                if r["layer"] != L or r["alpha"] == 0.0:
                    continue
                b = baseline.get((r["domain"], r["seed"]))
                if not b:
                    continue
                d = r["counts"][h] - b[h]
                if r["kind"] == "body":
                    body_deltas.append(d)
                elif r["kind"] == "random":
                    rand_deltas.append(d)
            M_body[li, hi] = float(np.mean(body_deltas)) if body_deltas else 0.0
            M_rand[li, hi] = float(np.mean(rand_deltas)) if rand_deltas else 0.0

    M_spec = M_body - M_rand
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(layers) + 1.5))
    import matplotlib.colors as mcolors
    vmax = max(abs(M_spec.min()), abs(M_spec.max()), 0.05)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(M_spec, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(PERIOD_ORDER)))
    ax.set_xticklabels(PERIOD_ORDER, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{L}" for L in layers])
    ax.set_xlabel("horizon-lexicon (counts in continuation)")
    ax.set_ylabel("steering layer")
    ax.set_title("Phase 4 specificity: body-probe Δ minus random-direction Δ\n"
                 "(positive = body direction shifts lexicon away from baseline more than random)")
    for i in range(M_spec.shape[0]):
        for j in range(M_spec.shape[1]):
            v = M_spec[i, j]
            if abs(v) > 0.03:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > vmax * 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="specificity")
    plt.tight_layout()
    plt.savefig(out_dir / "phase4_specificity_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"[p4] wrote {out_dir/'phase4_specificity_heatmap.png'}")

    # ----- Diagnostic prints -----
    print()
    print(f"[p4] specificity (body Δ − random Δ) by layer × horizon:")
    print(f"[p4] {'layer':>6s}  " + "  ".join(f"{h[:7]:>8s}" for h in PERIOD_ORDER))
    for li, L in enumerate(layers):
        row = "  ".join(f"{M_spec[li, hi]:+8.3f}" for hi in range(len(PERIOD_ORDER)))
        print(f"[p4] {f'L{L}':>6s}  {row}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/phase4_causal.jsonl")
    p.add_argument("--in", dest="inp", default="results/phase4_causal.jsonl")
    p.add_argument("--out-dir", default="results/probe_traj_anchored_L22_figs")
    p.add_argument("--plot", action="store_true",
                   help="run analysis (otherwise: run experiment)")
    p.add_argument("--body-probe-pkl",
                   default="results/qwen3.5-9b-base/body_probe_multilayer.pkl")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seeds", default="42,1337,7")
    p.add_argument("--domain-filter", default="")
    p.add_argument("--layers", default="18,20,22,24,26",
                   help="layers to steer at (must exist in body probe pkl)")
    p.add_argument("--alphas", default="-100,-60,-30,30,60,100",
                   help="steering strengths (signed)")
    p.add_argument("--n-controls", type=int, default=5,
                   help="number of random unit directions per layer")
    p.add_argument("--control-seed", type=int, default=0)
    args = p.parse_args()

    if args.plot:
        run_plot(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
