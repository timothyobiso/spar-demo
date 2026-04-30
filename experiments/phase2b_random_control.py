"""Phase 2b: random-direction matched-norm control for the Phase 2 steering test.

For each "tonight"-anchored prompt, replace v(target) at L22 with K random
unit directions, matched in norm. If steering with any of those reproduces
Phase 2's tonight/tomorrow suppression and decade boost, the effect isn't
probe-specific — generic perturbation explains it. If only the probe-direction
v(target) does this, the Phase 2 causal claim survives.

Run on Vast:
  STEER_LAYER=22 python3 experiments/phase2b_random_control.py \\
      --out results/phase2b_random.jsonl --seeds 42,1337,7

Then compare:
  python3 experiments/phase2b_random_control.py --plot \\
      --probe-in results/phase2_causal.jsonl \\
      --random-in results/phase2b_random.jsonl \\
      --out-dir results/probe_traj_anchored_L22_figs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from steering import engine_from_env
from experiments.phase2_causal import (
    DOMAINS, SOURCE_HEADER, SOURCE_LABEL, ALPHAS, PERIOD_ORDER,
    HORIZON_PATTERNS, count_horizon_words,
)


def run_experiment(args):
    print("[2b] loading engine...")
    engine = engine_from_env()
    if not engine.probes:
        raise RuntimeError("No period CAA probes loaded — check STEER_PROBES")
    layer = engine.probes[0].layer
    print(f"[2b] period probes at L{layer}; building K={args.n_controls} random unit directions")

    import torch as _torch
    rng = np.random.default_rng(args.control_seed)
    rand_dirs = rng.standard_normal((args.n_controls, engine.d_model)).astype(np.float32)
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
    rand_dirs_t = _torch.tensor(rand_dirs, dtype=engine._dtype).to(engine._device)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    domains = DOMAINS
    if args.domain_filter:
        domains = [d for d in domains if args.domain_filter in d[0]]
    n_per_prompt = args.n_controls * len(ALPHAS)
    total = len(domains) * len(seeds) * n_per_prompt
    print(f"[2b] {len(domains)} domains × {len(seeds)} seeds × {n_per_prompt} steered = {total} gens")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    i = 0
    with open(out_path, "w") as fout:
        for dom_id, dom_text in domains:
            for seed in seeds:
                base_prompt = f"{dom_text}\n\n{SOURCE_HEADER}"
                for k in range(args.n_controls):
                    v = rand_dirs_t[k]
                    for alpha in ALPHAS:
                        # Build steering vector at L22, magnitude alpha
                        steering_vectors = {layer: v * float(alpha)}
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
                            "i": i, "domain": dom_id, "seed": seed, "layer": layer,
                            "source": SOURCE_LABEL,
                            "rand_idx": k, "alpha": alpha,
                            "prompt": base_prompt, "generation": gen,
                            "counts": count_horizon_words(gen),
                        }
                        fout.write(json.dumps(rec) + "\n"); fout.flush()
                        if i % 30 == 0 or i == total:
                            print(f"[2b] {i}/{total}  {dom_id}#{seed} rand{k} α={alpha}")

    print(f"\n[2b] done. wrote {out_path}")


def _load(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run_plot(args):
    probe_recs = _load(Path(args.probe_in))
    rand_recs  = _load(Path(args.random_in))
    print(f"[2b] probe records: {len(probe_recs)}, random records: {len(rand_recs)}")

    # Baselines from probe file (target=None)
    base_counts = {}
    for r in probe_recs:
        if r["target"] is None:
            base_counts[(r["domain"], r["seed"])] = r["counts"]

    # Probe-direction Δ averaged over targets and α (any nonzero α)
    probe_deltas = {h: [] for h in PERIOD_ORDER}
    for r in probe_recs:
        if r["target"] is None:
            continue
        b = base_counts.get((r["domain"], r["seed"]))
        if not b:
            continue
        for h in PERIOD_ORDER:
            probe_deltas[h].append(r["counts"][h] - b[h])

    # Random-direction Δ at the same prompts (no baseline in this file → use probe baselines)
    rand_deltas = {h: [] for h in PERIOD_ORDER}
    for r in rand_recs:
        b = base_counts.get((r["domain"], r["seed"]))
        if not b:
            continue
        for h in PERIOD_ORDER:
            rand_deltas[h].append(r["counts"][h] - b[h])

    # Plot: per-horizon mean Δ, probe vs random
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(PERIOD_ORDER))
    width = 0.36

    probe_mean = np.array([np.mean(probe_deltas[h]) for h in PERIOD_ORDER])
    probe_sem  = np.array([np.std(probe_deltas[h])  / max(1, np.sqrt(len(probe_deltas[h])))
                           for h in PERIOD_ORDER])
    rand_mean  = np.array([np.mean(rand_deltas[h])  for h in PERIOD_ORDER])
    rand_sem   = np.array([np.std(rand_deltas[h])   / max(1, np.sqrt(len(rand_deltas[h])))
                           for h in PERIOD_ORDER])

    ax.bar(x - width / 2, probe_mean, width, yerr=probe_sem, color="#1F6FEB",
           edgecolor="white", capsize=3,
           label=f"probe-direction steering (n={len(probe_deltas['tonight'])})")
    ax.bar(x + width / 2, rand_mean, width, yerr=rand_sem, color="gray",
           edgecolor="white", capsize=3,
           label=f"random-direction steering (n={len(rand_deltas['tonight'])})")

    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(PERIOD_ORDER, fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Δ (steered − baseline) horizon-lexicon counts")
    ax.set_title("Phase 2b: probe-direction vs random-direction steering "
                 "(matched norm, all α pooled)")
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    out = Path(args.out_dir) / "phase2b_probe_vs_random.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[2b] wrote {out}")

    # Diagnostics
    print()
    print(f"[2b] {'horizon':>10s}  {'probe Δ':>9s}  {'random Δ':>9s}  {'specificity':>12s}")
    for i_h, h in enumerate(PERIOD_ORDER):
        spec = probe_mean[i_h] - rand_mean[i_h]
        print(f"[2b] {h:>10s}  {probe_mean[i_h]:>+9.3f}  {rand_mean[i_h]:>+9.3f}  {spec:>+12.3f}")
    print()
    print("[2b] specificity = probe Δ − random Δ. Probe-specific effect should be NONZERO.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/phase2b_random.jsonl")
    p.add_argument("--probe-in", default="results/phase2_causal.jsonl")
    p.add_argument("--random-in", default="results/phase2b_random.jsonl")
    p.add_argument("--out-dir", default="results/probe_traj_anchored_L22_figs")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seeds", default="42,1337,7")
    p.add_argument("--domain-filter", default="")
    p.add_argument("--n-controls", type=int, default=5,
                   help="number of random unit directions per prompt")
    p.add_argument("--control-seed", type=int, default=0)
    args = p.parse_args()

    if args.plot:
        run_plot(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
