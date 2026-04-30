"""Phase 2: causal steering test for the L22 horizon direction.

Question: does the L22 horizon direction *causally* drive horizon-relevant
generation, or is the probe just correlationally reading lexical features?

Protocol (Rimsky-CAA + Zhang/Nanda patching, applied to a horizon-CoT setting):
  - All prompts anchored to "tonight" — the model's default attractor here.
  - For each prompt, generate baseline (α=0) + steered (α × v(target) at L22)
    where target ∈ {tomorrow, one_week, one_month, one_year, a_decade}.
  - α sweep: {30, 60, 100} for dose-response.
  - Score each generation by counting horizon-lexicon hits per period
    (regex over decade-words, year-words, etc.).

Causal claim survives if steering toward target_X reliably *increases*
target_X-lexicon counts and *decreases* tonight-lexicon counts in the
continuation, across prompts. If not, the probe is correlation-only.

Run on Vast:
  STEER_LAYER=22 python3 experiments/phase2_causal.py \\
      --out results/phase2_causal.jsonl --seeds 42,1337,7

Then plot:
  python3 experiments/phase2_causal.py --plot \\
      --in results/phase2_causal.jsonl \\
      --out-dir results/probe_traj_anchored_L22_figs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent.parent))
from steering import engine_from_env


# ---------------- horizon lexicons (richer than probe_trajectory) ----------

HORIZON_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("tonight",   re.compile(
        r"\b(tonight|this evening|next few hours|next couple of hours|"
        r"this night|after dinner|before bed|in the next hour|"
        r"right now|tonight'?s)\b", re.I)),
    ("tomorrow",  re.compile(
        r"\b(tomorrow|next day|tomorrow morning|tomorrow afternoon|"
        r"in 24 hours|first thing in the morning)\b", re.I)),
    ("one_week",  re.compile(
        r"\b(next week|in a week|in one week|over the (?:next|coming) week|"
        r"seven days|next monday|next tuesday|next friday|"
        r"by next (?:monday|tuesday|wednesday|thursday|friday)|"
        r"in the next few days|next weekend)\b", re.I)),
    ("one_month", re.compile(
        r"\b(next month|in a month|over the (?:next|coming) month|"
        r"thirty days|in 30 days|in four weeks|in three weeks|"
        r"a month from now|in the coming weeks)\b", re.I)),
    ("one_year",  re.compile(
        r"\b(next year|in a year|in one year|over the (?:next|coming) year|"
        r"in twelve months|in 12 months|next spring|next fall|"
        r"by 2027|in 2027|in 2028|by 2028|a year from now|"
        r"in the coming months)\b", re.I)),
    ("a_decade",  re.compile(
        r"\b(decade|ten years|10 years|over the next decade|"
        r"in the next decade|by 2035|by 2040|in the 2030s|"
        r"a decade from now|in ten years|long.?term|in the long run)\b", re.I)),
]

PERIOD_ORDER = ["tonight", "tomorrow", "one_week", "one_month", "one_year", "a_decade"]
PERIOD_COLORS = {
    "tonight":   "#9333ea", "tomorrow":  "#3b82f6", "one_week":  "#06b6d4",
    "one_month": "#10b981", "one_year":  "#f59e0b", "a_decade":  "#ef4444",
}


def count_horizon_words(text: str) -> dict[str, int]:
    return {h: len(p.findall(text or "")) for h, p in HORIZON_PATTERNS}


# ---------------- experiment ---------------------------------------------

DOMAINS = [
    ("surgery", "Plan how to recover after a major surgery."),
    ("startup", "Plan how to launch a successful technology startup."),
    ("career",  "Advise a recent college graduate planning their career."),
    ("climate", "Help a city government plan its response to climate change."),
    ("fitness", "Design a fitness plan for someone starting from scratch."),
]

SOURCE_LABEL = "tonight"
SOURCE_HEADER = "Tonight, the most important thing is to"
TARGETS = ["tomorrow", "one_week", "one_month", "one_year", "a_decade"]
ALPHAS = [30.0, 60.0, 100.0]


def run_experiment(args):
    print("[phase2] loading engine...")
    engine = engine_from_env()
    if not engine.probes:
        raise RuntimeError("No period CAA probes loaded — check STEER_PROBES")
    layer = engine.probes[0].layer
    print(f"[phase2] period CAA probes at L{layer}: "
          + ", ".join(p.key for p in engine.probes))

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    domains = DOMAINS
    if args.domain_filter:
        domains = [d for d in domains if args.domain_filter in d[0]]
    n_baseline = len(domains) * len(seeds)
    n_steered  = n_baseline * len(TARGETS) * len(ALPHAS)
    total = n_baseline + n_steered
    print(f"[phase2] {n_baseline} baselines + {n_steered} steered = {total} generations")

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
                    "i": i, "domain": dom_id, "seed": seed, "layer": layer,
                    "source": SOURCE_LABEL, "target": None, "alpha": 0.0,
                    "prompt": base_prompt, "generation": gen,
                    "counts": count_horizon_words(gen),
                }
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                if i % 20 == 0 or i == total:
                    print(f"[phase2] {i}/{total}  baseline {dom_id}#{seed}")

                # Steered
                for target in TARGETS:
                    probe_key = f"period_caa/{target}@{layer}"
                    for alpha in ALPHAS:
                        gen = ""
                        for chunk in engine.generate_stream(
                            base_prompt, args.max_tokens,
                            temperature=args.temp, top_p=args.top_p,
                            seed=seed,
                            alphas={probe_key: alpha},
                        ):
                            gen = chunk
                        i += 1
                        rec = {
                            "i": i, "domain": dom_id, "seed": seed, "layer": layer,
                            "source": SOURCE_LABEL, "target": target, "alpha": alpha,
                            "prompt": base_prompt, "generation": gen,
                            "counts": count_horizon_words(gen),
                        }
                        fout.write(json.dumps(rec) + "\n"); fout.flush()
                        if i % 20 == 0 or i == total:
                            print(f"[phase2] {i}/{total}  {dom_id}#{seed} "
                                  f"→{target} α={alpha}")

    print(f"\n[phase2] done. wrote {out_path}")


# ---------------- analysis & plotting ------------------------------------

def load(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run_plot(args):
    records = load(Path(args.inp))
    print(f"[phase2] loaded {len(records)} records from {args.inp}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index baseline counts per (domain, seed) for paired comparison
    base_counts: dict[tuple[str, int], dict[str, int]] = {}
    for r in records:
        if r["target"] is None:
            base_counts[(r["domain"], r["seed"])] = r["counts"]

    # ----- Heatmap: target_steered × measured_horizon (averaged over α & seeds & domains) -----
    M = np.zeros((len(TARGETS), len(PERIOD_ORDER)))
    counts_n = np.zeros(len(TARGETS), dtype=int)
    delta_M = np.zeros((len(TARGETS), len(PERIOD_ORDER)))
    for r in records:
        t = r["target"]
        if t is None:
            continue
        # Average across all alphas (could split, but heatmap aggregates first)
        ti = TARGETS.index(t)
        for hi, h in enumerate(PERIOD_ORDER):
            M[ti, hi] += r["counts"][h]
            base = base_counts.get((r["domain"], r["seed"]), {}).get(h, 0)
            delta_M[ti, hi] += r["counts"][h] - base
        counts_n[ti] += 1
    M /= np.maximum(counts_n[:, None], 1)
    delta_M /= np.maximum(counts_n[:, None], 1)

    # ----- Plot 1: delta-counts heatmap (steered − baseline per horizon-word category) -----
    fig, ax = plt.subplots(figsize=(8, 4))
    vmax = max(abs(delta_M.min()), abs(delta_M.max()), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(delta_M, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(PERIOD_ORDER)))
    ax.set_xticklabels(PERIOD_ORDER, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(TARGETS)))
    ax.set_yticklabels(TARGETS, fontsize=9)
    ax.set_xlabel("measured horizon-lexicon (counts in generation)")
    ax.set_ylabel("steering target (v(target) at L22)")
    ax.set_title("Δ(steered − baseline) horizon-lexicon counts\n"
                 "Diagonal+ = causal: steering toward X yields more X-words")
    for i in range(delta_M.shape[0]):
        for j in range(delta_M.shape[1]):
            v = delta_M[i, j]
            if abs(v) > 0.05:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > vmax * 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Δ count vs baseline")
    plt.tight_layout()
    plt.savefig(out_dir / "phase2_delta_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"[phase2] wrote {out_dir/'phase2_delta_heatmap.png'}")

    # ----- Plot 2: dose-response — per target, α vs (target words − tonight words) -----
    fig, ax = plt.subplots(figsize=(8, 5))
    alphas_sorted = sorted({r["alpha"] for r in records if r["target"] is not None})
    alphas_x = [0.0] + alphas_sorted

    for target in TARGETS:
        ys = []
        ys_err = []
        # Per α, collect (target_words - tonight_words) values
        for alpha in alphas_x:
            vals = []
            for r in records:
                if alpha == 0.0:
                    if r["target"] is None:
                        diff = r["counts"][target] - r["counts"]["tonight"]
                        vals.append(diff)
                else:
                    if r["target"] == target and r["alpha"] == alpha:
                        diff = r["counts"][target] - r["counts"]["tonight"]
                        vals.append(diff)
            ys.append(float(np.mean(vals)) if vals else 0.0)
            ys_err.append(float(np.std(vals) / max(1, np.sqrt(len(vals)))) if vals else 0.0)
        ax.errorbar(alphas_x, ys, yerr=ys_err, marker="o", linewidth=2.0,
                    color=PERIOD_COLORS[target],
                    label=f"→ {target}", capsize=3)

    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax.set_xlabel("steering strength α")
    ax.set_ylabel("(target-lexicon − tonight-lexicon) word counts")
    ax.set_title("Dose-response: causal shift away from tonight toward target\n"
                 "(positive = steering pulls generation toward target)")
    ax.legend(fontsize=9, title="target horizon")
    plt.tight_layout()
    plt.savefig(out_dir / "phase2_dose_response.png", dpi=150)
    plt.close(fig)
    print(f"[phase2] wrote {out_dir/'phase2_dose_response.png'}")

    # ----- Diagnostic prints -----
    print()
    print("[phase2] mean Δ-counts (steered − baseline), averaged over α and prompts:")
    print(f"[phase2] {'target':>10s}  " +
          "  ".join(f"{h[:7]:>8s}" for h in PERIOD_ORDER))
    for ti, t in enumerate(TARGETS):
        row = "  ".join(f"{delta_M[ti, hi]:+8.2f}" for hi in range(len(PERIOD_ORDER)))
        on_diag = delta_M[ti, PERIOD_ORDER.index(t)]
        print(f"[phase2] {t:>10s}  {row}    [diag={on_diag:+.2f}]")

    # Causal score: average diagonal Δ, average off-diagonal Δ
    diag_vals = [delta_M[ti, PERIOD_ORDER.index(t)] for ti, t in enumerate(TARGETS)]
    print()
    print(f"[phase2] mean diagonal Δ (causal target shift): {np.mean(diag_vals):+.3f}")
    print(f"[phase2] tonight-column Δ avg (should be NEGATIVE if steering away):")
    for ti, t in enumerate(TARGETS):
        print(f"[phase2]   →{t}: tonight-words Δ = {delta_M[ti, 0]:+.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/phase2_causal.jsonl")
    p.add_argument("--in", dest="inp", default="results/phase2_causal.jsonl")
    p.add_argument("--out-dir", default="results/probe_traj_anchored_L22_figs")
    p.add_argument("--plot", action="store_true",
                   help="run analysis (otherwise: run experiment)")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seeds", default="42,1337,7")
    p.add_argument("--domain-filter", default="")
    args = p.parse_args()

    if args.plot:
        run_plot(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
