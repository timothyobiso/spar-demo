"""Forward-vs-reverse chained-CoT comparison plot.

Decisive test for whether the probe trajectory in chained CoT is driven by
the model's currently-reasoned-about horizon (slides claim) or by a
position/cumulative-context/lexical confound. Run forward generation
(tonight → decade) and reverse generation (decade → tonight), then plot
projection vs horizon for both.

Predictions:
  - probe tracks horizon semantically  → forward CLIMBS, reverse DESCENDS
  - probe tracks position/length        → both CLIMB
  - probe is noise                      → both flat-ish

Run:
  python3 experiments/plot_chain_compare.py \\
      --forward results/probe_traj_chained_k30.jsonl \\
      --reverse results/probe_traj_chained_reverse.jsonl \\
      --out results/probe_traj_chained_k30_figs/forward_vs_reverse.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PHASE_ORDER = ["tonight", "tomorrow", "one_week", "one_month", "one_year", "a_decade"]
PHASE_LOG_DAYS = np.array([-0.602, 0.000, 0.845, 1.477, 2.562, 3.562])


def load(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def aggregate(records: list[dict]):
    """Return (means, sems) of probe projection per phase, plus per-direction
    control trajectories (avg across traces) of shape (6, K)."""
    by_phase: dict[str, list[float]] = {h: [] for h in PHASE_ORDER}
    by_phase_ctrl: dict[str, list[list[float]]] = {h: [] for h in PHASE_ORDER}
    for r in records:
        for ph in r["phases"]:
            if ph["proj_mean"] is None:
                continue
            by_phase[ph["label"]].append(ph["proj_mean"])
            if ph["ctrl_proj_mean"]:
                by_phase_ctrl[ph["label"]].append(ph["ctrl_proj_mean"])

    means = np.array([float(np.mean(by_phase[h])) for h in PHASE_ORDER])
    sems  = np.array([float(np.std(by_phase[h]) / max(1, np.sqrt(len(by_phase[h]))))
                      for h in PHASE_ORDER])

    K = len(by_phase_ctrl[PHASE_ORDER[0]][0]) if by_phase_ctrl[PHASE_ORDER[0]] else 0
    ctrl = np.zeros((len(PHASE_ORDER), K))
    for i, h in enumerate(PHASE_ORDER):
        arr = np.array(by_phase_ctrl[h])
        ctrl[i] = arr.mean(axis=0)
    return means, sems, ctrl


def alignment(traj: np.ndarray) -> float:
    t = traj - traj.mean()
    d = PHASE_LOG_DAYS - PHASE_LOG_DAYS.mean()
    return float(np.dot(t, d))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forward", required=True)
    p.add_argument("--reverse", required=True)
    p.add_argument("--random", default=None,
                   help="optional: random-order chained JSONL (for 3-way overlay)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    fwd = load(Path(args.forward))
    rev = load(Path(args.reverse))
    rnd = load(Path(args.random)) if args.random else None
    fwd_means, fwd_sems, fwd_ctrl = aggregate(fwd)
    rev_means, rev_sems, rev_ctrl = aggregate(rev)
    if rnd:
        rnd_means, rnd_sems, rnd_ctrl = aggregate(rnd)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                   gridspec_kw={"width_ratios": [3, 1.5]})
    x = np.arange(len(PHASE_ORDER))

    # Random null envelopes (5–95th pct) for both
    if fwd_ctrl.shape[1]:
        p05 = np.percentile(fwd_ctrl, 5, axis=1)
        p95 = np.percentile(fwd_ctrl, 95, axis=1)
        ax.fill_between(x, p05, p95, color="#1F6FEB", alpha=0.10,
                        label=f"forward null (5–95 pct, K={fwd_ctrl.shape[1]})")
    if rev_ctrl.shape[1]:
        p05 = np.percentile(rev_ctrl, 5, axis=1)
        p95 = np.percentile(rev_ctrl, 95, axis=1)
        ax.fill_between(x, p05, p95, color="#dc2626", alpha=0.10,
                        label=f"reverse null (5–95 pct, K={rev_ctrl.shape[1]})")

    ax.errorbar(x, fwd_means, yerr=fwd_sems, marker="o", color="#1F6FEB",
                linewidth=2.5, markersize=10, capsize=4,
                label=f"forward order  (n={len(fwd)} traces)",
                zorder=10)
    ax.errorbar(x, rev_means, yerr=rev_sems, marker="s", color="#dc2626",
                linewidth=2.5, markersize=10, capsize=4,
                label=f"reverse order  (n={len(rev)} traces)",
                zorder=10)
    if rnd:
        ax.errorbar(x, rnd_means, yerr=rnd_sems, marker="D", color="#15803D",
                    linewidth=2.5, markersize=10, capsize=4,
                    label=f"random order  (n={len(rnd)} traces)",
                    zorder=10)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(PHASE_ORDER, fontsize=10)
    ax.set_xlabel("horizon (semantic content of phase)")
    ax.set_ylabel(f"mean probe projection (L{fwd[0]['layer']})")
    ax.set_title("Forward vs reverse chained CoT — does probe follow horizon or position?")
    ax.legend(loc="upper left", fontsize=9)

    # Right panel: alignment statistic for forward vs reverse vs nulls
    fwd_aln = alignment(fwd_means)
    rev_aln = alignment(rev_means)
    fwd_ctrl_aln = [alignment(fwd_ctrl[:, k]) for k in range(fwd_ctrl.shape[1])]
    rev_ctrl_aln = [alignment(rev_ctrl[:, k]) for k in range(rev_ctrl.shape[1])]
    rnd_aln = alignment(rnd_means) if rnd else None
    rnd_ctrl_aln = [alignment(rnd_ctrl[:, k]) for k in range(rnd_ctrl.shape[1])] if rnd else []

    all_alns = fwd_ctrl_aln + rev_ctrl_aln + rnd_ctrl_aln + [fwd_aln, rev_aln]
    if rnd_aln is not None:
        all_alns.append(rnd_aln)
    bins = np.linspace(min(min(all_alns) - 1, -3), max(max(all_alns) + 1, +3), 15)
    ax2.hist(fwd_ctrl_aln, bins=bins, color="#1F6FEB", alpha=0.35,
             label=f"forward null (K={len(fwd_ctrl_aln)})", edgecolor="white")
    ax2.hist(rev_ctrl_aln, bins=bins, color="#dc2626", alpha=0.35,
             label=f"reverse null (K={len(rev_ctrl_aln)})", edgecolor="white")
    if rnd_ctrl_aln:
        ax2.hist(rnd_ctrl_aln, bins=bins, color="#15803D", alpha=0.35,
                 label=f"random null (K={len(rnd_ctrl_aln)})", edgecolor="white")
    ax2.axvline(fwd_aln, color="#1F6FEB", linewidth=2.6,
                label=f"forward probe = {fwd_aln:+.2f}")
    ax2.axvline(rev_aln, color="#dc2626", linewidth=2.6,
                label=f"reverse probe = {rev_aln:+.2f}")
    if rnd_aln is not None:
        ax2.axvline(rnd_aln, color="#15803D", linewidth=2.6,
                    label=f"random-order probe = {rnd_aln:+.2f}")
    ax2.axvline(0, color="black", linewidth=0.3, alpha=0.4)
    ax2.set_xlabel("trajectory · log_days")
    ax2.set_ylabel("count")
    ax2.set_title("Magnitude-weighted alignment\n(positive = climbs with horizon)")
    ax2.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"[compare] wrote {args.out}")
    print(f"[compare]   forward alignment: {fwd_aln:+.3f}  (random max {max(fwd_ctrl_aln):+.3f})")
    print(f"[compare]   reverse alignment: {rev_aln:+.3f}  (random max {max(rev_ctrl_aln):+.3f})")
    if rnd_aln is not None:
        n_above = sum(1 for v in rnd_ctrl_aln if v >= rnd_aln)
        p_perm = (n_above + 1) / (len(rnd_ctrl_aln) + 1)
        print(f"[compare]   random-order alignment: {rnd_aln:+.3f}  "
              f"(random max {max(rnd_ctrl_aln):+.3f}, perm p={p_perm:.3f})")
    print(f"[compare]   key test: random-order alignment should be ≈ forward IF probe is purely semantic")


if __name__ == "__main__":
    main()
