"""Chained-generation probe trajectory: ONE CoT, six horizons in sequence.

After each phase's content, the script injects the next phase header and
asks the model to continue. The result is a single concatenated trace
that progresses Tonight → Tomorrow → Week → Month → Year → Decade. We
forward-pass the whole trace once and read the L22 probe at every token,
then aggregate per phase.

This tests the slides-proposal claim: does the probe track the model's
*currently-reasoned-about* horizon as it shifts within a single forward
pass? The killer plot is one line that climbs in six steps.

Run on Vast:
  STEER_LAYER=22 python3 experiments/probe_trajectory_chained.py \\
      --out results/probe_traj_chained.jsonl

Then plot:
  python3 experiments/probe_trajectory_chained.py --plot \\
      --in results/probe_traj_chained.jsonl \\
      --out-dir results/probe_traj_chained_figs
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
from experiments.probe_trajectory import tokens_in_span, random_unit_directions


DOMAINS = [
    ("surgery", "Plan how to recover after a major surgery."),
    ("startup", "Plan how to launch a successful technology startup."),
    ("career",  "Advise a recent college graduate planning their career."),
    ("climate", "Help a city government plan its response to climate change."),
    ("fitness", "Design a fitness plan for someone starting from scratch."),
]

# (label, header). Header gets injected before each phase's body generation.
PHASES = [
    ("tonight",   "Phase 1 — Tonight:"),
    ("tomorrow",  "Phase 2 — Tomorrow:"),
    ("one_week",  "Phase 3 — Next week:"),
    ("one_month", "Phase 4 — Next month:"),
    ("one_year",  "Phase 5 — Next year:"),
    ("a_decade",  "Phase 6 — A decade later:"),
]

PHASE_COLORS = {
    "tonight":   "#9333ea",
    "tomorrow":  "#3b82f6",
    "one_week":  "#06b6d4",
    "one_month": "#10b981",
    "one_year":  "#f59e0b",
    "a_decade":  "#ef4444",
}
PHASE_ORDER = [p[0] for p in PHASES]


# ---------------- chained generation ---------------------------------------

def chained_generate(engine, base_prompt: str, tokens_per_phase: int,
                     seed: int, temp: float, top_p: float,
                     reverse: bool = False,
                     random_order: bool = False) -> tuple[str, list[dict]]:
    """Run six chained phases. Each phase: append header, generate body, repeat.

    Order modes:
      - default: tonight → decade
      - reverse=True: decade → tonight (semantic reverse, position grows)
      - random_order=True: deterministic shuffle by seed (decisive test —
        per-horizon projections should match horizon regardless of position)
    """
    if random_order:
        import random as _random
        rng = _random.Random(int(seed) * 7919 + 13)
        phases = list(PHASES)
        rng.shuffle(phases)
        suffix = " in mixed order."
    elif reverse:
        phases = list(reversed(PHASES))
        suffix = " in reverse order, latest first."
    else:
        phases = PHASES
        suffix = " in order, earliest first."
    text = base_prompt.strip() + "\nThis plan walks through six time scales" + suffix
    phase_records = []

    for i, (label, header) in enumerate(phases):
        header_block = f"\n\n{header}\n"
        char_header_start = len(text)
        text += header_block
        char_body_start = len(text)

        gen = ""
        for chunk in engine.generate_stream(
            text, tokens_per_phase,
            temperature=temp, top_p=top_p,
            seed=seed + i,  # different seed per phase to avoid trivial sampling reuse
            alphas={},
        ):
            gen = chunk
        text += gen
        char_body_end = len(text)

        phase_records.append({
            "label": label,
            "header": header,
            "char_header_start": char_header_start,
            "char_body_start": char_body_start,
            "char_body_end": char_body_end,
            "body_text": gen,
        })

    return text, phase_records


# ---------------- experiment runner ----------------------------------------

def run_experiment(args):
    print("[chain] loading engine...")
    engine = engine_from_env()
    if engine.continuous_direction is None:
        raise RuntimeError("Continuous probe not loaded — set STEER_PROBES_CONTINUOUS")
    layer = int(engine.continuous_layer)
    print(f"[chain] continuous probe at L{layer}, ||w||={engine.continuous_w_norm:.2f}")

    import torch as _torch
    rand_dirs_np = random_unit_directions(engine.d_model, args.n_controls, seed=args.control_seed)
    rand_dirs = _torch.tensor(rand_dirs_np, dtype=engine._dtype).to(engine._device)
    print(f"[chain] {args.n_controls} random control directions @ L{layer}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    domains = DOMAINS
    if args.domain_filter:
        domains = [d for d in domains if args.domain_filter in d[0]]
    total = len(domains) * len(seeds)
    print(f"[chain] running {len(domains)} domains × {len(seeds)} seeds = {total} chained traces")

    with open(out_path, "w") as fout:
        gi = 0
        for dom_id, dom_text in domains:
            for seed in seeds:
                gi += 1
                print(f"\n[chain] [{gi}/{total}] {dom_id} seed={seed}")
                full_text, phase_recs = chained_generate(
                    engine, dom_text, args.tokens_per_phase, seed,
                    args.temp, args.top_p,
                    reverse=args.reverse,
                    random_order=args.random_order,
                )

                # Single forward pass on full chained text
                traj = engine.read_probe_trajectory(
                    full_text, layer=layer, extra_directions=rand_dirs,
                )
                offsets = traj["offsets"]
                proj = traj["projection"]
                ctrl_proj = traj["extra_projections"]

                # Per-phase aggregation (body tokens only, skipping header)
                phase_out = []
                for ph in phase_recs:
                    t_lo, t_hi = tokens_in_span(
                        offsets, ph["char_body_start"], ph["char_body_end"],
                    )
                    if t_lo < 0:
                        continue
                    body_proj = proj[t_lo:t_hi]
                    body_ctrl = ctrl_proj[:, t_lo:t_hi]
                    phase_out.append({
                        "label": ph["label"],
                        "header": ph["header"],
                        "body_text": ph["body_text"],
                        "char_body_start": ph["char_body_start"],
                        "char_body_end": ph["char_body_end"],
                        "tok_body_range": [int(t_lo), int(t_hi)],
                        "proj_mean":  float(np.mean(body_proj)) if body_proj.size else None,
                        "proj_last":  float(body_proj[-1])     if body_proj.size else None,
                        "ctrl_proj_mean": body_ctrl.mean(axis=1).tolist() if body_ctrl.size else None,
                    })

                means_str = " ".join(
                    f"{p['label'][:4]}={p['proj_mean']:+.2f}" for p in phase_out if p["proj_mean"] is not None
                )
                print(f"[chain]   tokens={len(proj)}  phase means: {means_str}")

                record = {
                    "domain": dom_id,
                    "seed": seed,
                    "reverse": bool(args.reverse),
                    "random_order": bool(args.random_order),
                    "phase_order": [p["label"] for p in phase_recs],
                    "layer": layer,
                    "w_norm": engine.continuous_w_norm,
                    "intercept": engine.continuous_intercept,
                    "tokens_per_phase": args.tokens_per_phase,
                    "temperature": args.temp,
                    "top_p": args.top_p,
                    "control_seed": args.control_seed,
                    "full_text": full_text,
                    "phases": phase_out,
                    "tokens": traj["tokens"],
                    "token_offsets": offsets,
                    "projection": proj.tolist(),
                    "control_projection": ctrl_proj.tolist(),
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()

    print(f"\n[chain] done. wrote {out_path}")


# ---------------- plotting --------------------------------------------------

def load_records(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


PHASE_LOG_DAYS = np.array([-0.602, 0.000, 0.845, 1.477, 2.562, 3.562])


def _trajectory_spearman(traj: np.ndarray) -> float:
    n = len(traj)
    rx = np.argsort(np.argsort(traj))
    return float(np.corrcoef(rx, np.arange(n))[0, 1])


def _trajectory_alignment(traj: np.ndarray) -> float:
    """Dot product of (centered) trajectory with (centered) log_days targets.

    Sensitive to BOTH monotonicity AND magnitude. Scale: positive → trajectory
    rises with horizon; near-zero → flat or anti-aligned; large → both well
    correlated and large absolute swing across phases.
    """
    t = traj - traj.mean()
    d = PHASE_LOG_DAYS - PHASE_LOG_DAYS.mean()
    return float(np.dot(t, d))


def plot_phase_trajectory(records: list[dict], out_path: Path):
    """Headline plot: trajectory line + control envelope; right panel: monotonicity-ρ distribution."""
    by_phase: dict[str, list[float]] = {h: [] for h in PHASE_ORDER}
    by_phase_ctrl: dict[str, list[list[float]]] = {h: [] for h in PHASE_ORDER}
    for r in records:
        for ph in r["phases"]:
            if ph["proj_mean"] is None:
                continue
            by_phase[ph["label"]].append(ph["proj_mean"])
            if ph["ctrl_proj_mean"]:
                by_phase_ctrl[ph["label"]].append(ph["ctrl_proj_mean"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                   gridspec_kw={"width_ratios": [3, 1.5]})
    x = np.arange(len(PHASE_ORDER))
    n_traces = max(len(by_phase[h]) for h in PHASE_ORDER)

    # Per-trace lines (faint)
    for ti in range(n_traces):
        ys = [by_phase[h][ti] if ti < len(by_phase[h]) else np.nan for h in PHASE_ORDER]
        ax.plot(x, ys, color="#1F6FEB", alpha=0.18, linewidth=1.0)

    # Mean line + SEM
    means = np.array([float(np.mean(by_phase[h])) for h in PHASE_ORDER])
    sems  = np.array([float(np.std(by_phase[h]) / max(1, np.sqrt(len(by_phase[h]))))
                      for h in PHASE_ORDER])
    ax.errorbar(x, means, yerr=sems, marker="o", color="#1F6FEB",
                linewidth=2.6, markersize=10, capsize=4,
                label=f"continuous probe (mean ± SEM, n={n_traces} traces)",
                zorder=10)

    # Random-direction null: per-direction trajectory of phase means
    K = len(by_phase_ctrl[PHASE_ORDER[0]][0]) if by_phase_ctrl[PHASE_ORDER[0]] else 0
    probe_rho = _trajectory_spearman(means)
    ctrl_rhos: list[float] = []
    if K:
        ctrl_arr = np.zeros((len(PHASE_ORDER), K))
        for i, h in enumerate(PHASE_ORDER):
            arr = np.array(by_phase_ctrl[h])  # (n_traces, K)
            ctrl_arr[i] = arr.mean(axis=0)
        # Percentile envelope (5th-95th) — robust to K
        p05 = np.percentile(ctrl_arr, 5, axis=1)
        p95 = np.percentile(ctrl_arr, 95, axis=1)
        ax.fill_between(x, p05, p95, color="gray", alpha=0.20,
                        label=f"random-direction null (5–95th pct, K={K})")
        for k in range(K):
            ax.plot(x, ctrl_arr[:, k], color="gray", alpha=0.20, linewidth=0.6)
        # Per-direction trajectory monotonicity
        ctrl_rhos = [_trajectory_spearman(ctrl_arr[:, k]) for k in range(K)]

    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(PHASE_ORDER, fontsize=10)
    ax.set_xlabel("phase within a single chained CoT")
    ax.set_ylabel(f"mean probe projection (L{records[0]['layer']})")
    ax.set_title("Probe trajectory across six horizons in one chained generation")
    ax.legend(loc="upper left", fontsize=9)

    # Right panel: magnitude-weighted alignment with log_days targets
    probe_align = _trajectory_alignment(means)
    ctrl_aligns: list[float] = []
    if K:
        ctrl_aligns = [_trajectory_alignment(ctrl_arr[:, k]) for k in range(K)]
        ax2.hist(ctrl_aligns, bins=12, color="gray", alpha=0.6,
                 edgecolor="white", label=f"random (K={K})")
        ax2.axvline(probe_align, color="#1F6FEB", linewidth=2.8,
                    label=f"probe = {probe_align:+.2f}")
        n_above = sum(1 for v in ctrl_aligns if v >= probe_align)
        p_perm = (n_above + 1) / (K + 1)
        ax2.set_xlabel("trajectory · log_days\n(monotonicity × magnitude)")
        ax2.set_ylabel("count")
        ax2.set_title(f"Magnitude-weighted alignment\nperm. p={p_perm:.3f}")
        ax2.axvline(0, color="black", linewidth=0.3, alpha=0.4)
        ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[chain] wrote {out_path}")
    print(f"[chain]   probe ρ (rank vs ideal): {probe_rho:+.3f}  "
          f"alignment (·log_days): {probe_align:+.3f}")
    if ctrl_rhos:
        n_rho_above = sum(1 for r in ctrl_rhos if r >= probe_rho)
        n_aln_above = sum(1 for v in ctrl_aligns if v >= probe_align)
        print(f"[chain]   K={K} controls: rho median={np.median(ctrl_rhos):+.3f} "
              f"max={np.max(ctrl_rhos):+.3f} #≥probe={n_rho_above}")
        print(f"[chain]   K={K} controls: align median={np.median(ctrl_aligns):+.3f} "
              f"max={np.max(ctrl_aligns):+.3f} #≥probe={n_aln_above}  "
              f"(perm p={(n_aln_above+1)/(K+1):.3f})")
    return means, sems


def plot_per_trace_sparkline(record: dict, out_path: Path):
    """Per-token probe trajectory with phase shading."""
    proj = np.array(record["projection"])
    ctrl = np.array(record["control_projection"])  # (K, T)
    offsets = record["token_offsets"]
    T = len(proj)

    fig, (ax_top, ax) = plt.subplots(
        2, 1, figsize=(14, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 4]},
    )

    for ph in record["phases"]:
        t_lo, t_hi = ph["tok_body_range"]
        c = PHASE_COLORS[ph["label"]]
        ax_top.barh(0, t_hi - t_lo, left=t_lo, color=c, height=0.8, alpha=0.85)
        # Shade body region in main panel
        ax.axvspan(t_lo, t_hi, color=c, alpha=0.10)

    ax_top.set_yticks([])
    ax_top.set_xlim(0, T)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[h]) for h in PHASE_ORDER]
    ax_top.legend(handles, PHASE_ORDER, fontsize=7, loc="upper right",
                  ncol=6, frameon=False)
    ax_top.set_title(f"{record['domain']} (seed={record['seed']}, L{record['layer']})",
                     fontsize=10, loc="left")

    ctrl_mean = ctrl.mean(axis=0)
    ctrl_std = ctrl.std(axis=0)
    ax.fill_between(np.arange(T), ctrl_mean - ctrl_std, ctrl_mean + ctrl_std,
                    color="gray", alpha=0.20, label="random ±1σ")
    ax.plot(np.arange(T), proj, color="#1F6FEB", linewidth=1.4,
            label="continuous probe")
    # Per-phase mean as horizontal segments
    for ph in record["phases"]:
        if ph["proj_mean"] is None:
            continue
        t_lo, t_hi = ph["tok_body_range"]
        ax.hlines(ph["proj_mean"], t_lo, t_hi, color="black", linewidth=2.0,
                  alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax.set_xlim(0, T)
    ax.set_xlabel("token index")
    ax.set_ylabel("residual · probe direction")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[chain] wrote {out_path}")


def run_plot(args):
    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(inp)
    print(f"[chain] loaded {len(records)} chained traces from {inp}")

    means, sems = plot_phase_trajectory(records, out_dir / "phase_trajectory.png")
    print(f"[chain] phase means: " +
          " ".join(f"{h}={m:+.3f}±{s:.2f}" for h, m, s in zip(PHASE_ORDER, means, sems)))

    # Per-trace sparkline (one per domain at first seed)
    seen = set()
    for r in records:
        if r["domain"] in seen:
            continue
        seen.add(r["domain"])
        plot_per_trace_sparkline(r, out_dir / f"sparkline_{r['domain']}.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/probe_traj_chained.jsonl")
    p.add_argument("--in", dest="inp", default="results/probe_traj_chained.jsonl")
    p.add_argument("--out-dir", default="results/probe_traj_chained_figs")
    p.add_argument("--plot", action="store_true",
                   help="run plotter (otherwise: run experiment)")
    p.add_argument("--tokens-per-phase", type=int, default=40)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seeds", default="42,1337,7",
                   help="comma-separated seeds")
    p.add_argument("--domain-filter", default="")
    p.add_argument("--n-controls", type=int, default=8)
    p.add_argument("--control-seed", type=int, default=0)
    p.add_argument("--reverse", action="store_true",
                   help="generate phases in reverse order (decade first).")
    p.add_argument("--random-order", action="store_true",
                   help="shuffle phase order per trace (decisive test — "
                        "per-horizon projections should match horizon regardless "
                        "of position/order).")
    args = p.parse_args()

    if args.plot:
        run_plot(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
