"""Stock-price-prediction battery for temporal probe steering.

Asks Qwen3.5-9B-Base to forecast a stock price at a stated horizon, while we
intervene with one of three steering modes (period CAA, interp, continuous).
Each cell of the grid produces one generation. Output is JSONL.

Grid:
  stocks (real + fake) × horizons × backgrounds × steering configs × seeds

Run from project root:
  python3 experiments/stock_battery.py --out results/stock_battery_v1.jsonl

Smaller pilot:
  python3 experiments/stock_battery.py --out ... --backgrounds standard --seeds 42

Larger sweep:
  python3 experiments/stock_battery.py --out ... --include-strength-sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make repo root importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from steering import SteeringEngine, engine_from_env


# ---------------- data ------------------------------------------------------

REAL_STOCKS = [
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "sector": "technology",
     "approx_price": 145.0, "real": True},
    {"ticker": "AAPL", "name": "Apple Inc.",         "sector": "technology",
     "approx_price": 230.0, "real": True},
    {"ticker": "TSLA", "name": "Tesla, Inc.",        "sector": "automotive",
     "approx_price": 240.0, "real": True},
    {"ticker": "JNJ",  "name": "Johnson & Johnson",  "sector": "healthcare",
     "approx_price": 160.0, "real": True},
]

FAKE_STOCKS = [
    {"ticker": "HELIX", "name": "Helix Therapeutics", "sector": "biotech",
     "approx_price": 50.0, "real": False},
    {"ticker": "ZNTH",  "name": "Zenith Energy Corp", "sector": "energy",
     "approx_price": 35.0, "real": False},
    {"ticker": "ATLS",  "name": "Atlas Logistics",    "sector": "logistics",
     "approx_price": 80.0, "real": False},
    {"ticker": "STLA",  "name": "Stella Robotics",    "sector": "technology",
     "approx_price": 120.0, "real": False},
]

HORIZONS = [
    {"label": "tomorrow",         "days": 1},
    {"label": "in one week",      "days": 7},
    {"label": "in one month",     "days": 30},
    {"label": "in one year",      "days": 365},
    {"label": "in ten years",     "days": 3650},
]

# Backgrounds: how much / what kind of context the prompt provides.
BACKGROUNDS = {
    "minimal": "{ticker} ({name}) is currently trading at ${price:.2f}.",

    "standard": ("{ticker} ({name}) operates in the {sector} sector "
                 "and is currently trading at ${price:.2f}. "
                 "Recent trading has been steady."),

    "misleading": ("{ticker} ({name}) operates in the {sector} sector. "
                   "Despite weak fundamentals and declining quarterly "
                   "revenue, the stock has surged to ${price:.2f} on "
                   "speculative buying."),
}


def build_prompt(stock: dict, horizon: dict, bg_name: str) -> str:
    bg = BACKGROUNDS[bg_name].format(
        ticker=stock["ticker"], name=stock["name"],
        sector=stock["sector"], price=stock["approx_price"],
    )
    # Completion-style "forecast list" prompt. The model has to commit a number
    # as the very next token after "$" — no room to spiral in <think> blocks
    # or copy from few-shot examples (we tried boxed/few-shot in iteration; it
    # invited Qwen-Base to loop or leak example numbers). The first number IS
    # the answer because the prompt structure forces it.
    return (
        f"{bg}\n\n"
        f"Forecast:\n"
        f"- Current price: ${stock['approx_price']:.2f}\n"
        f"- {horizon['label'].capitalize()}: $"
    )


# ---------------- steering configs -----------------------------------------

def steering_configs(include_strength_sweep: bool = False) -> list[dict]:
    cfgs: list[dict] = [{"name": "baseline", "mode": "baseline"}]

    # Mode 1: per-period
    period_alpha_pairs = [
        ("tonight",   +60.0),
        ("tonight",   -60.0),
        ("one_year",  +60.0),
        ("a_decade",  +60.0),
    ]
    for period, a in period_alpha_pairs:
        sign = "+" if a >= 0 else "-"
        cfgs.append({
            "name": f"periods/{period}/{sign}{abs(int(a))}",
            "mode": "periods",
            "alphas": {f"period_caa/{period}@22": a},
        })

    # Mode 2: interp time × strength
    for t in [0.0, 2.5, 5.0]:
        cfgs.append({"name": f"interp/t={t:.1f}/s=60",
                     "mode": "interp", "time": t, "strength": 60.0})

    # Mode 3: continuous time × strength
    for t in [0.0, 2.5, 5.0]:
        cfgs.append({"name": f"continuous/t={t:.1f}/s=60",
                     "mode": "continuous", "time": t, "strength": 60.0})

    if include_strength_sweep:
        # Strength dose-response on a single time point, both interp and continuous
        for s in [20.0, 40.0, 60.0, 100.0]:
            cfgs.append({"name": f"interp/t=5.0/s={int(s)}",
                         "mode": "interp", "time": 5.0, "strength": s})
            cfgs.append({"name": f"interp/t=0.0/s={int(s)}",
                         "mode": "interp", "time": 0.0, "strength": s})
            cfgs.append({"name": f"continuous/t=5.0/s={int(s)}",
                         "mode": "continuous", "time": 5.0, "strength": s})
            cfgs.append({"name": f"continuous/t=0.0/s={int(s)}",
                         "mode": "continuous", "time": 0.0, "strength": s})

    return cfgs


def build_steering_kwargs(engine: SteeringEngine, cfg: dict) -> dict:
    if cfg["mode"] == "baseline":
        return {"alphas": {}}
    if cfg["mode"] == "periods":
        return {"alphas": cfg["alphas"]}
    if cfg["mode"] == "interp":
        return {"steering_vectors": engine.vec_interp(cfg["time"], cfg["strength"])}
    if cfg["mode"] == "continuous":
        return {"steering_vectors": engine.vec_continuous(cfg["time"], cfg["strength"])}
    raise ValueError(f"unknown mode {cfg['mode']}")


# ---------------- runner ---------------------------------------------------

def run_battery(args):
    print("[battery] loading engine...")
    engine = engine_from_env()

    if not engine.probes:
        print("[battery] WARNING: no period probes loaded — Mode 1 will be no-op")
    if engine.continuous_direction is None:
        print("[battery] WARNING: no continuous probe loaded — Mode 3 will be no-op")

    stocks = list(REAL_STOCKS)
    if args.include_fake:
        stocks += FAKE_STOCKS
    if args.stocks:
        wanted = set(args.stocks.split(","))
        stocks = [s for s in stocks if s["ticker"] in wanted]

    horizons = HORIZONS[: args.n_horizons] if args.n_horizons > 0 else HORIZONS
    backgrounds = [b.strip() for b in args.backgrounds.split(",") if b.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    cfgs = steering_configs(include_strength_sweep=args.include_strength_sweep)
    if args.steering_filter:
        cfgs = [c for c in cfgs if args.steering_filter in c["name"]]

    total = len(stocks) * len(horizons) * len(backgrounds) * len(cfgs) * len(seeds)
    print(f"[battery] grid: {len(stocks)} stocks × {len(horizons)} horizons × "
          f"{len(backgrounds)} bgs × {len(cfgs)} steering × {len(seeds)} seeds "
          f"= {total} generations")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    i = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for stock in stocks:
            for horizon in horizons:
                for bg_name in backgrounds:
                    prompt = build_prompt(stock, horizon, bg_name)
                    for cfg in cfgs:
                        kwargs = build_steering_kwargs(engine, cfg)
                        for seed in seeds:
                            i += 1
                            gen = ""
                            for chunk in engine.generate_stream(
                                prompt, args.max_tokens,
                                temperature=args.temp, top_p=args.top_p,
                                seed=seed, **kwargs,
                            ):
                                gen = chunk
                            record = {
                                "i": i,
                                "stock": stock,
                                "horizon": horizon,
                                "background": bg_name,
                                "steering": cfg,
                                "seed": seed,
                                "prompt": prompt,
                                "generation": gen,
                                "max_tokens": args.max_tokens,
                                "temperature": args.temp,
                                "top_p": args.top_p,
                            }
                            f.write(json.dumps(record) + "\n")
                            f.flush()
                            if i % 10 == 0 or i == total:
                                elapsed = time.time() - t0
                                rate = i / elapsed if elapsed > 0 else 0.0
                                eta = (total - i) / rate if rate > 0 else 0.0
                                print(f"[battery] {i}/{total} ({100*i/total:.1f}%) "
                                      f"rate={rate:.2f}/s ETA={eta/60:.1f}min")
    print(f"[battery] done. wrote {out_path} ({i} records)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/stock_battery_v1.jsonl")
    p.add_argument("--max-tokens", type=int, default=40)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seeds", default="42,1337", help="comma-separated seeds")
    p.add_argument("--include-fake", action="store_true", default=True,
                   help="include fake stocks (default: yes)")
    p.add_argument("--no-fake", dest="include_fake", action="store_false",
                   help="real stocks only")
    p.add_argument("--stocks", default="",
                   help="comma-separated tickers to restrict to (e.g. NVDA,HELIX)")
    p.add_argument("--n-horizons", type=int, default=0,
                   help="limit to first N horizons (0 = all)")
    p.add_argument("--backgrounds", default="minimal,standard,misleading",
                   help="comma-separated background variants")
    p.add_argument("--steering-filter", default="",
                   help="only steering configs whose name contains this substring")
    p.add_argument("--include-strength-sweep", action="store_true",
                   help="add α dose-response cells for interp & continuous modes")
    args = p.parse_args()
    run_battery(args)


if __name__ == "__main__":
    main()
