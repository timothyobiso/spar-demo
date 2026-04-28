"""Headless smoke test: instantiate the engine in mock mode and stream output.

Run:
    STEER_MOCK=1 uv run python test_mock.py
"""

from __future__ import annotations

import os

os.environ["STEER_MOCK"] = "1"

from steering import engine_from_env


def main() -> None:
    engine = engine_from_env()
    assert engine.mock, "expected mock mode"
    assert len(engine.probes) == 2, f"expected 2 mock probes, got {len(engine.probes)}"
    print(f"probes: {[p.key for p in engine.probes]}")

    alphas = {engine.probes[0].key: 12.0, engine.probes[1].key: -5.0}
    stream = engine.generate_stream(
        "The chef looked at the kitchen",
        max_new_tokens=20,
        alphas=alphas,
        temperature=0.8,
        top_p=0.9,
    )
    final = ""
    for chunk in stream:
        final = chunk
    print(f"final output ({len(final)} chars):")
    print(final)
    assert final, "empty stream"
    assert "mock" in final.lower(), "mock header missing"
    print("\nOK")


if __name__ == "__main__":
    main()
