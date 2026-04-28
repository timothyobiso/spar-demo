"""Causal steering engine for tam probes.

Loads a HF causal LM and a probes.pkl produced by the tam experiments, extracts
each probe's learned direction in residual-stream space, and generates text with
those directions added to specified layers via forward hooks.

probes.pkl structure expected (as emitted by tam/experiments/run_probes.py):
    probes[setname][target][layer_idx] = {"probe": Ridge, "scaler": StandardScaler, "pca": PCA}
"""

from __future__ import annotations

import os
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np


@dataclass
class ProbeSpec:
    key: str
    setname: str
    target: str
    layer: int
    direction: Any  # np.ndarray after discovery; torch.Tensor after engine init (real mode)
    display_name: str


def _extract_direction(entry: dict) -> np.ndarray:
    if "direction" in entry:
        direction = np.asarray(entry["direction"], dtype=np.float32)
    else:
        probe = entry["probe"]
        scaler = entry["scaler"]
        pca = entry["pca"]
        direction = (pca.components_.T @ probe.coef_) / scaler.scale_
    n = np.linalg.norm(direction)
    return direction / n if n > 0 else direction


def _prettify(target: str) -> str:
    return target.replace("_", " ").strip().title()


def discover_probes(
    probe_path: Path,
    display_names: Optional[dict[str, str]] = None,
    target_layer: Optional[int] = None,
) -> list[ProbeSpec]:
    """Walk the nested pickle and return one ProbeSpec per (setname, target).

    For each (setname, target) we keep the layer closest to ``target_layer``
    (defaults to the max layer present), since temporal info lives in late
    layers per the tam paper.
    """
    with open(probe_path, "rb") as f:
        probes = pickle.load(f)

    display_names = display_names or {}
    candidates: dict[tuple[str, str], tuple[int, dict]] = {}

    for setname, by_target in probes.items():
        if not isinstance(by_target, dict):
            continue
        for target, by_layer in by_target.items():
            if not isinstance(by_layer, dict):
                continue
            for layer, entry in by_layer.items():
                if not (isinstance(entry, dict) and ("probe" in entry or "direction" in entry)):
                    continue
                key = (setname, target)
                layer_i = int(layer)
                if key not in candidates:
                    candidates[key] = (layer_i, entry)
                else:
                    current_layer = candidates[key][0]
                    if target_layer is not None:
                        if abs(layer_i - target_layer) < abs(current_layer - target_layer):
                            candidates[key] = (layer_i, entry)
                    elif layer_i > current_layer:
                        candidates[key] = (layer_i, entry)

    specs: list[ProbeSpec] = []
    for (setname, target), (layer, entry) in candidates.items():
        try:
            direction = _extract_direction(entry)
        except Exception as e:
            print(f"[probes] skipping {setname}/{target}@{layer}: {e}")
            continue
        full_key = f"{setname}/{target}@{layer}"
        display = display_names.get(full_key) or display_names.get(target) or _prettify(target)
        specs.append(ProbeSpec(
            key=full_key,
            setname=setname,
            target=target,
            layer=layer,
            direction=direction.astype(np.float32),
            display_name=display,
        ))
    specs.sort(key=lambda p: (p.setname, p.target))
    return specs


class SteeringEngine:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        probe_path: Optional[str] = None,
        mock: bool = False,
        target_layer: Optional[int] = None,
        display_names: Optional[dict[str, str]] = None,
        device_map: Optional[str] = None,
    ):
        """Load model + probes.

        Places the model on CUDA at import time when CUDA is available (either
        a real GPU, or HF Spaces ZeroGPU's import-time emulation). Per HF's
        ZeroGPU docs, module-level CUDA placement is significantly more
        efficient than moving the model inside ``@spaces.GPU`` functions.
        """
        self.mock = mock
        self.model_name = model_name
        self._device: Optional[str] = None
        self._dtype = None

        if mock:
            self.tokenizer = None
            self.model = None
            self.n_layers = 36
            self.d_model = 2560
            rng = np.random.default_rng(0)
            mock_specs = [
                ("mock", "temporal_vocab", 29, "Temporal Vocab (mock)"),
                ("mock", "temporal_reasoning", 30, "Temporal Reasoning (mock)"),
            ]
            self.probes = []
            for setname, target, layer, display in mock_specs:
                v = rng.standard_normal(self.d_model).astype(np.float32)
                v /= np.linalg.norm(v)
                self.probes.append(ProbeSpec(
                    key=f"{setname}/{target}@{layer}",
                    setname=setname, target=target, layer=layer,
                    direction=v,
                    display_name=display,
                ))
            return

        import torch  # lazy: mock mode shouldn't require torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[engine] loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._dtype = dtype
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[engine] loading model ({dtype}) on {target_device}"
              + (f" (device_map={device_map})" if device_map else ""))
        if device_map is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
            )
            self._device = str(next(self.model.parameters()).device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, trust_remote_code=True,
            )
            if target_device == "cuda":
                self.model = self.model.to("cuda")
            self._device = target_device
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size

        if probe_path and Path(probe_path).exists():
            print(f"[engine] loading probes: {probe_path}")
            self.probes = discover_probes(
                Path(probe_path),
                display_names=display_names,
                target_layer=target_layer,
            )
            for p in self.probes:
                p.direction = torch.tensor(p.direction, dtype=dtype).to(self._device)
            print(f"[engine] found {len(self.probes)} probe(s): "
                  + ", ".join(p.key for p in self.probes))
        else:
            print(f"[engine] no probes at {probe_path}; running with 0 features")
            self.probes = []

    def _blocks(self):
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        raise RuntimeError("Could not locate decoder layer stack on model")

    def _register_hooks(self, alphas: dict[str, float]):
        # STEER_SPREAD="-2,0,2" → also hook layers L-2 and L+2 for each probe.
        # The slider α is total effective; we divide by len(spread) per-layer to
        # avoid compounding amplification when multiple adjacent layers fire.
        spread_env = os.environ.get("STEER_SPREAD", "0")
        try:
            spread = [int(x.strip()) for x in spread_env.split(",") if x.strip()]
        except Exception:
            spread = [0]
        if not spread:
            spread = [0]

        by_layer: dict[int, torch.Tensor] = {}
        n_spread = len(spread)
        for p in self.probes:
            a = float(alphas.get(p.key, 0.0) or 0.0)
            if a == 0.0:
                continue
            per_layer_alpha = a / n_spread
            for offset in spread:
                target = p.layer + offset
                if not (0 <= target < self.n_layers):
                    continue
                contrib = p.direction * per_layer_alpha
                by_layer[target] = (by_layer[target] + contrib) if target in by_layer else contrib.clone()

        handles = []
        blocks = self._blocks()
        for layer_idx, vec in by_layer.items():
            module = blocks[layer_idx]
            try:
                dev = next(module.parameters()).device
            except StopIteration:
                dev = next(self.model.parameters()).device
            v = vec.to(device=dev)

            def make_hook(v_local):
                def hook(_module, _inputs, outputs):
                    if isinstance(outputs, tuple):
                        hidden = outputs[0] + v_local.to(outputs[0].dtype)
                        return (hidden,) + outputs[1:]
                    return outputs + v_local.to(outputs.dtype)
                return hook

            handles.append(module.register_forward_hook(make_hook(v)))
        return handles

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int,
        alphas: dict[str, float],
        temperature: float = 0.8,
        top_p: float = 0.9,
        seed: Optional[int] = None,
    ) -> Iterator[str]:
        if seed is not None:
            import torch as _torch
            _torch.manual_seed(int(seed))
        if self.mock:
            import time
            nonzero = {k: v for k, v in alphas.items() if v}
            header = f"[mock α={nonzero}] " if nonzero else "[mock baseline] "
            filler = (
                "the chef carefully considered the ingredients on hand and began to "
                "plan the evening menu with an eye toward timing and prep order "
            ).split()
            acc = header
            for i in range(min(max_new_tokens, 120)):
                acc += filler[i % len(filler)] + " "
                yield acc
                time.sleep(0.03)
            return

        from transformers import TextIteratorStreamer

        handles = self._register_hooks(alphas)
        try:
            input_device = next(self.model.parameters()).device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(input_device)
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120.0,
            )
            kwargs = dict(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                do_sample=temperature > 0,
                streamer=streamer,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
            thread.start()

            acc = ""
            for chunk in streamer:
                acc += chunk
                yield acc
            thread.join()
        finally:
            for h in handles:
                h.remove()


    def read_probes(self, prompt: str) -> dict[str, float]:
        """Project the last-token activation at each probe's layer onto its direction.

        Returns {probe_key: scalar projection}. Positive = activation lies in the
        period direction; negative = opposite. Useful for showing what the model's
        current state "thinks" about each period, before any steering is applied.
        """
        if self.mock or not self.probes:
            return {p.key: 0.0 for p in self.probes}

        import torch as _torch
        layers_needed = sorted({p.layer for p in self.probes})
        captures: dict[int, _torch.Tensor] = {}
        blocks = self._blocks()

        def make_hook(L: int):
            def hook(_m, _i, outputs):
                x = outputs[0] if isinstance(outputs, tuple) else outputs
                captures[L] = x[:, -1, :].squeeze(0).detach()
            return hook

        handles = [blocks[L].register_forward_hook(make_hook(L)) for L in layers_needed]
        try:
            input_device = next(self.model.parameters()).device
            ids = self.tokenizer(prompt, return_tensors="pt").to(input_device)
            with _torch.no_grad():
                self.model(**ids)
        finally:
            for h in handles:
                h.remove()

        out: dict[str, float] = {}
        for p in self.probes:
            act = captures[p.layer].to(p.direction.dtype)
            out[p.key] = float(_torch.dot(act, p.direction).item())
        return out


def engine_from_env() -> SteeringEngine:
    """Build a SteeringEngine using env vars so Modal and local share config."""
    model_name = os.environ.get("STEER_MODEL", "Qwen/Qwen3.5-9B-Base")
    probe_path = os.environ.get(
        "STEER_PROBES", "./results/qwen3.5-9b-base/probes_caa_v3.pkl"
    )
    target_layer_env = os.environ.get("STEER_LAYER")
    target_layer = int(target_layer_env) if target_layer_env else None
    mock = os.environ.get("STEER_MOCK", "0") == "1"

    # Friendly display names; keys can be either "setname/target@layer" or just "target".
    display_names = {
        "log_time_horizon": "Temporal Vocab",
        "planning_depth": "Temporal Reasoning",
        "time_horizon": "Horizon (long↔short)",
        "tonight": "Tonight",
        "tomorrow": "Tomorrow",
        "one_week": "One Week",
        "one_month": "One Month",
        "one_year": "One Year",
        "a_decade": "A Decade",
    }

    return SteeringEngine(
        model_name=model_name,
        probe_path=probe_path,
        mock=mock,
        target_layer=target_layer,
        display_names=display_names,
    )
