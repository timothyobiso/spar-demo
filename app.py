"""Gradio UI for causal steering with tam probes — chat-style with sidebar.

Three steering modes side-by-side, switch via tabs in the sidebar:
  1. Periods       — six discrete CAA sliders (the original)
  2. Time → period — one Time slider blends across the six CAA probes
  3. Time → continuous — one Time slider rides a single Ridge-regression direction
"""

from __future__ import annotations

import os
import time

import gradio as gr

from steering import SteeringEngine, engine_from_env

try:
    import spaces
    _ON_SPACES = True
except ImportError:
    _ON_SPACES = False

    class _NoSpaces:
        def GPU(self, duration: int = 60):
            def decorator(f):
                return f
            return decorator

    spaces = _NoSpaces()


GPU_DURATION_SECONDS = int(os.environ.get("STEER_GPU_DURATION", "120"))
ALPHA_MAX = float(os.environ.get("STEER_ALPHA_MAX", "100"))
STRENGTH_MAX = float(os.environ.get("STEER_STRENGTH_MAX", str(ALPHA_MAX)))


engine = engine_from_env()
PERIOD_ORDER = ["tonight", "tomorrow", "one_week", "one_month", "one_year", "a_decade"]
PERIOD_LABELS = ["tonight", "tomorrow", "1 week", "1 month", "1 year", "a decade"]
N_PERIODS = 6


@spaces.GPU(duration=GPU_DURATION_SECONDS)
def _gen_alphas(prompt_text, max_new, temp, top_p_v, alphas, seed=None):
    for chunk in engine.generate_stream(prompt_text, int(max_new), alphas=alphas,
                                        temperature=float(temp), top_p=float(top_p_v),
                                        seed=seed):
        yield chunk


@spaces.GPU(duration=GPU_DURATION_SECONDS)
def _gen_vectors(prompt_text, max_new, temp, top_p_v, vectors, seed=None):
    for chunk in engine.generate_stream(prompt_text, int(max_new),
                                        steering_vectors=vectors,
                                        temperature=float(temp), top_p=float(top_p_v),
                                        seed=seed):
        yield chunk


def _alpha_summary(alphas: dict[str, float], display_map: dict[str, str]) -> str:
    parts = []
    for k, v in alphas.items():
        if v == 0:
            continue
        name = display_map.get(k, k)
        parts.append(f"{name} {v:+.0f}")
    return ", ".join(parts) or "no steering"


def _interp_summary(time_pos: float, strength: float) -> str:
    if strength == 0:
        return "no steering"
    lo = int(time_pos)
    hi = min(lo + 1, N_PERIODS - 1)
    frac = time_pos - lo
    if frac < 0.05:
        which = PERIOD_LABELS[lo]
    elif frac > 0.95:
        which = PERIOD_LABELS[hi]
    else:
        which = f"{PERIOD_LABELS[lo]}↔{PERIOD_LABELS[hi]}({frac:.2f})"
    return f"interp[{which}] α={strength:+.0f}"


def _cont_summary(time_pos: float, strength: float) -> str:
    if strength == 0:
        return "no steering"
    signed = (time_pos - 2.5) / 2.5
    if abs(signed) < 0.02:
        return "no steering (center)"
    direction = "longer" if signed > 0 else "shorter"
    return f"cont[{direction}, signed={signed:+.2f}] α={strength:+.0f}"


def build_ui(engine: SteeringEngine) -> gr.Blocks:
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "-apple-system", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    )

    custom_css = """
    .gradio-container { max-width: none !important; padding: 0 1rem !important; }
    #chatbox { height: calc(100vh - 220px) !important; min-height: 400px; }
    .message-bubble .prose { font-size: 0.95rem; }
    footer { display: none !important; }
    """

    with gr.Blocks(title="Temporal Steering", theme=theme, css=custom_css) as demo:
        # ---- shared state -------------------------------------------------
        active_mode = gr.State("periods")

        with gr.Sidebar(label="Steering & generation", open=False, width=360, position="left"):
            gr.Markdown("### Steering mode")

            sliders: dict[str, gr.Slider] = {}
            display_map: dict[str, str] = {}

            with gr.Tabs() as mode_tabs:
                # --- Mode 1: Periods (original 6 sliders) ------------------
                with gr.Tab("Periods", id="periods") as tab_periods:
                    gr.Markdown(
                        "<span style='font-size:0.85em; color:#888;'>"
                        "+α toward the period · −α away · 0 means no steering"
                        "</span>"
                    )
                    if not engine.probes:
                        gr.Markdown("_No probes discovered._")
                    else:
                        for p in engine.probes:
                            sliders[p.key] = gr.Slider(
                                minimum=-ALPHA_MAX, maximum=ALPHA_MAX, value=0.0, step=0.5,
                                label=p.display_name,
                            )
                            display_map[p.key] = p.display_name

                # --- Mode 2: Time interp on CAA probes ---------------------
                with gr.Tab("Time → period", id="interp") as tab_interp:
                    gr.Markdown(
                        "<span style='font-size:0.85em; color:#888;'>"
                        "Slides between the six CAA probes (piecewise linear blend). "
                        "0 = tonight · 5 = decade. Strength = magnitude of the blended push."
                        "</span>"
                    )
                    interp_time = gr.Slider(
                        minimum=0.0, maximum=N_PERIODS - 1, value=0.0, step=0.05,
                        label="Time (0=tonight … 5=decade)",
                    )
                    interp_strength = gr.Slider(
                        minimum=0.0, maximum=STRENGTH_MAX, value=0.0, step=0.5,
                        label="Strength (α)",
                    )

                # --- Mode 3: Time → continuous probe -----------------------
                with gr.Tab("Time → continuous", id="continuous") as tab_cont:
                    if engine.continuous_direction is None:
                        gr.Markdown(
                            "_Continuous probe not loaded. "
                            "Train via `experiments/train_continuous_v3.py` "
                            "and set `STEER_PROBES_CONTINUOUS`._"
                        )
                    else:
                        gr.Markdown(
                            "<span style='font-size:0.85em; color:#888;'>"
                            "One Ridge-regression direction (log10 days). "
                            "Center (2.5) = no push · &lt;2.5 = shorter · &gt;2.5 = longer."
                            "</span>"
                        )
                    cont_time = gr.Slider(
                        minimum=0.0, maximum=N_PERIODS - 1, value=2.5, step=0.05,
                        label="Time (0=tonight … 5=decade)",
                    )
                    cont_strength = gr.Slider(
                        minimum=0.0, maximum=STRENGTH_MAX, value=0.0, step=0.5,
                        label="Strength (α)",
                    )

            tab_periods.select(lambda: "periods", outputs=active_mode)
            tab_interp.select(lambda: "interp", outputs=active_mode)
            tab_cont.select(lambda: "continuous", outputs=active_mode)

            gr.Markdown("### Generation")
            max_tokens = gr.Slider(8, 512, value=80, step=1, label="Max new tokens")
            with gr.Accordion("Sampling", open=False):
                temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")

            with gr.Row():
                reset_btn = gr.Button("Reset sliders", size="sm")
                clear_btn = gr.Button("Clear chat", size="sm")

            gr.Markdown("### Probe readings on input")
            gr.Markdown(
                "<span style='font-size:0.85em; color:#888;'>"
                "Projects the last-token activation onto each probe direction."
                "</span>"
            )
            read_btn = gr.Button("Read probes from input", size="sm")
            probe_readings = gr.Markdown("_click Read after typing a prompt_")

        # Main area
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("### Temporal Probe Steering")
                if engine.mock:
                    gr.Markdown("_mock mode — no model loaded_")
                else:
                    host = "ZeroGPU" if _ON_SPACES else "local"
                    cont_layer = engine.continuous_layer if engine.continuous_direction is not None else "—"
                    gr.Markdown(
                        f"<span style='font-size:0.85em; color:#888;'>"
                        f"{engine.model_name} · {len(engine.probes)} CAA probes · "
                        f"continuous@L{cont_layer} · {host}"
                        "</span>",
                    )
            with gr.Column(scale=1, min_width=140):
                gr.HTML(
                    '<div style="text-align:right; padding-top:0.6em; font-size:0.85em;">'
                    '<a href="?__theme=light" style="text-decoration:none; margin-right:0.7em;">☀ Light</a>'
                    '<a href="?__theme=dark"  style="text-decoration:none;">☾ Dark</a>'
                    '</div>'
                )

        chat = gr.Chatbot(
            elem_id="chatbox",
            show_label=False,
            value=[],
        )

        with gr.Row():
            input_box = gr.Textbox(
                placeholder="Type a prompt — try 'The next thing on my schedule is'",
                show_label=False,
                lines=1,
                scale=8,
                autofocus=True,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)
            compare_btn = gr.Button("Compare", variant="secondary", scale=1, min_width=80)

        slider_keys = list(sliders.keys())
        slider_list = list(sliders.values())

        # ---- per-mode steering builders -----------------------------------
        def _build_mode_call(mode, alpha_vals, interp_t, interp_s, cont_t, cont_s):
            """Return (gen_fn, kwargs_dict, summary_text)."""
            if mode == "periods":
                alphas = dict(zip(slider_keys, alpha_vals))
                return _gen_alphas, {"alphas": alphas}, _alpha_summary(alphas, display_map)
            if mode == "interp":
                vec = engine.vec_interp(float(interp_t), float(interp_s))
                return _gen_vectors, {"vectors": vec}, _interp_summary(float(interp_t), float(interp_s))
            if mode == "continuous":
                vec = engine.vec_continuous(float(cont_t), float(cont_s))
                return _gen_vectors, {"vectors": vec}, _cont_summary(float(cont_t), float(cont_s))
            return _gen_alphas, {"alphas": {}}, "no steering"

        def respond(message, history, mode, max_new, temp, top_p_v,
                    interp_t, interp_s, cont_t, cont_s, *alpha_vals):
            if not message or not message.strip():
                yield "", history
                return
            history = list(history or [])
            gen_fn, kwargs, tag = _build_mode_call(mode, alpha_vals, interp_t, interp_s, cont_t, cont_s)
            mode_label = {"periods": "Periods", "interp": "Time→period", "continuous": "Time→cont."}[mode]
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": ""})
            yield "", history
            header = f"_{mode_label} · {tag}_\n\n" if tag != "no steering" else f"_{mode_label} · baseline_\n\n"
            for chunk in gen_fn(message, max_new, temp, top_p_v, **kwargs):
                history[-1]["content"] = header + (message + chunk)
                yield "", history

        def compare(message, history, mode, max_new, temp, top_p_v,
                    interp_t, interp_s, cont_t, cont_s, *alpha_vals):
            if not message or not message.strip():
                yield "", history
                return
            history = list(history or [])
            gen_fn, kwargs, tag = _build_mode_call(mode, alpha_vals, interp_t, interp_s, cont_t, cont_s)
            mode_label = {"periods": "Periods", "interp": "Time→period", "continuous": "Time→cont."}[mode]
            seed = int(time.time() * 1000) % (2**31)

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "**Baseline (α=0)**\n\n"})
            yield "", history
            # baseline call shares the prompt/seed with steered, just with empty steering
            if mode == "periods":
                baseline_kwargs = {"alphas": {}}
            else:
                baseline_kwargs = {"vectors": {}}
            baseline_fn = _gen_alphas if mode == "periods" else _gen_vectors
            for chunk in baseline_fn(message, max_new, temp, top_p_v, seed=seed, **baseline_kwargs):
                history[-1]["content"] = "**Baseline (α=0)**\n\n" + (message + chunk)
                yield "", history

            history.append({"role": "assistant", "content": f"**Steered — {mode_label}: {tag}**\n\n"})
            yield "", history
            for chunk in gen_fn(message, max_new, temp, top_p_v, seed=seed, **kwargs):
                history[-1]["content"] = f"**Steered — {mode_label}: {tag}**\n\n" + (message + chunk)
                yield "", history

        common_inputs = [input_box, chat, active_mode, max_tokens, temperature, top_p,
                         interp_time, interp_strength, cont_time, cont_strength,
                         *slider_list]

        send_btn.click(respond, inputs=common_inputs, outputs=[input_box, chat])
        input_box.submit(respond, inputs=common_inputs, outputs=[input_box, chat])
        compare_btn.click(compare, inputs=common_inputs, outputs=[input_box, chat])
        clear_btn.click(lambda: [], outputs=chat)

        def reset_all():
            return [0.0] * len(slider_list) + [0.0, 0.0, 2.5, 0.0]

        reset_btn.click(reset_all,
                        outputs=[*slider_list, interp_time, interp_strength, cont_time, cont_strength])

        @spaces.GPU(duration=GPU_DURATION_SECONDS)
        def _read_probes_gpu(prompt_text):
            return engine.read_probes(prompt_text)

        def read_fn(prompt_text):
            if not prompt_text or not prompt_text.strip():
                return "_empty prompt — type something first_"
            readings = _read_probes_gpu(prompt_text)
            if not readings:
                return "_no probes loaded_"
            lines = []
            ranked = sorted(readings.items(), key=lambda kv: -kv[1])
            for k, v in ranked:
                name = display_map.get(k, k)
                bar = "▰" * min(int(abs(v) * 2), 20) or "·"
                arrow = "→" if v >= 0 else "←"
                lines.append(f"`{arrow} {v:+6.2f}`  {bar}  **{name}**")
            return "\n\n".join(lines)

        read_btn.click(read_fn, inputs=input_box, outputs=probe_readings)

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("STEER_PORT", "7860"))
    share = os.environ.get("STEER_SHARE", "0") == "1"
    build_ui(engine).queue().launch(server_name="0.0.0.0", server_port=port, share=share)
