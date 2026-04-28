"""Gradio UI for causal steering with tam probes — chat-style with sidebar."""

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


engine = engine_from_env()


@spaces.GPU(duration=GPU_DURATION_SECONDS)
def _steered_generate(prompt_text: str, max_new: int, temp: float, top_p_v: float,
                      alphas: dict[str, float], seed: int | None = None):
    for chunk in engine.generate_stream(prompt_text, int(max_new), alphas,
                                        float(temp), float(top_p_v), seed=seed):
        yield chunk


def _alpha_summary(alphas: dict[str, float], display_map: dict[str, str]) -> str:
    parts = []
    for k, v in alphas.items():
        if v == 0:
            continue
        name = display_map.get(k, k)
        parts.append(f"{name} {v:+.0f}")
    return ", ".join(parts) or "no steering"


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
        with gr.Sidebar(label="Steering & generation", open=False, width=340, position="left"):
            gr.Markdown("### Temporal sliders")
            gr.Markdown(
                "<span style='font-size:0.85em; color:#888;'>"
                "+α toward the period · −α away · 0 means no steering"
                "</span>"
            )
            sliders: dict[str, gr.Slider] = {}
            display_map: dict[str, str] = {}
            if not engine.probes:
                gr.Markdown("_No probes discovered._")
            else:
                for p in engine.probes:
                    sliders[p.key] = gr.Slider(
                        minimum=-ALPHA_MAX, maximum=ALPHA_MAX, value=0.0, step=0.5,
                        label=p.display_name,
                    )
                    display_map[p.key] = p.display_name

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
                "Projects the last-token activation onto each probe direction. "
                "Shows what the model is 'currently thinking' along each axis, "
                "before any steering."
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
                    gr.Markdown(
                        f"<span style='font-size:0.85em; color:#888;'>"
                        f"{engine.model_name} · {len(engine.probes)} probes · {host}"
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
                placeholder="Type a prompt — try 'The next thing on my schedule is' with Tonight=+70",
                show_label=False,
                lines=1,
                scale=8,
                autofocus=True,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)
            compare_btn = gr.Button("Compare", variant="secondary", scale=1, min_width=80)

        slider_keys = list(sliders.keys())
        slider_list = list(sliders.values())

        def respond(message, history, max_new, temp, top_p_v, *alpha_vals):
            if not message or not message.strip():
                yield "", history
                return
            history = list(history or [])
            alphas = dict(zip(slider_keys, alpha_vals))
            tag = _alpha_summary(alphas, display_map)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": ""})
            yield "", history
            header = f"_{tag}_\n\n" if tag != "no steering" else ""
            for chunk in _steered_generate(message, max_new, temp, top_p_v, alphas):
                history[-1]["content"] = header + (message + chunk)
                yield "", history

        def compare(message, history, max_new, temp, top_p_v, *alpha_vals):
            if not message or not message.strip():
                yield "", history
                return
            history = list(history or [])
            alphas = dict(zip(slider_keys, alpha_vals))
            tag = _alpha_summary(alphas, display_map)
            zero_alphas = {k: 0.0 for k in alphas}
            seed = int(time.time() * 1000) % (2**31)

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "**Baseline (α=0)**\n\n"})
            yield "", history
            for chunk in _steered_generate(message, max_new, temp, top_p_v, zero_alphas, seed=seed):
                history[-1]["content"] = "**Baseline (α=0)**\n\n" + (message + chunk)
                yield "", history

            history.append({"role": "assistant", "content": f"**Steered — {tag}**\n\n"})
            yield "", history
            for chunk in _steered_generate(message, max_new, temp, top_p_v, alphas, seed=seed):
                history[-1]["content"] = f"**Steered — {tag}**\n\n" + (message + chunk)
                yield "", history

        send_btn.click(
            respond,
            inputs=[input_box, chat, max_tokens, temperature, top_p, *slider_list],
            outputs=[input_box, chat],
        )
        input_box.submit(
            respond,
            inputs=[input_box, chat, max_tokens, temperature, top_p, *slider_list],
            outputs=[input_box, chat],
        )
        compare_btn.click(
            compare,
            inputs=[input_box, chat, max_tokens, temperature, top_p, *slider_list],
            outputs=[input_box, chat],
        )
        clear_btn.click(lambda: [], outputs=chat)
        reset_btn.click(lambda: [0.0] * len(slider_list), outputs=slider_list)

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
