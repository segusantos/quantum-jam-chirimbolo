"""Utility functions for the BB84 interactive notebook."""

from typing import Dict, List, Any, Callable

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
import ipywidgets as widgets

from .bb84_protocol import BB84Parameters, BB84Protocol
from .noise import NoiseChannel


NOISE_PRESETS = {
    "Sin ruido": {"name": "none", "param": None, "label": "", "max": 0.0},
    "Depolarizante": {"name": "depolarizing", "param": "p", "label": "p", "max": 0.3},
    "Bit flip": {"name": "bit_flip", "param": "p", "label": "p", "max": 0.3},
    "Phase flip": {"name": "phase_flip", "param": "p", "label": "p", "max": 0.3},
    "Amortiguamiento de amplitud": {"name": "amplitude_damping", "param": "gamma", "label": "gamma", "max": 0.3},
    "Damping de fase": {"name": "phase_damping", "param": "lambda", "label": "lambda", "max": 0.3},
}


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a value between lower and upper bounds."""
    return max(lower, min(upper, value))


def normalize_seed(raw: Any) -> int | None:
    """Convert raw input to a valid seed integer or None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def build_noise_channel(option: str, value: float, readout0: float, readout1: float) -> NoiseChannel:
    """Build a noise channel from UI parameters."""
    preset = NOISE_PRESETS[option]
    params = {}
    if preset["param"]:
        params[preset["param"]] = clamp(value)
    if readout0 > 0:
        params["p0to1"] = clamp(readout0)
    if readout1 > 0:
        params["p1to0"] = clamp(readout1)
    return NoiseChannel(preset["name"], params)


def run_bb84(
    num_bits: int,
    seed_value: Any,
    eve: bool,
    eve_prob: float,
    noise_option: str,
    noise_value: float,
    readout0: float,
    readout1: float,
):
    """Execute a BB84 protocol run with the given parameters."""
    seed = normalize_seed(seed_value)
    params = BB84Parameters(
        num_bits=num_bits,
        seed=seed,
        eve_present=eve,
        eve_intercept_prob=clamp(eve_prob),
    )
    params.noise = build_noise_channel(noise_option, noise_value, readout0, readout1)
    protocol = BB84Protocol(params)
    return protocol.run()


def style_result_table(result):
    """Apply color styling to the result dataframe."""
    df = result.to_dataframe()

    def highlight(row):
        base = "#ffffff"
        if row["Sifted"] == "Si":
            base = "#edf7ed"
        if row["Coincide?"] == "❌":
            if row["Causa"] == "Eve":
                base = "#fdecea"
            elif row["Causa"] == "Ruido":
                base = "#fff4e5"
            else:
                base = "#ede7f6"
        return [f"background-color: {base}"] * len(row)

    styled = df.style.apply(highlight, axis=1).set_properties(**{"text-align": "center"})
    try:
        styled = styled.hide(axis="index")
    except AttributeError:
        styled = styled.hide_index()
    return styled


def lerp_color(color_a: tuple, color_b: tuple, t: float) -> tuple:
    """Linear interpolation between two RGB colors."""
    return tuple(int((1 - t) * a + t * b) for a, b in zip(color_a, color_b))


def color_to_hex(color: tuple) -> str:
    """Convert RGB tuple to hex color string."""
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def detection_badge(prob: float, label: str) -> str:
    """Generate an HTML badge with color-coded detection probability."""
    value = clamp(prob)
    color = lerp_color((235, 87, 87), (67, 160, 71), value)
    return (
        f"<span style='display:inline-block;margin-right:8px;padding:6px 12px;border-radius:8px;"
        f"background:{color_to_hex(color)};color:#102a43;font-weight:600;'>{label}: {value:.6f}</span>"
    )


def key_summary(result) -> HTML:
    """Generate a summary of BB84 run results."""
    lines = [
        "RESULTADO BB84",
        "------------------------------------------",
        f"Bases iguales : {result.equal_bases()}",
        f"QBER          : {result.qber:.4f}",
        f"Clave Alice   : {result.sifted_alice_bits or '-'}",
        f"Clave Bob     : {result.sifted_bob_bits or '-'}",
    ]
    return HTML("<pre>" + "\n".join(lines) + "</pre>")


def detection_summary(result, detection_bits: int) -> HTML:
    """Generate detection probability summary with visual badges."""
    sample = min(detection_bits, result.sifted_key_length())
    if sample == 0:
        return HTML("<p><strong>No hay suficientes bits cribados para el muestreo.</strong></p>")
    empirical = result.detection_probability(sample)
    ideal = 1.0 - (0.75**sample)
    mismatches = sum(1 for i in range(sample) if result.sifted_alice_bits[i] != result.sifted_bob_bits[i])
    badges = detection_badge(empirical, "P(det) empirica")
    if result.params.eve_present:
        badges += detection_badge(ideal, "P(det) ideal")
    body = (
        "<div style='margin-top:6px;'>"
        + badges
        + f"<div style='margin-top:4px;'>Bits muestreados: {sample} | discrepancias: {mismatches}</div>"
        + "</div>"
    )
    return HTML(body)


def split_after_detection(result, detection_bits: int) -> tuple[str, str, str, str]:
    """Split keys into detection sample and remaining bits."""
    sample = min(detection_bits, result.sifted_key_length())
    return (
        result.sifted_alice_bits[sample:],
        result.sifted_bob_bits[sample:],
        result.sifted_alice_bits[:sample],
        result.sifted_bob_bits[:sample],
    )


def sweep_noise(
    num_bits: int,
    seed_value: Any,
    eve: bool,
    eve_prob: float,
    noise_option: str,
    values: List[float],
    readout0: float,
    readout1: float,
    detection_bits: int,
) -> List[Dict[str, float]]:
    """Sweep noise parameter and collect QBER and detection probability."""
    data = []
    for value in values:
        result = run_bb84(num_bits, seed_value, eve, eve_prob, noise_option, value, readout0, readout1)
        sample = min(detection_bits, result.sifted_key_length())
        data.append({"value": value, "qber": result.qber, "p_detect": result.detection_probability(sample)})
    return data


def render_noise_curves(data: List[Dict[str, float]], noise_label: str):
    """Render dual-axis plot of QBER and detection probability vs noise parameter."""
    if not data:
        return None
    values = [item["value"] for item in data]
    qber = [item["qber"] for item in data]
    p_detect = [item["p_detect"] for item in data]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(values, qber, marker="o", color="#1f77b4", label="QBER")
    ax.set_xlabel(f"Parametro de ruido ({noise_label})")
    ax.set_ylabel("QBER")
    ax.grid(alpha=0.25)
    twin = ax.twinx()
    twin.plot(values, p_detect, marker="s", color="#2e7d32", label="P(det)")
    twin.set_ylabel("P(det)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    plt.tight_layout()
    return fig


def format_key_preview(key: str, limit: int = 64) -> str:
    """Format a key string with ellipsis if too long."""
    if not key:
        return "-"
    if len(key) <= limit:
        return key
    head = max(limit // 2, 1)
    tail = max(limit - head - 3, 0)
    if tail <= 0:
        return key[:limit]
    return key[:head] + "..." + key[-tail:]


def configure_noise_slider(dropdown: widgets.Dropdown, slider: widgets.FloatSlider) -> None:
    """Configure noise slider based on selected noise type."""

    def update(_=None):
        preset = NOISE_PRESETS[dropdown.value]
        slider.description = preset["label"] or "valor"
        slider.max = preset["max"]
        slider.step = max(preset["max"] / 50.0, 0.01) if preset["max"] > 0 else 0.01
        slider.disabled = preset["param"] is None
        if slider.max == 0:
            slider.value = 0.0
        elif slider.value > slider.max:
            slider.value = slider.max

    dropdown.observe(update, names="value")
    update()


def bind_controls(
    controls: Dict[str, widgets.Widget], callback: Callable, output: widgets.Output
) -> Callable:
    """Bind widget controls to a callback function with automatic output clearing."""

    def handler(_=None):
        kwargs = {name: control.value for name, control in controls.items()}
        with output:
            output.clear_output(wait=True)
            callback(**kwargs)

    for control in controls.values():
        control.observe(handler, names="value")
    handler()
    return handler


# ============================================================================
# Part-specific setup functions for cleaner notebook presentation
# ============================================================================


def setup_part1():
    """Setup widgets and rendering for Part 1: Basic BB84."""
    from IPython.display import display

    slider = widgets.IntSlider(value=12, min=4, max=64, step=1, description="Bits (N)")
    output = widgets.Output()

    def render(num_bits):
        result = run_bb84(num_bits, "", False, 0.0, "Sin ruido", 0.0, 0.0, 0.0)
        display(key_summary(result))
        display(style_result_table(result))

    bind_controls({"num_bits": slider}, render, output)
    display(widgets.VBox([slider, output]))


def setup_part2():
    """Setup widgets and rendering for Part 2: Eve attack and detection."""
    from IPython.display import display

    num_bits = widgets.IntSlider(value=24, min=8, max=128, step=2, description="Bits (N)")
    seed = widgets.Text(value="", description="Semilla", placeholder="vacio = aleatorio")
    eve = widgets.Checkbox(value=True, description="Eve presente")
    eve_prob = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05, description="P(intercept)")
    detection = widgets.IntSlider(value=12, min=2, max=64, step=1, description="m (deteccion)")
    output = widgets.Output()

    def toggle_eve(change):
        eve_prob.disabled = not change["new"]

    eve.observe(toggle_eve, names="value")
    eve_prob.disabled = not eve.value

    def render(num_bits, seed_value, eve, eve_prob, detection_bits):
        result = run_bb84(num_bits, seed_value, eve, eve_prob, "Sin ruido", 0.0, 0.0, 0.0)
        display(key_summary(result))
        display(style_result_table(result))
        display(detection_summary(result, detection_bits))

    controls = {
        "num_bits": num_bits,
        "seed_value": seed,
        "eve": eve,
        "eve_prob": eve_prob,
        "detection_bits": detection,
    }

    bind_controls(controls, render, output)
    display(
        widgets.VBox(
            [
                widgets.HBox([num_bits, detection]),
                widgets.HBox([seed, eve, eve_prob]),
                output,
            ]
        )
    )


def setup_part3():
    """Setup widgets and rendering for Part 3: Noise exploration."""
    from IPython.display import display

    num_bits = widgets.IntSlider(value=48, min=16, max=256, step=4, description="Bits (N)")
    seed = widgets.Text(value="", description="Semilla", placeholder="vacio = aleatorio")
    eve = widgets.Checkbox(value=True, description="Eve presente")
    eve_prob = widgets.FloatSlider(value=0.3, min=0.0, max=1.0, step=0.05, description="P(intercept)")
    detection = widgets.IntSlider(value=16, min=4, max=96, step=2, description="m (deteccion)")
    noise = widgets.Dropdown(options=list(NOISE_PRESETS.keys()), value="Depolarizante", description="Ruido")
    noise_value = widgets.FloatSlider(value=0.1, min=0.0, max=0.3, step=0.01, description="p")
    readout0 = widgets.FloatSlider(value=0.0, min=0.0, max=0.2, step=0.01, description="p0->1")
    readout1 = widgets.FloatSlider(value=0.0, min=0.0, max=0.2, step=0.01, description="p1->0")
    points = widgets.IntSlider(value=7, min=3, max=15, step=1, description="barrido")
    output = widgets.Output()

    configure_noise_slider(noise, noise_value)

    def toggle_eve(change):
        eve_prob.disabled = not change["new"]

    eve.observe(toggle_eve, names="value")
    eve_prob.disabled = not eve.value

    def render(num_bits, seed_value, eve, eve_prob, detection_bits, noise_option, noise_value, readout0, readout1, sweep_points):
        result = run_bb84(num_bits, seed_value, eve, eve_prob, noise_option, noise_value, readout0, readout1)
        display(key_summary(result))
        sample = min(detection_bits, result.sifted_key_length())
        info = (
            f"<p>Bits cribados: <strong>{result.sifted_key_length()}</strong> | "
            f"QBER: <strong>{result.qber:.4f}</strong> | Muestreo m: {sample}</p>"
        )
        display(HTML(info))
        display(style_result_table(result))
        display(detection_summary(result, detection_bits))

        preset = NOISE_PRESETS[noise_option]
        data = []
        if preset["param"] and sweep_points > 1:
            upper = preset["max"] if preset["max"] > 0 else max(noise_value, 0.1)
            upper = max(upper, noise_value if noise_value > 0 else 0.1)
            values = np.linspace(0.0, upper, sweep_points)
            values = np.unique(np.append(values, noise_value))
            data = sweep_noise(num_bits, seed_value, eve, eve_prob, noise_option, values, readout0, readout1, detection_bits)
        fig = render_noise_curves(data, preset["label"] or "valor")
        if fig is not None:
            display(fig)
            plt.close(fig)

    controls = {
        "num_bits": num_bits,
        "seed_value": seed,
        "eve": eve,
        "eve_prob": eve_prob,
        "detection_bits": detection,
        "noise_option": noise,
        "noise_value": noise_value,
        "readout0": readout0,
        "readout1": readout1,
        "sweep_points": points,
    }

    bind_controls(controls, render, output)
    display(
        widgets.VBox(
            [
                widgets.HBox([num_bits, detection, points]),
                widgets.HBox([seed, eve, eve_prob]),
                widgets.HBox([noise, noise_value]),
                widgets.HBox([readout0, readout1]),
                output,
            ]
        )
    )


def setup_part4():
    """Setup widgets and rendering for Part 4: Error correction and privacy amplification."""
    from IPython.display import display
    from .error_correction import CascadeErrorCorrector
    from .privacy import PrivacyAmplifier

    num_bits = widgets.IntSlider(value=64, min=16, max=256, step=4, description="Bits (N)")
    seed = widgets.Text(value="", description="Semilla", placeholder="vacio = aleatorio")
    eve = widgets.Checkbox(value=True, description="Eve presente")
    eve_prob = widgets.FloatSlider(value=0.4, min=0.0, max=1.0, step=0.05, description="P(intercept)")
    detection = widgets.IntSlider(value=20, min=4, max=128, step=2, description="m (deteccion)")
    noise = widgets.Dropdown(options=list(NOISE_PRESETS.keys()), value="Depolarizante", description="Ruido")
    noise_value = widgets.FloatSlider(value=0.05, min=0.0, max=0.3, step=0.01, description="p")
    readout0 = widgets.FloatSlider(value=0.0, min=0.0, max=0.2, step=0.01, description="p0->1")
    readout1 = widgets.FloatSlider(value=0.0, min=0.0, max=0.2, step=0.01, description="p1->0")
    block = widgets.IntSlider(value=4, min=2, max=16, step=1, description="bloque")
    rounds = widgets.IntSlider(value=3, min=1, max=5, step=1, description="rondas")
    security = widgets.IntSlider(value=40, min=10, max=80, step=5, description="seguridad")
    target = widgets.IntSlider(value=0, min=0, max=128, step=4, description="target (0 auto)")
    output = widgets.Output()

    configure_noise_slider(noise, noise_value)

    def toggle_eve(change):
        eve_prob.disabled = not change["new"]

    eve.observe(toggle_eve, names="value")
    eve_prob.disabled = not eve.value

    def render(num_bits, seed_value, eve, eve_prob, detection_bits, noise_option, noise_value, readout0, readout1, bloque, rondas, seguridad, target_bits):
        result = run_bb84(num_bits, seed_value, eve, eve_prob, noise_option, noise_value, readout0, readout1)
        display(key_summary(result))
        display(style_result_table(result))
        display(detection_summary(result, detection_bits))

        alice_rest, bob_rest, alice_sample, bob_sample = split_after_detection(result, detection_bits)
        sample = len(alice_sample)
        sample_mismatches = sum(1 for a, b in zip(alice_sample, bob_sample) if a != b)
        detection_lines = [
            "MUESTREO PARA DETECCION",
            "------------------------------------------",
            f"m                 : {sample}",
            f"Discrepancias     : {sample_mismatches}",
            f"Alice (m)         : {format_key_preview(alice_sample, 48)}",
            f"Bob   (m)         : {format_key_preview(bob_sample, 48)}",
        ]
        display(HTML("<pre>" + "\n".join(detection_lines) + "</pre>"))

        if not alice_rest:
            display(HTML("<p><strong>No quedan bits tras el muestreo. Disminuye m o incrementa N.</strong></p>"))
            return

        corrector = CascadeErrorCorrector(block_size=bloque, rounds=rondas)
        cascade = corrector.correct(alice_rest, bob_rest)
        cascade_lines = [
            "CASCADE (correccion)",
            "------------------------------------------",
            f"Longitud entrada  : {len(alice_rest)}",
            f"Bloque inicial    : {bloque}",
            f"Rondas            : {rondas}",
            f"Correcciones      : {len(cascade.corrections)}",
            f"Bits fugados      : {cascade.leakage_bits}",
            f"Errores residuales: {cascade.residual_errors}",
            f"Clave corregida   : {format_key_preview(cascade.corrected_key, 64)}",
        ]
        display(HTML("<pre>" + "\n".join(cascade_lines) + "</pre>"))

        amplifier = PrivacyAmplifier(security_parameter=seguridad)
        target_length = None if target_bits <= 0 else target_bits
        privacy = amplifier.apply(cascade.corrected_key, cascade.leakage_bits, target_length=target_length)
        privacy_lines = [
            "PRIVACY AMPLIFICATION",
            "------------------------------------------",
            f"Hash              : {privacy.hash_function}",
            f"Seguridad (bits)  : {seguridad}",
            f"Target solicitado : {target_length if target_length is not None else 'auto'}",
            f"Longitud final    : {privacy.target_length}",
            f"Bits descartados  : {privacy.discarded_bits}",
            f"Clave final       : {format_key_preview(privacy.final_key, 64)}",
        ]
        display(HTML("<pre>" + "\n".join(privacy_lines) + "</pre>"))

        if cascade.residual_errors > 0:
            display(HTML("<p style='color:#c62828;font-weight:600;'>Persisten errores tras Cascade; aumenta rondas o m.</p>"))

    controls = {
        "num_bits": num_bits,
        "seed_value": seed,
        "eve": eve,
        "eve_prob": eve_prob,
        "detection_bits": detection,
        "noise_option": noise,
        "noise_value": noise_value,
        "readout0": readout0,
        "readout1": readout1,
        "bloque": block,
        "rondas": rounds,
        "seguridad": security,
        "target_bits": target,
    }

    bind_controls(controls, render, output)
    display(
        widgets.VBox(
            [
                widgets.HBox([num_bits, detection]),
                widgets.HBox([seed, eve, eve_prob]),
                widgets.HBox([noise, noise_value]),
                widgets.HBox([readout0, readout1]),
                widgets.HBox([block, rounds, security, target]),
                output,
            ]
        )
    )
