"""Feature ELM on Hugging Face ZeroGPU.

The C++/CUDA library is loaded through its C API (lib/libfeature_elm_capi.so, built by
docker/Dockerfile.capi). ZeroGPU attaches a GPU only while a @spaces.GPU function runs, and CUDA must
not be initialised in the main process, so:

* drawing/classification and CPU training run in the main process (CPU only);
* GPU training and the GPU sweep run inside @spaces.GPU functions.

Outside ZeroGPU (locally, or on regular GPU hardware) the decorator is a no-op.
"""

import spaces  # must be imported before anything that could initialise CUDA

import ctypes
import importlib.util
import json
import os

import gradio as gr
import numpy as np
import pandas as pd

import digitprep

HERE = os.path.dirname(os.path.abspath(__file__))
BUFFER_BYTES = 1 << 16


# --- native library ---------------------------------------------------------------------------
def _preload_cuda_libraries() -> None:
    """Load cudart/cuBLAS/cuSOLVER from NVIDIA's pip wheels with RTLD_GLOBAL.

    libfeature_elm_capi.so is linked against libcudart.so.12, libcublas.so.12 and libcusolver.so.11;
    once these are loaded globally the dynamic linker resolves them by soname. Loading a library
    does not initialise CUDA, so this is safe in ZeroGPU's main process.
    """
    order = [
        ("nvidia.cuda_runtime", "libcudart.so.12"),
        ("nvidia.nvjitlink", "libnvJitLink.so.12"),
        ("nvidia.cublas", "libcublasLt.so.12"),
        ("nvidia.cublas", "libcublas.so.12"),
        ("nvidia.cusparse", "libcusparse.so.12"),
        ("nvidia.cusolver", "libcusolver.so.11"),
    ]
    for package, name in order:
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            continue
        path = os.path.join(list(spec.submodule_search_locations)[0], "lib", name)
        if os.path.exists(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


_preload_cuda_libraries()
_lib = ctypes.CDLL(os.path.join(HERE, "lib", "libfeature_elm_capi.so"))
for _fn in (_lib.felm_init, _lib.felm_evaluate, _lib.felm_classify):
    _fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    _fn.restype = ctypes.c_int
_lib.felm_health.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
_lib.felm_benchmark.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
_lib.felm_load_classifier.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]


def _call(fn, *args) -> dict:
    buffer = ctypes.create_string_buffer(BUFFER_BYTES)
    fn(*args, buffer, BUFFER_BYTES)
    return json.loads(buffer.value.decode())


_init = _call(_lib.felm_init, os.path.join(HERE, "data", "digits_8x8.csv").encode())
if _init.get("status") != "ok":
    raise RuntimeError(f"felm_init failed: {_init}")

# Hand-drawn digits use the MNIST+UCI model built by scripts/build_handwriting_data.py and
# felm-train (~97% on held-out handwriting vs ~47% for the start-up UCI-only model).
_model = _call(_lib.felm_load_classifier, os.path.join(HERE, "data", "handwriting_8x8.felm").encode())
if _model.get("status") != "ok":
    raise RuntimeError(f"felm_load_classifier failed: {_model}")


# --- GPU entry points (a GPU is attached only while these run) -----------------------------------
@spaces.GPU(duration=30)
def _evaluate_gpu(payload: str) -> dict:
    return _call(_lib.felm_evaluate, payload.encode())


@spaces.GPU(duration=60)
def _sweep_gpu() -> dict:
    return _call(_lib.felm_benchmark, 1)


@spaces.GPU(duration=15)
def _health_gpu() -> dict:
    return _call(_lib.felm_health)


# --- drawing -----------------------------------------------------------------------------------
def classify(editor_value):
    """Sketchpad drawing -> prediction, on the CPU in the main process (no GPU needed)."""
    if editor_value is None:
        return None, None
    image = editor_value.get("composite") if isinstance(editor_value, dict) else editor_value
    if image is None:
        return None, None
    feats = digitprep.features(digitprep.ink_from_rgba(np.asarray(image)))
    if feats is None:
        return None, None
    result = _call(_lib.felm_classify, json.dumps({"pixels": feats.tolist()}).encode())
    if result.get("status") != "ok":
        return None, None
    scores = np.array(result["scores"], dtype=np.float64)
    confidences = np.exp(6.0 * scores) / np.exp(6.0 * scores).sum()
    return {str(i): float(c) for i, c in enumerate(confidences)}, digitprep.preview(feats)


# --- training ----------------------------------------------------------------------------------
def _confusion_html(matrix) -> str:
    peak = max(max(row) for row in matrix) or 1
    cells = []
    for r, row in enumerate(matrix):
        tds = "".join(
            f'<td title="true {r}, predicted {c}: {v}" style="background:rgba(57,135,229,{0.08 + 0.92 * v / peak:.2f});'
            f'color:{"#fff" if v / peak > 0.5 else "inherit"}">{v or ""}</td>'
            for c, v in enumerate(row)
        )
        cells.append(f"<tr><th>{r}</th>{tds}</tr>")
    head = "".join(f"<th>{c}</th>" for c in range(10))
    return (
        '<table class="cm"><tr><th></th>' + head + "</tr>" + "".join(cells) + "</table>"
        "<p class='cm-note'>rows: true digit, columns: predicted</p>"
    )


def train(model, hidden, activation, precision, backend, history):
    payload = json.dumps(
        {"model": model, "hidden": int(hidden), "activation": activation, "precision": precision, "backend": backend}
    )
    result = _evaluate_gpu(payload) if backend == "gpu" else _call(_lib.felm_evaluate, payload.encode())
    if result.get("status") != "ok":
        raise gr.Error(result.get("message", "training failed"))
    summary = (
        f"### {result['test_accuracy'] * 100:.1f}% test accuracy\n"
        f"Trained on **{result['train_samples']}** digits in **{result['train_ms']:.1f} ms** on the "
        f"**{result['backend'].upper()}** ({result['precision']}); predicting {result['test_samples']} "
        f"held-out digits took {result['predict_ms']:.1f} ms."
    )
    row = {
        "model": result["model"],
        "hidden": result["hidden"],
        "backend": result["backend"],
        "precision": result["precision"],
        "accuracy %": round(result["test_accuracy"] * 100, 1),
        "train ms": round(result["train_ms"], 1),
    }
    history = pd.concat([pd.DataFrame([row]), history], ignore_index=True).head(10)
    return summary, _confusion_html(result["confusion"]), history


def sweep():
    result = _sweep_gpu()
    rows = []
    for r in result["rows"]:
        rows.append({"hidden": r["hidden"], "backend": "CPU", "train ms": round(r["cpu_train_ms"], 2)})
        if r.get("gpu_train_ms") is not None:
            rows.append({"hidden": r["hidden"], "backend": "GPU", "train ms": round(r["gpu_train_ms"], 2)})
    frame = pd.DataFrame(rows)
    device = result.get("device") or "no GPU attached"
    return frame, frame.pivot(index="hidden", columns="backend", values="train ms").reset_index(), f"Device: **{device}**"


def health():
    info = _health_gpu()
    if info.get("gpu_available"):
        return f"GPU attached: **{info['device']}** · library v{info['version']}"
    return "No GPU visible to the library (running on CPU)."


# --- UI ----------------------------------------------------------------------------------------
CSS = """
.cm { border-collapse: collapse; font-variant-numeric: tabular-nums; }
.cm th { color: var(--body-text-color-subdued); font-weight: 500; padding: 2px 6px; }
.cm td { width: 30px; height: 30px; text-align: center; border-radius: 4px; }
.cm-note { color: var(--body-text-color-subdued); font-size: 0.85em; }
"""

with gr.Blocks(title="Feature ELM · ZeroGPU") as demo:
    gr.Markdown(
        "# Feature ELM on ZeroGPU\n"
        "Extreme Learning Machines in C++20 and CUDA, trained in one least-squares solve. "
        "[Source](https://github.com/e-choness/feature_extraction_cuda_elm) · "
        "[Docs](https://e-choness.github.io/feature_extraction_cuda_elm/)"
    )
    with gr.Tab("Draw a digit"):
        with gr.Row():
            pad = gr.Sketchpad(
                label="Draw 0-9",
                canvas_size=(280, 280),
                type="numpy",
                brush=gr.Brush(default_size=18, colors=["#000000"], color_mode="fixed"),
                layers=False,
                transforms=(),
                placeholder="Pick the brush on the left, then draw one digit",
            )
            with gr.Column():
                label = gr.Label(label="Prediction", num_top_classes=3)
                preview = gr.Image(label="What the model sees (8×8)", height=170, width=170, interactive=False)
        pad.change(classify, pad, [label, preview], show_progress="hidden")
    with gr.Tab("Train & evaluate"):
        with gr.Row():
            model = gr.Dropdown(["elm", "os-elm", "ml-elm"], value="elm", label="Model")
            hidden = gr.Slider(16, 2048, value=512, step=16, label="Hidden nodes")
            activation = gr.Dropdown(["sigmoid", "tanh", "relu"], value="sigmoid", label="Activation")
            precision = gr.Dropdown(["float32", "float64"], value="float32", label="Precision")
            backend = gr.Radio(["gpu", "cpu"], value="gpu", label="Backend")
        run = gr.Button("Train", variant="primary")
        summary = gr.Markdown()
        with gr.Row():
            confusion = gr.HTML()
            history = gr.Dataframe(
                pd.DataFrame(columns=["model", "hidden", "backend", "precision", "accuracy %", "train ms"]),
                label="Run history",
                interactive=False,
            )
        run.click(train, [model, hidden, activation, precision, backend, history], [summary, confusion, history])
    with gr.Tab("CPU vs GPU"):
        gr.Markdown("Batch ELM (float32) training time as the hidden layer grows. Runs on a ZeroGPU slot.")
        go = gr.Button("Run sweep", variant="primary")
        device = gr.Markdown()
        plot = gr.LinePlot(x="hidden", y="train ms", color="backend", title="Training time (ms)", height=320)
        table = gr.Dataframe(interactive=False)
        go.click(sweep, None, [plot, table, device])
        check = gr.Button("Which GPU am I on?", size="sm")
        check.click(health, None, device)

if __name__ == "__main__":
    demo.queue(max_size=16).launch(css=CSS, theme=gr.themes.Soft(primary_hue="green"))
