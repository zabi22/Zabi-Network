from __future__ import annotations

import io
import json
import base64
import os
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, Response

from core.config import ExperimentConfig
from src.data import build_dataloaders, build_folder_dataloaders, load_single_image, FolderImageDataset
from core.losses import build_loss_fn, ConditionalGateLoss
from core.metrics import MetricsAccumulator
from src.model import build_model, HybridNet
from core.utils import set_seed, get_device, GradientInspector

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# ---------------------------------------------------------------------------
# Global state shared between the web server and the background training thread
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {
    "status": "idle",           # idle | training | finished | error
    "progress": 0.0,
    "current_epoch": 0,
    "total_epochs": 0,
    "logs": [],                 # list of log-line strings
    "epoch_metrics": [],        # per-epoch dict {epoch, train_loss, val_loss, ...}
    "test_metrics": {},
    "model_summary": "",
    "error": "",
    "stop_requested": False,
    "config": ExperimentConfig().to_dict(),
    "trained_model": None,      # holds the trained model for prediction
    "class_names": None,        # class label names
    "dataset_folder": None,     # path to user-provided image folder
}
_lock = threading.Lock()


def _log(msg: str) -> None:
    with _lock:
        _state["logs"].append(msg)


def _set(key: str, value: Any) -> None:
    with _lock:
        _state[key] = value


# ---------------------------------------------------------------------------
# Background training routine
# ---------------------------------------------------------------------------
def _training_thread(cfg: ExperimentConfig) -> None:
    try:
        _set("status", "training")
        _set("stop_requested", False)
        _set("logs", [])
        _set("epoch_metrics", [])
        _set("test_metrics", {})
        _set("error", "")

        set_seed(cfg.seed)
        if cfg.model.width_multiplier != 1.0 or cfg.model.depth_multiplier != 1.0:
            cfg = cfg.apply_width_depth_multipliers()

        _log("Building model...")
        model = build_model(cfg.model)

        # capture summary
        param_counts = model.get_param_count()
        lines = ["MODEL SUMMARY", "=" * 50]
        for name, count in param_counts.items():
            label = name.upper() if name in ("total", "trainable") else name
            lines.append(f"  {label:30s} : {count:>12,} params")
        _set("model_summary", "\n".join(lines))
        _log("\n".join(lines))

        device = get_device(cfg.distributed.local_rank)
        model = model.to(device)
        _log(f"Device: {device}")

        _log("Building data loaders...")
        folder = None
        with _lock:
            folder = _state.get("dataset_folder")

        class_names = [f"class_{i}" for i in range(cfg.model.num_classes)]

        if folder and os.path.isdir(folder):
            _log(f"Loading real images from: {folder}")
            train_loader, val_loader, test_loader, class_names = build_folder_dataloaders(
                folder, cfg.model, batch_size=cfg.data.batch_size, seed=cfg.seed
            )
            cfg.model.num_classes = len(class_names)
            _log(f"Found {len(class_names)} classes: {class_names}")
        else:
            train_loader, val_loader, test_loader = build_dataloaders(
                cfg.data, cfg.model, distributed=False, seed=cfg.seed
            )
        _set("class_names", class_names)
        _log(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        optimizer = _build_optimizer(model, cfg)
        loss_fn = build_loss_fn(
            cfg.train.loss_fn, cfg.train.label_smoothing,
            cfg.train.focal_gamma, cfg.train.focal_alpha,
        ).to(device)
        gate_loss_fn = ConditionalGateLoss() if cfg.model.conditional_execution else None

        use_amp = cfg.train.mixed_precision and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        _set("total_epochs", cfg.train.epochs)
        best_val_loss = float("inf")

        for epoch in range(cfg.train.epochs):
            with _lock:
                if _state["stop_requested"]:
                    _log("Training stopped by user.")
                    _set("status", "finished")
                    return

            _set("current_epoch", epoch + 1)
            _set("progress", epoch / max(cfg.train.epochs, 1) * 100)
            _log(f"\n--- Epoch {epoch + 1}/{cfg.train.epochs} ---")

            # TRAIN
            model.train()
            train_acc = MetricsAccumulator(cfg.model.num_classes)
            optimizer.zero_grad(set_to_none=True)
            for batch_idx, (images, targets) in enumerate(train_loader):
                with _lock:
                    if _state["stop_requested"]:
                        _log("Training stopped by user.")
                        _set("status", "finished")
                        return

                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                amp_dtype = torch.float16 if use_amp else torch.float32
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits, info = model(images)
                    loss = loss_fn(logits, targets)
                    if gate_loss_fn and "gate_probs" in info:
                        loss = loss + gate_loss_fn(info["gate_probs"])
                    loss = loss / cfg.train.accumulation_steps

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % cfg.train.accumulation_steps == 0:
                    if cfg.train.gradient_clip_norm > 0:
                        if use_amp:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip_norm)
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                preds = logits.argmax(dim=-1)
                train_acc.update(preds, targets, loss.item() * cfg.train.accumulation_steps, images.size(0))

            train_metrics = train_acc.compute_all()

            # VAL
            model.eval()
            val_acc = MetricsAccumulator(cfg.model.num_classes)
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    amp_dtype = torch.float16 if use_amp else torch.float32
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        logits, _ = model(images)
                        loss = loss_fn(logits, targets)
                    preds = logits.argmax(dim=-1)
                    val_acc.update(preds, targets, loss.item(), images.size(0))

            val_metrics = val_acc.compute_all()

            row = {
                "epoch": epoch + 1,
                "train_loss": round(float(train_metrics["loss"]), 4),
                "train_accuracy": round(float(train_metrics["accuracy"]), 4),
                "train_f1": round(float(train_metrics["macro_f1"]), 4),
                "val_loss": round(float(val_metrics["loss"]), 4),
                "val_accuracy": round(float(val_metrics["accuracy"]), 4),
                "val_f1": round(float(val_metrics["macro_f1"]), 4),
                "val_precision": round(float(val_metrics["macro_precision"]), 4),
                "val_recall": round(float(val_metrics["macro_recall"]), 4),
            }
            with _lock:
                _state["epoch_metrics"].append(row)

            is_best = row["val_loss"] < best_val_loss
            if is_best:
                best_val_loss = row["val_loss"]

            _log(
                f"  train_loss={row['train_loss']:.4f}  train_acc={row['train_accuracy']:.4f}  "
                f"val_loss={row['val_loss']:.4f}  val_acc={row['val_accuracy']:.4f}  "
                f"val_f1={row['val_f1']:.4f}"
                + ("  *best*" if is_best else "")
            )

        _set("progress", 100.0)

        # TEST
        _log("\nRunning test evaluation...")
        model.eval()
        test_acc = MetricsAccumulator(cfg.model.num_classes)
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                amp_dtype = torch.float16 if use_amp else torch.float32
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits, _ = model(images)
                    loss = loss_fn(logits, targets)
                preds = logits.argmax(dim=-1)
                test_acc.update(preds, targets, loss.item(), images.size(0))

        tm = test_acc.compute_all()
        test_result = {
            "test_loss": round(float(tm["loss"]), 4),
            "test_accuracy": round(float(tm["accuracy"]), 4),
            "test_macro_f1": round(float(tm["macro_f1"]), 4),
            "test_macro_precision": round(float(tm["macro_precision"]), 4),
            "test_macro_recall": round(float(tm["macro_recall"]), 4),
            "test_weighted_f1": round(float(tm["weighted_f1"]), 4),
            "confusion_matrix": test_acc.confusion_matrix_string(),
        }
        _set("test_metrics", test_result)
        _log(f"\nTest results: loss={test_result['test_loss']:.4f}  "
             f"acc={test_result['test_accuracy']:.4f}  f1={test_result['test_macro_f1']:.4f}")
        _log("\nConfusion Matrix:\n" + test_result["confusion_matrix"])
        _set("trained_model", model)
        _set("status", "finished")

    except Exception:
        _set("error", traceback.format_exc())
        _log(f"\nERROR:\n{traceback.format_exc()}")
        _set("status", "error")


def _build_optimizer(model, cfg):
    tc = cfg.train
    params = [p for p in model.parameters() if p.requires_grad]
    if tc.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=tc.learning_rate, weight_decay=tc.weight_decay, betas=tc.betas, eps=tc.eps)
    elif tc.optimizer == "adam":
        return torch.optim.Adam(params, lr=tc.learning_rate, weight_decay=tc.weight_decay, betas=tc.betas, eps=tc.eps)
    elif tc.optimizer == "sgd":
        return torch.optim.SGD(params, lr=tc.learning_rate, weight_decay=tc.weight_decay, momentum=tc.momentum)
    return torch.optim.AdamW(params, lr=tc.learning_rate)


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(_state["config"])


@app.route("/api/config", methods=["POST"])
def set_config():
    data = request.get_json(force=True)
    with _lock:
        _state["config"] = data
    return jsonify({"ok": True})


@app.route("/api/train", methods=["POST"])
def start_training():
    with _lock:
        if _state["status"] == "training":
            return jsonify({"ok": False, "error": "Training already in progress"}), 409

    cfg = ExperimentConfig._from_dict(_state["config"])
    overrides = request.get_json(force=True) or {}
    if "epochs" in overrides:
        cfg.train.epochs = int(overrides["epochs"])
    if "batch_size" in overrides:
        cfg.data.batch_size = int(overrides["batch_size"])
    if "learning_rate" in overrides:
        cfg.train.learning_rate = float(overrides["learning_rate"])
    if "num_train_samples" in overrides:
        cfg.data.num_train_samples = int(overrides["num_train_samples"])
    if "num_val_samples" in overrides:
        cfg.data.num_val_samples = int(overrides["num_val_samples"])
    if "num_test_samples" in overrides:
        cfg.data.num_test_samples = int(overrides["num_test_samples"])

    cfg.train.mixed_precision = False
    cfg.data.num_workers = 0
    cfg.log.use_tensorboard = False

    with _lock:
        _state["config"] = cfg.to_dict()

    t = threading.Thread(target=_training_thread, args=(cfg,), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def stop_training():
    _set("stop_requested", True)
    return jsonify({"ok": True})


@app.route("/api/status")
def get_status():
    with _lock:
        return jsonify({
            "status": _state["status"],
            "progress": _state["progress"],
            "current_epoch": _state["current_epoch"],
            "total_epochs": _state["total_epochs"],
            "epoch_metrics": _state["epoch_metrics"],
            "test_metrics": _state["test_metrics"],
            "model_summary": _state["model_summary"],
            "error": _state["error"],
            "log_count": len(_state["logs"]),
        })


@app.route("/api/logs")
def get_logs():
    start = int(request.args.get("start", 0))
    with _lock:
        return jsonify({"logs": _state["logs"][start:]})


@app.route("/api/stream")
def stream():
    def generate():
        last_len = 0
        while True:
            with _lock:
                status = _state["status"]
                logs = _state["logs"][last_len:]
                last_len = len(_state["logs"])
                metrics = _state["epoch_metrics"]
                progress = _state["progress"]
            data = json.dumps({
                "status": status,
                "new_logs": logs,
                "epoch_metrics": metrics,
                "progress": progress,
            })
            yield f"data: {data}\n\n"
            if status in ("finished", "error", "idle"):
                break
            time.sleep(1)
    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/profile", methods=["POST"])
def run_profile():
    cfg = ExperimentConfig._from_dict(_state["config"])
    cfg.train.mixed_precision = False
    set_seed(cfg.seed)
    model = build_model(cfg.model)
    counts = model.get_param_count()
    batch = 4
    x = torch.randn(batch, cfg.model.input_channels, cfg.model.input_height, cfg.model.input_width)
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        logits, _ = model(x)
    elapsed = (time.perf_counter() - start) * 1000
    return jsonify({
        "param_counts": counts,
        "forward_time_ms": round(elapsed, 2),
        "output_shape": list(logits.shape),
        "batch_size": batch,
    })


@app.route("/api/set-dataset-folder", methods=["POST"])
def set_dataset_folder():
    data = request.get_json(force=True)
    folder = data.get("folder", "").strip()
    if not folder:
        _set("dataset_folder", None)
        return jsonify({"ok": True, "message": "Cleared. Will use synthetic data."})
    if not os.path.isdir(folder):
        return jsonify({"ok": False, "error": f"Folder not found: {folder}"}), 400
    ds = FolderImageDataset(root=folder)
    _set("dataset_folder", folder)
    return jsonify({
        "ok": True,
        "num_images": len(ds),
        "classes": ds.class_names,
        "num_classes": ds.num_classes,
    })


@app.route("/api/upload-images", methods=["POST"])
def upload_images():
    """Upload images into class subfolders for training."""
    class_name = request.form.get("class_name", "default").strip()
    if not class_name:
        class_name = "default"
    dest = os.path.join(UPLOAD_FOLDER, class_name)
    os.makedirs(dest, exist_ok=True)
    files = request.files.getlist("images")
    saved = 0
    for f in files:
        if f.filename:
            safe_name = f.filename.replace("/", "_").replace("\\", "_")
            f.save(os.path.join(dest, safe_name))
            saved += 1
    ds = FolderImageDataset(root=UPLOAD_FOLDER)
    _set("dataset_folder", UPLOAD_FOLDER)
    return jsonify({
        "ok": True,
        "saved": saved,
        "class": class_name,
        "total_images": len(ds),
        "classes": ds.class_names,
        "num_classes": ds.num_classes,
    })


@app.route("/api/dataset-info")
def dataset_info():
    with _lock:
        folder = _state.get("dataset_folder")
    if folder and os.path.isdir(folder):
        ds = FolderImageDataset(root=folder)
        return jsonify({
            "source": "folder",
            "path": folder,
            "num_images": len(ds),
            "classes": ds.class_names,
            "num_classes": ds.num_classes,
        })
    return jsonify({"source": "synthetic", "path": None})


@app.route("/api/predict", methods=["POST"])
def predict():
    with _lock:
        model = _state.get("trained_model")
        class_names = _state.get("class_names")
    if model is None:
        return jsonify({"ok": False, "error": "No trained model. Train first."}), 400

    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image file uploaded."}), 400

    file = request.files["image"]
    image_bytes = file.read()
    cfg = ExperimentConfig._from_dict(_state["config"])
    tensor = load_single_image(
        image_bytes,
        channels=cfg.model.input_channels,
        height=cfg.model.input_height,
        width=cfg.model.input_width,
    )

    device = next(model.parameters()).device
    tensor = tensor.to(device)
    model.eval()
    with torch.no_grad():
        logits, _ = model(tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0)

    top_k = min(5, probs.size(0))
    values, indices = probs.topk(top_k)
    predictions = []
    for i in range(top_k):
        idx = indices[i].item()
        label = class_names[idx] if class_names and idx < len(class_names) else f"class_{idx}"
        predictions.append({
            "class_index": idx,
            "class_name": label,
            "confidence": round(values[i].item() * 100, 2),
        })

    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename else "png"
    mime = f"image/{'jpeg' if ext in ('jpg','jpeg') else ext}"

    return jsonify({
        "ok": True,
        "predictions": predictions,
        "image_data": f"data:{mime};base64,{img_b64}",
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
