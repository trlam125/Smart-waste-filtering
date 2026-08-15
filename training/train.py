from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.paths import (  # noqa: E402
    ood_reference_path,
    resolve_project_path,
    torch_home,
    training_output_dir,
)

os.environ["TORCH_HOME"] = str(torch_home())

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from app.class_schema import WASTE_CLASS_KEYS  # noqa: E402
from training.dataset_utils import DEFAULT_DATASET_SOURCE, prepare_dataset  # noqa: E402
from training.build_ood_reference import build_ood_reference  # noqa: E402
from app.model_factory import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    SUPPORTED_ARCHITECTURES,
    build_eval_transform,
    build_train_transform,
    create_model,
)


class CanonicalImageFolder(ImageFolder):
    """ImageFolder that forces the exact project class order."""

    def find_classes(self, directory: str):  # type: ignore[override]
        root = Path(directory)
        present = {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")}
        expected = set(WASTE_CLASS_KEYS)
        missing = expected - present
        extra = present - expected
        if missing or extra:
            messages: list[str] = []
            if missing:
                messages.append(f"missing={sorted(missing)}")
            if extra:
                messages.append(f"extra={sorted(extra)}")
            raise RuntimeError(
                f"Split {root} does not match the canonical 11-class schema: " + ", ".join(messages)
            )
        classes = list(WASTE_CLASS_KEYS)
        return classes, {name: index for index, name in enumerate(classes)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Smart Waste Scanner on the canonical 11-class dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET_SOURCE,
        help=(
            "Dataset ZIP or extracted dataset root. Default: "
            "data/dataset/dataset.zip"
        ),
    )
    parser.add_argument(
        "--arch",
        choices=SUPPORTED_ARCHITECTURES,
        default="efficientnet_b0",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument(
        "--class-weighting",
        choices=("none", "balanced", "sqrt"),
        default="sqrt",
        help="Loss weighting for the imbalanced 11-class dataset.",
    )
    default_workers = 0 if os.name == "nt" else min(8, os.cpu_count() or 2)
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="DataLoader workers. Defaults to 0 on Windows for CUDA stability.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start from ImageNet weights (recommended).",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mixed precision on CUDA.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Directory for checkpoints and metrics. On Colab the default is "
            "/content/smart_waste_scanner_runtime/training/<arch>; elsewhere runs/<arch>."
        ),
    )
    parser.add_argument(
        "--deploy",
        type=Path,
        default=Path("models/best_model.pt"),
        help="Copy the best inference checkpoint here after training.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a last_checkpoint.pt created by this script.",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = training_output_dir(args.arch)
    return args


def resolve_dataset_root(path: Path) -> Path:
    return prepare_dataset(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducible cuDNN selection. benchmark=True can choose different kernels
    # between runs even when all RNGs are seeded.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("--device mps was requested but Apple MPS is unavailable")
    return torch.device(name)


def class_counts(dataset: CanonicalImageFolder) -> list[int]:
    counts = Counter(int(target) for target in dataset.targets)
    return [counts.get(i, 0) for i in range(len(WASTE_CLASS_KEYS))]


def class_weights(counts: list[int], mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    if any(value <= 0 for value in counts):
        raise RuntimeError(f"Every class needs at least one training image, got: {counts}")

    total = float(sum(counts))
    n_classes = float(len(counts))
    balanced = np.array([total / (n_classes * count) for count in counts], dtype=np.float32)
    if mode == "sqrt":
        balanced = np.sqrt(balanced)
    balanced = balanced / balanced.mean()
    return torch.tensor(balanced, dtype=torch.float32)


def build_loaders(args: argparse.Namespace, data_root: Path, device: torch.device):
    train_transform = build_train_transform(args.image_size)
    eval_transform = build_eval_transform(args.image_size)

    train_ds = CanonicalImageFolder(data_root / "train", transform=train_transform)
    val_ds = CanonicalImageFolder(data_root / "val", transform=eval_transform)
    test_ds = CanonicalImageFolder(data_root / "test", transform=eval_transform)

    expected_mapping = {name: i for i, name in enumerate(WASTE_CLASS_KEYS)}
    for split_name, dataset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        if dataset.class_to_idx != expected_mapping:
            raise RuntimeError(f"Unexpected class mapping in {split_name}: {dataset.class_to_idx}")

    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
        **loader_kwargs,
    )
    # Validation/test workers do not need to stay alive for the whole training run.
    # This avoids retaining extra worker processes and DLL mappings on Windows.
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
        **loader_kwargs,
    )
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def confusion_metrics(confusion: torch.Tensor) -> dict[str, Any]:
    cm = confusion.to(torch.float64)
    tp = torch.diag(cm)
    support = cm.sum(dim=1)
    predicted = cm.sum(dim=0)
    precision = tp / predicted.clamp_min(1.0)
    recall = tp / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    accuracy = float(tp.sum() / cm.sum().clamp_min(1.0))

    per_class = {
        key: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, key in enumerate(WASTE_CLASS_KEYS)
    }
    return {
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
    }


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": [int(value) for value in numpy_state[1]],
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return payload


def restore_rng_state(payload: dict[str, Any] | None) -> None:
    if not isinstance(payload, dict):
        return

    python_state = payload.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = payload.get("numpy")
    if isinstance(numpy_state, dict):
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["pos"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )

    torch_state = payload.get("torch")
    if torch.is_tensor(torch_state):
        torch.set_rng_state(torch_state.cpu())

    cuda_states = payload.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_states, list) and cuda_states:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states if torch.is_tensor(state)])


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    amp_enabled: bool,
    grad_clip: float,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    confusion = torch.zeros((len(WASTE_CLASS_KEYS), len(WASTE_CLASS_KEYS)), dtype=torch.int64)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            if device.type == "cuda":
                autocast_context = torch.amp.autocast("cuda", enabled=amp_enabled)
            else:
                autocast_context = nullcontext()
            with autocast_context:
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                assert optimizer is not None
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        batch_size = int(targets.size(0))
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        predictions = logits.argmax(dim=1)
        flat = (targets * len(WASTE_CLASS_KEYS) + predictions).detach().to("cpu")
        confusion += torch.bincount(
            flat,
            minlength=len(WASTE_CLASS_KEYS) * len(WASTE_CLASS_KEYS),
        ).reshape(len(WASTE_CLASS_KEYS), len(WASTE_CLASS_KEYS))

    metrics = confusion_metrics(confusion)
    metrics["loss"] = total_loss / max(1, total_samples)
    metrics["confusion_matrix"] = confusion.tolist()
    return metrics


def cosine_learning_rate(base_lr: float, epoch: int, total_epochs: int) -> float:
    """Cosine LR from base_lr at epoch 1 to 0 at the final configured epoch."""
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if total_epochs == 1:
        return float(base_lr)
    progress = max(0.0, min(1.0, (max(1, epoch) - 1) / (total_epochs - 1)))
    return float(base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


@torch.inference_mode()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits_parts.append(model(images).detach().to("cpu", dtype=torch.float32))
        target_parts.append(targets.detach().to("cpu", dtype=torch.long))
    if not logits_parts:
        raise RuntimeError("Validation loader is empty; cannot calibrate temperature")
    return torch.cat(logits_parts, dim=0), torch.cat(target_parts, dim=0)


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    min_temperature: float = 0.05,
    max_temperature: float = 10.0,
    iterations: int = 64,
) -> dict[str, float]:
    """Fit one scalar temperature on validation logits by bounded golden-section search."""
    import torch.nn.functional as F

    if logits.ndim != 2 or targets.ndim != 1 or logits.size(0) != targets.size(0):
        raise ValueError("Invalid logits/targets shapes for temperature calibration")

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        return float(F.cross_entropy(logits / temperature, targets).item())

    left = math.log(min_temperature)
    right = math.log(max_temperature)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc = objective(c)
    fd = objective(d)
    for _ in range(max(8, iterations)):
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = objective(d)

    best_log_temperature = (left + right) / 2.0
    temperature = math.exp(best_log_temperature)
    nll_before = objective(0.0)
    nll_after = objective(best_log_temperature)
    if not math.isfinite(nll_after) or nll_after > nll_before:
        temperature = 1.0
        nll_after = nll_before
    return {
        "temperature": float(temperature),
        "validation_nll_before": float(nll_before),
        "validation_nll_after": float(nll_after),
    }


def deployment_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float,
) -> dict[str, Any]:
    """Evaluate the same temperature-scaled probabilities used by deployment."""
    import torch.nn.functional as F

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be positive and finite, got {temperature}")
    raw_logits = logits.to(dtype=torch.float32)
    calibrated_logits = raw_logits / float(temperature)
    predictions = calibrated_logits.argmax(dim=1)
    flat = targets * len(WASTE_CLASS_KEYS) + predictions
    confusion = torch.bincount(
        flat,
        minlength=len(WASTE_CLASS_KEYS) * len(WASTE_CLASS_KEYS),
    ).reshape(len(WASTE_CLASS_KEYS), len(WASTE_CLASS_KEYS))
    metrics = confusion_metrics(confusion)
    metrics["loss"] = float(F.cross_entropy(calibrated_logits, targets).item())
    metrics["nll_before_temperature"] = float(F.cross_entropy(raw_logits, targets).item())
    metrics["nll_after_temperature"] = metrics["loss"]
    metrics["temperature"] = float(temperature)
    metrics["confusion_matrix"] = confusion.tolist()
    return metrics


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_confusion_csv(path: Path, confusion: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual\\predicted", *WASTE_CLASS_KEYS])
        for key, row in zip(WASTE_CLASS_KEYS, confusion):
            writer.writerow([key, *row])


def save_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    fields = [
        "epoch",
        "lr",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in history])


def load_resume(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
    device: torch.device,
    *,
    expected_arch: str,
    expected_image_size: int,
) -> tuple[int, float, int, list[dict[str, Any]], dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if tuple(checkpoint.get("class_names", ())) != WASTE_CLASS_KEYS:
        raise RuntimeError("Resume checkpoint uses a different class schema")
    if str(checkpoint.get("arch", "")).strip().lower() != expected_arch:
        raise RuntimeError("Resume checkpoint architecture does not match --arch")
    if int(checkpoint.get("image_size", expected_image_size)) != expected_image_size:
        raise RuntimeError("Resume checkpoint image_size does not match --image-size")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and isinstance(scaler_state, dict):
        scaler.load_state_dict(scaler_state)
    restore_rng_state(checkpoint.get("rng_state"))
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_val_macro_f1", -math.inf)),
        int(checkpoint.get("epochs_without_improvement", 0)),
        list(checkpoint.get("history", [])),
        checkpoint,
    )


def main() -> int:
    args = parse_args()
    if args.image_size < 96:
        raise ValueError("--image-size should be at least 96")
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("epochs, batch-size and lr must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    data_root = resolve_dataset_root(args.data)
    output_dir = resolve_project_path(args.output)
    deploy_path = resolve_project_path(args.deploy)
    output_dir.mkdir(parents=True, exist_ok=True)
    deploy_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(
        args, data_root, device
    )
    counts = class_counts(train_ds)
    weights = class_weights(counts, args.class_weighting)

    print(f"Dataset: {data_root}")
    print(f"Device: {device}")
    print(f"Architecture: {args.arch} | image_size={args.image_size}")
    print(f"Samples: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    for key, count in zip(WASTE_CLASS_KEYS, counts):
        print(f"  {key:14s} train={count}")

    model = create_model(args.arch, len(WASTE_CLASS_KEYS), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = make_grad_scaler(amp_enabled)

    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_checkpoint.pt"
    start_epoch = 1
    best_val_macro_f1 = -math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume is not None:
        resume_path = resolve_project_path(args.resume)
        (
            start_epoch,
            best_val_macro_f1,
            epochs_without_improvement,
            history,
            resume_checkpoint,
        ) = load_resume(
            resume_path,
            model,
            optimizer,
            scaler,
            device,
            expected_arch=args.arch,
            expected_image_size=args.image_size,
        )

        # Restore the historical best checkpoint into a new output directory. New v4
        # last checkpoints are self-contained; older checkpoints fall back to the
        # sibling best_model.pt when available.
        embedded_best = resume_checkpoint.get("best_model_checkpoint")
        if isinstance(embedded_best, dict) and isinstance(embedded_best.get("model_state_dict"), dict):
            torch.save(embedded_best, best_path)
        else:
            sibling_best = resume_path.parent / "best_model.pt"
            if sibling_best.is_file():
                if sibling_best.resolve() != best_path.resolve():
                    shutil.copy2(sibling_best, best_path)
            elif not best_path.is_file():
                print(
                    "Warning: resume checkpoint has no recoverable historical best_model.pt; "
                    "the first resumed epoch will establish a new best checkpoint."
                )
                best_val_macro_f1 = -math.inf
                epochs_without_improvement = 0
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        lr_now = cosine_learning_rate(args.lr, epoch, args.epochs)
        set_optimizer_lr(optimizer, lr_now)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            amp_enabled=False,
            grad_clip=0.0,
        )

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        save_history_csv(output_dir / "history.csv", history)

        improved = val_metrics["macro_f1"] > (best_val_macro_f1 + args.min_delta)
        if improved:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            epochs_without_improvement = 0
            best_payload = {
                "format_version": 4,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "arch": args.arch,
                "image_size": args.image_size,
                "class_names": list(WASTE_CLASS_KEYS),
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
                "temperature": 1.0,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "train_class_counts": dict(zip(WASTE_CLASS_KEYS, counts)),
                "model_state_dict": cpu_state_dict(model),
            }
            torch.save(best_payload, best_path)
            save_json(output_dir / "best_val_metrics.json", val_metrics)
            save_confusion_csv(output_dir / "best_val_confusion_matrix.csv", val_metrics["confusion_matrix"])
        else:
            epochs_without_improvement += 1

        try:
            current_best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
        except TypeError:
            current_best_checkpoint = torch.load(best_path, map_location="cpu")
        last_payload = {
            "format_version": 4,
            "arch": args.arch,
            "image_size": args.image_size,
            "class_names": list(WASTE_CLASS_KEYS),
            "epoch": epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "lr_schedule": {
                "name": "cosine_absolute_epoch",
                "base_lr": args.lr,
                "total_epochs": args.epochs,
            },
            "model_state_dict": cpu_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "rng_state": capture_rng_state(),
            "best_model_checkpoint": current_best_checkpoint,
        }
        torch.save(last_payload, last_path)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} f1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} f1={val_metrics['macro_f1']:.4f}" +
            ("  <-- best" if improved else "")
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epochs_without_improvement} epochs without macro-F1 improvement.")
            break

    if not best_path.is_file():
        # This can only happen when resuming a legacy checkpoint without its sibling
        # best_model.pt and requesting no additional epochs. Promote the resumed model
        # rather than crashing; the summary records that this fallback occurred.
        fallback_val_f1 = float(history[-1].get("val_macro_f1", -math.inf)) if history else -math.inf
        best_val_macro_f1 = fallback_val_f1
        fallback_payload = {
            "format_version": 4,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "arch": args.arch,
            "image_size": args.image_size,
            "class_names": list(WASTE_CLASS_KEYS),
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "temperature": 1.0,
            "epoch": max(0, start_epoch - 1),
            "val_metrics": {"macro_f1": fallback_val_f1, "resume_fallback": True},
            "train_class_counts": dict(zip(WASTE_CLASS_KEYS, counts)),
            "model_state_dict": cpu_state_dict(model),
        }
        torch.save(fallback_payload, best_path)
        print("Warning: promoted resumed model because no historical best checkpoint was recoverable.")

    try:
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    except TypeError:
        best_checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)

    val_logits, val_targets = collect_logits(model, val_loader, device)
    calibration = fit_temperature(val_logits, val_targets)
    best_checkpoint["temperature"] = calibration["temperature"]
    best_checkpoint["temperature_calibration"] = calibration
    torch.save(best_checkpoint, best_path)
    save_json(output_dir / "temperature_calibration.json", calibration)
    print(
        f"Temperature calibration: T={calibration['temperature']:.4f} | "
        f"val NLL {calibration['validation_nll_before']:.4f} -> "
        f"{calibration['validation_nll_after']:.4f}"
    )

    test_logits, test_targets = collect_logits(model, test_loader, device)
    test_metrics = deployment_metrics_from_logits(
        test_logits,
        test_targets,
        temperature=float(calibration["temperature"]),
    )
    save_json(output_dir / "test_metrics.json", test_metrics)
    save_confusion_csv(output_dir / "test_confusion_matrix.csv", test_metrics["confusion_matrix"])

    summary = {
        "dataset_root": str(data_root),
        "architecture": args.arch,
        "image_size": args.image_size,
        "class_names": list(WASTE_CLASS_KEYS),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "train_class_counts": dict(zip(WASTE_CLASS_KEYS, counts)),
        "class_weighting": args.class_weighting,
        "best_val_macro_f1": best_val_macro_f1,
        "temperature_calibration": calibration,
        "test_metrics": test_metrics,
    }
    save_json(output_dir / "training_summary.json", summary)

    shutil.copy2(best_path, deploy_path)
    deploy_ood_reference_path = ood_reference_path()
    # The deployed app refuses to silently use an OOD bank from another checkpoint.
    # Rebuild the persistent bank whenever training publishes a new deploy model.
    model.to("cpu")
    # Release training-only GPU state before constructing the deploy-time OOD
    # feature extractor; otherwise a second model can needlessly compete with
    # optimizer/scaler tensors for VRAM on smaller GPUs.
    del model
    del optimizer
    if scaler is not None:
        del scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ood_summary = build_ood_reference(
        data_root,
        deploy_path,
        deploy_ood_reference_path,
        batch_size=max(8, args.batch_size),
        device_preference=args.device,
    )
    summary["ood_reference"] = ood_summary
    save_json(output_dir / "training_summary.json", summary)
    print("\nTraining complete")
    print(f"Best checkpoint: {best_path}")
    print(f"Deploy checkpoint: {deploy_path}")
    print(f"OOD reference (persistent): {deploy_ood_reference_path}")
    print(
        f"Test: accuracy={test_metrics['accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
