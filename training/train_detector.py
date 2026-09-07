from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.paths import detector_training_output_dir, resolve_project_path  # noqa: E402
from training.detector_dataset_utils import (  # noqa: E402
    DEFAULT_DETECTOR_DATASET_SOURCE,
    prepare_detector_dataset,
)
from training.detector_resume import (  # noqa: E402
    CONFIG_MARKER_NAME,
    build_resume_config,
    read_resume_config,
    resume_config_differences,
    write_resume_config,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the class-agnostic SmartWaste YOLO waste_object detector."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DETECTOR_DATASET_SOURCE,
        help=(
            "Detector ZIP or extracted root. Default: "
            "data/dataset/detector_class/ObjectDetector_dataset.zip"
        ),
    )
    parser.add_argument("--model", default="yolo26s.pt", help="Ultralytics pretrained model.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="smartwaste_detector")
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Training output parent directory.",
    )
    parser.add_argument(
        "--deploy",
        type=Path,
        default=Path("models/best_detector.pt"),
        help="Copy best.pt here when training finishes successfully.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume an interrupted Ultralytics run from last.pt.",
    )
    parser.add_argument(
        "--no-deploy",
        action="store_true",
        help="Do not copy best.pt into models/ (useful for a 3-epoch smoke test).",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0:
        parser.error("epochs, imgsz and batch must be positive")
    if args.workers < 0:
        parser.error("workers must be >= 0")
    return args

def _device_argument(raw: str):
    value = raw.strip()
    if not value or value.lower() == "auto":
        return None
    if "," in value:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        try:
            return [int(part) for part in parts]
        except ValueError:
            return value
    try:
        return int(value)
    except ValueError:
        return value

def _print_dataset_summary(root: Path, summary: dict[str, dict[str, int]]) -> None:
    print(f"Detector dataset root: {root}")
    total_images = 0
    total_boxes = 0
    for split in ("train", "val", "test"):
        stats = summary[split]
        total_images += stats["images"]
        total_boxes += stats["boxes"]
        print(
            f"  {split:5s}: images={stats['images']:4d} "
            f"labels={stats['labels']:4d} backgrounds={stats.get('backgrounds', 0):4d} "
            f"boxes={stats['boxes']:4d}"
        )
    print(f"  total: images={total_images} boxes={total_boxes}")

def _serializable_metrics(metrics) -> dict[str, float]:
    results = getattr(metrics, "results_dict", None)
    if not isinstance(results, dict):
        raise RuntimeError("Ultralytics test evaluation did not return results_dict metrics.")

    serialized: dict[str, float] = {}
    for key, value in results.items():
        try:
            serialized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    if not serialized:
        raise RuntimeError("Ultralytics test evaluation returned no scalar metrics.")
    return serialized

def _evaluate_test_split(
    yolo_cls,
    best_path: Path,
    data_yaml: Path,
    args: argparse.Namespace,
    output_project: Path,
) -> dict[str, float]:
    print("Evaluating best detector checkpoint on the independent test split...")
    best_model = yolo_cls(str(best_path))
    val_kwargs = {
        "data": str(data_yaml),
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(output_project),
        "name": f"{args.name}_test",
        "exist_ok": True,
    }
    device = _device_argument(args.device)
    if device is not None:
        val_kwargs["device"] = device
    metrics = best_model.val(**val_kwargs)
    serialized = _serializable_metrics(metrics)

    metrics_path = best_path.parent.parent / "test_metrics.json"
    metrics_path.write_text(
        json.dumps(serialized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Detector test metrics: {metrics_path.resolve()}")
    for key, value in serialized.items():
        print(f"  {key}: {value:.6f}")
    return serialized

def _clear_stale_fresh_checkpoints(run_dir: Path) -> None:
    """Prevent a fresh run from inheriting last.pt/best.pt from an older run."""
    weights_dir = run_dir / "weights"
    removed: list[Path] = []
    for name in ("last.pt", "best.pt"):
        checkpoint = weights_dir / name
        if checkpoint.is_file():
            checkpoint.unlink()
            removed.append(checkpoint)
    if removed:
        print("Fresh detector run: removed stale checkpoint(s):")
        for checkpoint in removed:
            print(f"  {checkpoint}")

def main() -> int:
    args = parse_args()
    resume_config = build_resume_config(
        args.data,
        model=args.model,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
    )
    root, data_yaml, summary = prepare_detector_dataset(args.data)
    _print_dataset_summary(root, summary)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run train_detector.bat or install "
            "requirements.txt."
        ) from exc

    output_project = detector_training_output_dir() if args.project is None else resolve_project_path(args.project)
    output_project.mkdir(parents=True, exist_ok=True)
    expected_run_dir = output_project / args.name

    if args.resume is not None:
        resume_path = resolve_project_path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_marker = resume_path.parent.parent / CONFIG_MARKER_NAME
        differences = resume_config_differences(
            read_resume_config(resume_marker),
            resume_config,
        )
        if differences:
            details = "\n".join(f"  - {item}" for item in differences)
            raise RuntimeError(
                "Refusing to resume detector training because the current training "
                f"configuration does not match the interrupted run:\n{details}\n"
                "Start a fresh detector run instead."
            )
        print(f"Resuming detector training: {resume_path}")
        model = YOLO(str(resume_path))
        model.train(resume=True)
    else:
        _clear_stale_fresh_checkpoints(expected_run_dir)
        write_resume_config(expected_run_dir / CONFIG_MARKER_NAME, resume_config)
        print(f"Starting detector training from pretrained model: {args.model}")
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=_device_argument(args.device),
            patience=args.patience,
            seed=args.seed,
            deterministic=True,
            cache=False,
            project=str(output_project),
            name=args.name,
            exist_ok=True,
        )

    trainer = getattr(model, "trainer", None)
    best_value = getattr(trainer, "best", None) if trainer is not None else None
    save_dir_value = getattr(trainer, "save_dir", None) if trainer is not None else None
    best_path = Path(best_value) if best_value else None
    if best_path is None or not best_path.is_file():
        if save_dir_value:
            candidate = Path(save_dir_value) / "weights" / "best.pt"
        else:
            candidate = output_project / args.name / "weights" / "best.pt"
        best_path = candidate

    if not best_path.is_file():
        raise FileNotFoundError(
            "Ultralytics reported training success but best.pt was not found at "
            f"{best_path}"
        )

    print(f"Best detector checkpoint: {best_path.resolve()}")
    _evaluate_test_split(YOLO, best_path, data_yaml, args, output_project)

    if not args.no_deploy:
        deploy_path = resolve_project_path(args.deploy)
        deploy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, deploy_path)
        print(f"Deploy detector checkpoint: {deploy_path}")
    else:
        print("Smoke-test mode: best.pt was not copied into models/.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
