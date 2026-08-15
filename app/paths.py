from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

COLAB_RUNTIME_ROOT = Path("/content/smart_waste_scanner_runtime")
LOCAL_RUNTIME_ROOT = PROJECT_ROOT / "data" / ".runtime"


def resolve_project_path(value: Path | str) -> Path:
    """Resolve a project-managed path from PROJECT_ROOT, never from cwd."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def project_path_from_env(name: str, default: Path | str) -> Path:
    raw = os.getenv(name, "").strip()
    return resolve_project_path(raw or default)


def is_google_colab() -> bool:
    """Best-effort Colab detection without importing google.colab."""
    if any(
        os.getenv(name)
        for name in ("COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_JUPYTER_IP")
    ):
        return True
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def runtime_root(*, colab: bool | None = None) -> Path:
    """Return the disposable runtime root.

    On Colab this intentionally lives on the VM disk under /content so it is
    discarded when the runtime resets. Elsewhere it stays inside the project.
    SMARTWASTE_RUNTIME_DIR can explicitly override either default.
    """
    raw = os.getenv("SMARTWASTE_RUNTIME_DIR", "").strip()
    if raw:
        return resolve_project_path(raw)
    detected = is_google_colab() if colab is None else bool(colab)
    return (COLAB_RUNTIME_ROOT if detected else LOCAL_RUNTIME_ROOT).resolve()


def _normalized_relative(value: str) -> str:
    return value.replace("\\", "/").strip().lstrip("./").rstrip("/")


def _runtime_env_path(
    env_name: str,
    runtime_child: str,
    *,
    colab: bool | None = None,
    legacy_project_defaults: tuple[str, ...] = (),
    local_default: Path | str | None = None,
) -> Path:
    """Resolve a disposable path, migrating legacy Colab project defaults.

    Absolute custom overrides are always respected. On Colab, old relative
    disposable defaults such as data/.torch are treated as legacy values and
    moved to the disposable runtime automatically.
    """
    detected = is_google_colab() if colab is None else bool(colab)
    raw = os.getenv(env_name, "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        legacy = {_normalized_relative(item) for item in legacy_project_defaults}
        is_legacy_colab_value = (
            detected
            and not candidate.is_absolute()
            and _normalized_relative(raw) in legacy
        )
        if not is_legacy_colab_value:
            return resolve_project_path(candidate)

    if detected:
        return (runtime_root(colab=True) / runtime_child).resolve()
    if local_default is not None:
        return resolve_project_path(local_default)
    return (runtime_root(colab=False) / runtime_child).resolve()


def dataset_extract_dir(*, colab: bool | None = None) -> Path:
    return _runtime_env_path(
        "SMARTWASTE_DATASET_EXTRACT_DIR",
        "dataset_extracted",
        colab=colab,
        legacy_project_defaults=("data/dataset/_extracted",),
        local_default="data/dataset/_extracted",
    )


def torch_home(*, colab: bool | None = None) -> Path:
    return _runtime_env_path(
        "TORCH_HOME",
        "torch_cache",
        colab=colab,
        legacy_project_defaults=("data/.torch",),
        local_default="data/.torch",
    )


def ood_reference_path(*, colab: bool | None = None) -> Path:
    """Return the persistent checkpoint-specific OOD bank path.

    The OOD bank is small and expensive to rebuild, so it intentionally stays
    beside the deploy model on persistent storage, including Google Colab.
    Relative overrides are anchored to PROJECT_ROOT. The previous Colab runtime
    default is migrated back to models/ood_reference.npz automatically.
    """
    detected = is_google_colab() if colab is None else bool(colab)
    raw = os.getenv("OOD_REFERENCE_PATH", "").strip()
    persistent_default = resolve_project_path("models/ood_reference.npz")
    if not raw:
        return persistent_default

    candidate = Path(raw).expanduser()
    if detected:
        legacy_runtime = (runtime_root(colab=True) / "ood_reference.npz").resolve()
        try:
            resolved_candidate = resolve_project_path(candidate)
        except OSError:
            resolved_candidate = candidate
        if resolved_candidate == legacy_runtime:
            return persistent_default

    return resolve_project_path(candidate)


def training_output_dir(
    architecture: str,
    *,
    colab: bool | None = None,
) -> Path:
    return (
        PROJECT_ROOT
        / "runs"
        / architecture
    ).resolve()


def collection_pending_dir(collected_data_dir: Path, *, colab: bool | None = None) -> Path:
    detected = is_google_colab() if colab is None else bool(colab)
    if detected:
        return (runtime_root(colab=True) / "pending").resolve()
    return (collected_data_dir / "_pending").resolve()
