# Smart Waste Scanner AI

Waste classification application using a supervised PyTorch model with the fixed 11-class schema of the SmartWaste dataset.

## 1. Place the dataset

No manual extraction is required. Place the ZIP file in the correct location with the exact filename:

```text
Smart waste scanner/
├── data/
│   └── dataset/
│       └── classifier_class/
│           └── dataset.zip
├── training/
├── models/
├── train_model.bat
└── ...
```

`dataset.zip` can contain `train/`, `val/`, and `test/` directly, or it can have a root directory wrapping them, such as `SmartWaste_Household_Kaan_11class_maxclean_v2`.

On the first run, the code automatically extracts the dataset. The cache location depends on the environment:

```text
Local : data/.runtime/classifier_dataset_extracted/
Colab : /content/smart_waste_scanner_runtime/classifier_dataset_extracted/
```

On Colab, the extracted files stay on the runtime SSD, so Google Drive only needs to keep `dataset.zip`. The cache is reused within the same runtime. If the Colab runtime resets, `/content` is cleared and the archive is extracted again. Replacing `dataset.zip` also causes the cache to be rebuilt automatically.

## 2. Merging external datasets (optional — improve class balance)

The base dataset has class imbalance (hazardous: 484, plastic_film: 602 vs plastic_rigid: 2075).
Use `training/merge_datasets.py` to merge external public datasets and produce a new balanced ZIP
that can be passed directly to `train.py`.

### Recommended external datasets to download

| # | Dataset | Download | Classes useful for |
|---|---------|----------|--------------------|
| 1 | **Kaggle 12-class** (`mostafaabla/garbage-classification`) | [kaggle.com/datasets/mostafaabla/garbage-classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) | `hazardous` (battery), `textile` (clothes+shoes), `organic`, basic classes |
| 2 | **RealWaste** (UCI) | [archive.ics.uci.edu/dataset/908/realwaste](https://archive.ics.uci.edu/dataset/908/realwaste) | `other` (Miscellaneous Trash), `textile` (Textile Trash), real-world diversity |
| 3 | **GlobalWasteData — GWD** (arxiv 2602.07463) | see paper for link | `plastic_film` (plastic_bag/wrap), `hazardous`, broadest coverage |
| 4 | **TACO** (classification crops) | [tacodata.com](http://tacodata.com) | `plastic_film`, `hazardous` (from cropped bounding boxes) |

Priority order if you can only download some: **Kaggle 12-class → RealWaste → GWD → TACO**

### Folder layout expected after download

After extracting each dataset, place them under `data/external/`:

```text
data/
└── external/
    ├── kaggle12/          ← extracted Kaggle 12-class (flat class folders)
    │   ├── battery/
    │   ├── clothes/
    │   ├── plastic/
    │   └── ...
    ├── realwaste/         ← extracted RealWaste (flat class folders)
    │   ├── Cardboard/
    │   ├── Food Organics/
    │   └── ...
    ├── gwd/               ← extracted GlobalWasteData (flat or with train/val/test)
    │   ├── plastic_film/
    │   ├── battery/
    │   └── ...
    └── taco_crops/        ← TACO crops extracted from bounding boxes
        ├── Plastic bag & wrapper/
        ├── Plastic film/
        └── ...
```

The script **auto-detects** each dataset type from its folder names — no manual config needed.
You can also force the type with the `path:type` syntax.

### Step 1 — Preview (no files written)

Check which classes gain how many images before committing:

```bat
.venv\Scripts\python.exe training\merge_datasets.py ^
    --extra data\external\kaggle12 ^
    --extra data\external\realwaste ^
    --preview
```

### Step 2 — Dry-run (see projected final counts)

```bat
.venv\Scripts\python.exe training\merge_datasets.py ^
    --extra data\external\kaggle12 ^
    --extra data\external\realwaste ^
    --dry-run
```

Example output:

```text
Class           train    val   test    total
------------------------------------------------
plastic_rigid    2075    228    228     2531
plastic_film     1102    110    110     1322   ← added
...
hazardous        1284    128    128     1540   ← added
```

### Step 3 — Merge and produce new ZIP

```bat
.venv\Scripts\python.exe training\merge_datasets.py ^
    --extra data\external\kaggle12 ^
    --extra data\external\realwaste ^
    --extra data\external\gwd ^
    --output data\dataset\classifier_class\dataset_merged.zip
```

With all 4 datasets and a per-class cap to prevent any single source dominating:

```bat
.venv\Scripts\python.exe training\merge_datasets.py ^
    --extra data\external\kaggle12:kaggle12 ^
    --extra data\external\realwaste:realwaste ^
    --extra data\external\gwd:gwd ^
    --extra data\external\taco_crops:taco ^
    --max-per-class 800 ^
    --output data\dataset\classifier_class\dataset_merged.zip
```

The output ZIP is a self-contained drop-in replacement for `dataset.zip` and follows exactly
the same `train/val/test/<class>/` layout. The original `dataset.zip` is **never modified**.

### Step 4 — Train with the merged dataset

```bat
.venv\Scripts\python.exe training\train.py ^
    --data data\dataset\classifier_class\dataset_merged.zip ^
    --arch efficientnet_b0 ^
    --epochs 40 ^
    --batch-size 16 ^
    --lr 3e-4 ^
    --weight-decay 1e-4 ^
    --label-smoothing 0.08 ^
    --class-weighting sqrt ^
    --patience 8 ^
    --device auto ^
    --workers 0
```

Or simply double-click `train_model.bat` after setting the default dataset path,
or pass the merged ZIP explicitly:

```bat
train_model.bat data\dataset\classifier_class\dataset_merged.zip
```

### All merge options

| Option | Default | Description |
|--------|---------|-------------|
| `--base` | `data/dataset/classifier_class/dataset.zip` | Base SmartWaste dataset |
| `--extra PATH[:TYPE]` | — | External dataset (repeat for multiple). TYPE: `kaggle12`, `realwaste`, `gwd`, `taco` |
| `--output` | `<base>_merged.zip` | Output ZIP path |
| `--max-per-class N` | 0 (no limit) | Cap new images per class to prevent imbalance |
| `--val-frac` | 0.09 | Fraction of new images → val split |
| `--test-frac` | 0.09 | Fraction of new images → test split |
| `--seed` | 42 | Random seed for reproducible splits |
| `--no-dedup` | off | Skip MD5 duplicate detection (faster) |
| `--dry-run` | off | Print projected counts, write nothing |
| `--preview` | off | Fastest scan: count new images only, no base dataset needed |

### What the script guarantees

- **Original dataset unchanged** — all edits go into the new output ZIP only.
- **No cross-split leakage** — new images from external sources are split independently
  before being assigned to train/val/test; no image appears in more than one split.
- **MD5 deduplication** — images already in the base dataset are silently skipped.
- **Filename collision-safe** — new images are renamed with a short hash suffix so
  same-named files from different sources never overwrite each other.
- **11-class schema enforced** — the output ZIP will fail `inspect_dataset.py` immediately
  if any class ends up empty, catching mapping errors before training starts.

## 3. 11-class schema

The class order is fixed for training, checkpoints, inference, and feedback:

1. `plastic_rigid`
2. `plastic_film`
3. `paper`
4. `cardboard`
5. `metal`
6. `glass`
7. `organic`
8. `hazardous`
9. `electronic`
10. `textile`
11. `other`

Each `train/`, `val/`, and `test/` split must contain exactly the 11 directories listed above. The script stops immediately if any class is missing, extra, or incorrectly named.

## 4. Train on Windows

The simplest way is to double-click or run:

```bat
train_model.bat
```

The script will automatically:

1. find `data\dataset\classifier_class\dataset.zip`;
2. create `.venv` if it does not exist;
3. install training dependencies;
4. extract the dataset if needed;
5. verify that all 11 classes are present;
6. train EfficientNet-B0;
7. save the best model to `models\best_model.pt`.

You can still provide a different dataset if desired:

```bat
train_model.bat "D:\path\custom-dataset.zip"
```

## 5. Train with Python

Python 3.11 or 3.12 is recommended.

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the default dataset:

```bat
python training\inspect_dataset.py
```

Default training command:

```bat
python training\train.py --arch efficientnet_b0 --image-size 224 --epochs 40 --batch-size 16 --lr 3e-4 --weight-decay 1e-4 --label-smoothing 0.08 --class-weighting sqrt --patience 8 --device auto --workers 0 --amp
```

There is no need to pass `--data`; the default is `data\dataset\classifier_class\dataset.zip`.

If VRAM is insufficient, reduce `--batch-size` to 16 or 8. Supported architectures: `efficientnet_b0`, `mobilenet_v3_large`, `resnet18`.

When `--output` is omitted, training results are stored in `runs/<arch>/` (for example `runs/efficientnet_b0/`, `runs/mobilenet_v3_large/`, or `runs/resnet18/`). The best checkpoint is automatically copied to:

```text
models/best_model.pt
```

Resume:

```bat
python training\train.py ^
  --arch efficientnet_b0 ^
  --image-size 224 ^
  --epochs 40 ^
  --batch-size 16 ^
  --lr 3e-4 ^
  --weight-decay 1e-4 ^
  --label-smoothing 0.08 ^
  --class-weighting sqrt ^
  --patience 8 ^
  --device auto ^
  --workers 0 ^
  --amp ^
  --resume runs\efficientnet_b0\last_checkpoint.pt
```

## 6. Run the application

After `models\best_model.pt` is available:

```bat
start.bat
```

By default, the app runs at `http://localhost:8000`.

If `models/best_model.pt` is not available yet, the web app will still open, but the classification API will report that the model is not ready.

## 7. Configuration

You can copy `.env.example` to `.env`. The main variables are:

```env
WASTE_MODEL_CHECKPOINT=models/best_model.pt
WASTE_DEVICE=auto
UNKNOWN_THRESHOLD=0.60
UNCERTAINTY_MARGIN=0.10
DATABASE_PATH=data/waste_scanner.db
```

`UNKNOWN_THRESHOLD` and `UNCERTAINTY_MARGIN` should be tuned using the validation/test results of the actual model.

## 8. Feedback learning

When a user confirms or corrects a label, the app stores the L2-normalized feature vector (feature embedding), taken immediately before the classifier head, as feedback memory. Feedback is namespaced by checkpoint hash, so data from different models is not mixed together.

## 9. Docker

```bash
docker compose --env-file .env up --build waste-scanner
```

Docker Compose mounts `./models` into the container. Train the model first and place `best_model.pt` in `models/`.

## 9. Collecting real-world data

Each time a scan is saved to history, the app temporarily keeps a high-quality JPEG image. The image **only becomes training data** after the user confirms or corrects the label. At that point, the image is moved into:

```text
data/collected/
├── plastic_rigid/
├── plastic_film/
├── paper/
├── cardboard/
├── metal/
├── glass/
├── organic/
├── hazardous/
├── electronic/
├── textile/
├── other/
└── metadata.csv
```

Related configuration variables:

```env
DATASET_COLLECTION_ENABLED=true
COLLECTED_DATA_DIR=
COLLECTED_IMAGE_MAX_DIMENSION=1600
COLLECTED_JPEG_QUALITY=92
```

## 10. Run on Google Colab

Place the entire project folder in Google Drive, for example:

```text
My Drive/
└── Colab Notebooks/
    └── Smart waste scanner/
        ├── app/
        ├── data/
        │   ├── dataset/
        │   │   └── classifier_class/
        │   │       └── dataset.zip
        │   ├── collected/
        │   └── waste_scanner.db
        ├── models/
        │   └── best_model.pt
        ├── runs/
        ├── training/
        ├── launcher.py
        ├── requirements.txt
        └── ...
```

There is no need to store the extracted dataset image directory on Google Drive. When running in Colab, `training/dataset_utils.py` automatically uses:

```text
/content/smart_waste_scanner_runtime/classifier_dataset_extracted
```

as the extraction cache. Because `/content` is temporary storage for the Colab runtime, this cache is lost when the runtime resets and is automatically recreated from `data/dataset/classifier_class/dataset.zip` on the next run.

Check:

```python
import torch

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then point to the project:

```python
from pathlib import Path
import os

PROJECT_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/Smart waste scanner"
)

if not PROJECT_DIR.is_dir():
    raise FileNotFoundError(PROJECT_DIR)

os.chdir(PROJECT_DIR)
print("Project:", Path.cwd())
```
Install dependencies

```python
!pip install -q -r requirements.txt
```
Check the dataset

Make sure this file exists:

```text
data/dataset/classifier_class/dataset.zip
```

Then run:

```python
!python training/inspect_dataset.py
```

In Colab, on the first run the project will extract the ZIP into:

```text
/content/smart_waste_scanner_runtime/classifier_dataset_extracted
```

You can check the paths the code is using with:

```python
from training.dataset_utils import DEFAULT_DATASET_SOURCE, DEFAULT_EXTRACT_DIR

print("Dataset ZIP:", DEFAULT_DATASET_SOURCE)
print("Extract to :", DEFAULT_EXTRACT_DIR)
```

Expected Colab output:

```text
Dataset ZIP: .../Smart waste scanner/data/dataset/classifier_class/dataset.zip
Extract to : /content/smart_waste_scanner_runtime/classifier_dataset_extracted
```

Train

Example with EfficientNet-B0:

```python
!python training/train.py \
    --arch efficientnet_b0 \
    --image-size 224 \
    --epochs 40 \
    --batch-size 32 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --label-smoothing 0.08 \
    --class-weighting sqrt \
    --patience 8 \
    --device cuda \
    --workers 2 \
    --amp
```

When `--output` is not provided, results are automatically saved according to the architecture:

```text
runs/efficientnet_b0/
runs/mobilenet_v3_large/
runs/resnet18/
```

The best deployment model is copied to:

```text
models/best_model.pt
```

If Colab is interrupted during training, you can resume, for example:

```python
!python training/train.py \
    --arch efficientnet_b0 \
    --image-size 224 \
    --epochs 40 \
    --batch-size 32 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --label-smoothing 0.08 \
    --class-weighting sqrt \
    --patience 8 \
    --device cuda \
    --workers 2 \
    --amp \
    --resume runs/efficientnet_b0/last_checkpoint.pt
```

Run the web app

If `models/best_model.pt` already exists, check the model first:

```python
from app.main import classifier

classifier.warmup()
print(classifier.status)
```

Expected state:

```text
state = ready
```

The project has two health-check endpoints:

```text
/api/health  -> the FastAPI process is alive
/api/ready   -> the model is loaded and ready for inference
```

`launcher.py --ngrok` only creates a public tunnel after `/api/ready` reports that the model is ready.

```python
import os
import getpass

os.environ["NGROK_AUTHTOKEN"] = getpass.getpass(
    "NGROK_AUTHTOKEN: "
)
```

Then:

```python
!python launcher.py --ngrok --port 8000
```

Or run in 1 cell:
```python
import os
import sys
import getpass
import subprocess
from pathlib import Path

PROJECT_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/Smart waste scanner"
)

PORT = 8000
STARTUP_TIMEOUT = 600
USE_NGROK = True

print("=" * 72)
print("SMART WASTE SCANNER - COLAB")
print("=" * 72)

from google.colab import drive

MY_DRIVE = Path("/content/drive/MyDrive")

def drive_ready():
    try:
        return MY_DRIVE.is_dir() and next(MY_DRIVE.iterdir(), None) is not None
    except Exception:
        return False

if not drive_ready():
    print("Connecting to Google Drive...")

    try:
        drive.flush_and_unmount()
    except Exception:
        pass

    drive.mount(
        "/content/drive",
        force_remount=True,
        timeout_ms=180000,
    )

if not drive_ready():
    raise RuntimeError("Could not access Google Drive.")

print("Google Drive: OK")

if not PROJECT_DIR.is_dir():
    raise FileNotFoundError(
        f"Project not found:\n{PROJECT_DIR}"
    )

os.chdir(PROJECT_DIR)

print("Project:", PROJECT_DIR)

required_project_files = [
    PROJECT_DIR / "launcher.py",
    PROJECT_DIR / "requirements.txt",
]

for path in required_project_files:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

print("\nChecking/installing dependencies...")

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--disable-pip-version-check",
        "-r",
        str(PROJECT_DIR / "requirements.txt"),
    ],
    check=True,
)

print("Dependencies: OK")

from dotenv import load_dotenv

load_dotenv(PROJECT_DIR / ".env")

import torch

print("\nPyTorch :", torch.__version__)
print("CUDA    :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))
else:
    print("GPU     : Not available - the app will run on CPU")

MODEL_DIR = PROJECT_DIR / "models"

required_models = [
    MODEL_DIR / "best_detector.pt",
    MODEL_DIR / "best_model.pt",
    MODEL_DIR / "ood_reference.npz",
]

print("\nChecking models:")

for model_path in required_models:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Required model is missing:\n{model_path}"
        )

    size_mb = model_path.stat().st_size / (1024 ** 2)

    print(
        f"OK  {model_path.name:<22}"
        f"{size_mb:8.2f} MB"
    )

if USE_NGROK and not os.environ.get("NGROK_AUTHTOKEN", "").strip():
    token = getpass.getpass("NGROK_AUTHTOKEN: ").strip()

    if not token:
        raise RuntimeError(
            "NGROK_AUTHTOKEN is required to create a public URL."
        )

    os.environ["NGROK_AUTHTOKEN"] = token

LAUNCHER = PROJECT_DIR / "launcher.py"

cmd = [
    sys.executable,
    "-u",
    str(LAUNCHER),
    "--port",
    str(PORT),
    "--replace-port",
    "--startup-timeout",
    str(STARTUP_TIMEOUT),
]

if USE_NGROK:
    cmd.append("--ngrok")

print("\n" + "=" * 72)
print("STARTING SMART WASTE SCANNER")
print("=" * 72)
print("Port :", PORT)
print("Ngrok:", USE_NGROK)
print("=" * 72 + "\n")

process = subprocess.Popen(
    cmd,
    cwd=str(PROJECT_DIR),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env={
        **os.environ,
        "PYTHONUNBUFFERED": "1",
    },
)

interrupted = False

try:
    for line in process.stdout:
        print(line, end="", flush=True)

except KeyboardInterrupt:
    interrupted = True

    print("\nStopping Smart Waste Scanner...")

    process.terminate()

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

finally:
    if process.stdout:
        process.stdout.close()

code = process.wait()

if not interrupted and code != 0:
    raise RuntimeError(
        f"Launcher exited with error code {code}"
    )
```

Synchronization between local and Colab

The local project and the project on Drive use the same source code. Environment differences are handled automatically or through `.env`:

```text
Local
- dataset.zip: data/dataset/classifier_class/dataset.zip
- extract:     data/.runtime/classifier_dataset_extracted
- model:       models/best_model.pt
- database:    data/waste_scanner.db
- collected:   data/collected

Google Colab
- dataset.zip: .../Drive/.../data/dataset/classifier_class/dataset.zip
- extract:     /content/smart_waste_scanner_runtime/classifier_dataset_extracted
- model:       models/best_model.pt on Drive
- database:    data/waste_scanner.db on Drive
- collected:   data/collected on Drive
```

When you modify the code, you only need to synchronize the source files between local and the project folder on Google Drive. There is no need to upload the dataset again or retrain the model if they have not changed.