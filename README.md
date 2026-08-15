# Smart Waste Scanner AI

Waste classification application using a supervised PyTorch model with the fixed 11-class schema of the SmartWaste dataset.

## 1. Place the dataset

No manual extraction is required. Place the ZIP file in the correct location with the exact filename:

```text
Smart waste scanner/
├── data/
│   └── dataset/
│       └── dataset.zip
├── training/
├── models/
├── train_model.bat
└── ...
```

`dataset.zip` can contain `train/`, `val/`, and `test/` directly, or it can have a root directory wrapping them, such as `SmartWaste_Household_EWaste_11class_native_v2`.

On the first run, the code automatically extracts the dataset. The cache location depends on the environment:

```text
Local : data/dataset/_extracted/
Colab : /content/smart_waste_scanner_dataset/
```

On Colab, the extracted files stay on the runtime SSD, so Google Drive only needs to keep `dataset.zip`. The cache is reused within the same runtime. If the Colab runtime resets, `/content` is cleared and the archive is extracted again. Replacing `dataset.zip` also causes the cache to be rebuilt automatically.

## 2. 11-class schema

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

## 3. Train on Windows

The simplest way is to double-click or run:

```bat
train_model.bat
```

The script will automatically:

1. find `data\dataset\dataset.zip`;
2. create `.venv` if it does not exist;
3. install training dependencies;
4. extract the dataset if needed;
5. verify that all 11 classes are present;
6. train EfficientNet-B0;
7. save the best model to `models\best_model.pt`.

You can still provide a different dataset if desired:

```bat
train_model.bat "D:\duong-dan\dataset-khac.zip"
```

## 4. Train with Python

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
python training\train.py --arch efficientnet_b0 --epochs 30 --batch-size 32 --device auto
```

There is no need to pass `--data`; the default is `data\dataset\dataset.zip`.

If VRAM is insufficient, reduce `--batch-size` to 16 or 8. Supported architectures: `efficientnet_b0`, `mobilenet_v3_large`, `resnet18`.

When `--output` is omitted, training results are stored in `runs/<arch>/` (for example `runs/efficientnet_b0/`, `runs/mobilenet_v3_large/`, or `runs/resnet18/`). The best checkpoint is automatically copied to:

```text
models/best_model.pt
```

Resume:

```bat
python training\train.py ^
  --epochs 40 ^
  --resume runs\efficientnet_b0\last_checkpoint.pt
```

## 5. Run the application

After `models\best_model.pt` is available:

```bat
start.bat
```

By default, the app runs at `http://localhost:8000`.

If `models/best_model.pt` is not available yet, the web app will still open, but the classification API will report that the model is not ready.

## 6. Configuration

You can copy `.env.example` to `.env`. The main variables are:

```env
WASTE_MODEL_CHECKPOINT=models/best_model.pt
WASTE_DEVICE=auto
UNKNOWN_THRESHOLD=0.45
UNCERTAINTY_MARGIN=0.10
DATABASE_PATH=data/waste_scanner.db
```

`UNKNOWN_THRESHOLD` and `UNCERTAINTY_MARGIN` should be tuned using the validation/test results of the actual model.

## 7. Feedback learning

When a user confirms or corrects a label, the app stores the L2-normalized feature vector (feature embedding), taken immediately before the classifier head, as feedback memory. Feedback is namespaced by checkpoint hash, so data from different models is not mixed together.

## 8. Docker

```bash
docker compose --env-file .env up --build waste-scanner
```

Docker Compose mounts `./models` into the container. Train the model first and place `best_model.pt` in `models/`.

## 9. ngrok (optional)

```bat
start.bat configure
start.bat ngrok
```

`pyngrok` is included in the shared `requirements.txt`, so installing the project requirements once is enough. When `launcher.py --ngrok` is used, the launcher waits for the AI readiness endpoint before creating the public tunnel.

## 10. Collecting real-world data

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

## 11. Run on Google Colab

Place the entire project folder in Google Drive, for example:

```text
My Drive/
└── Colab Notebooks/
    └── Smart waste scanner/
        ├── app/
        ├── data/
        │   ├── dataset/
        │   │   └── dataset.zip
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
/content/smart_waste_scanner_dataset
```

as the extraction cache. Because `/content` is temporary storage for the Colab runtime, this cache is lost when the runtime resets and is automatically recreated from `data/dataset/dataset.zip` on the next run.

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
data/dataset/dataset.zip
```

Then run:

```python
!python training/inspect_dataset.py
```

In Colab, on the first run the project will extract the ZIP into:

```text
/content/smart_waste_scanner_dataset
```

You can check the paths the code is using with:

```python
from training.dataset_utils import DEFAULT_DATASET_SOURCE, DEFAULT_EXTRACT_DIR

print("Dataset ZIP:", DEFAULT_DATASET_SOURCE)
print("Extract to :", DEFAULT_EXTRACT_DIR)
```

Expected Colab output:

```text
Dataset ZIP: .../Smart waste scanner/data/dataset/dataset.zip
Extract to : /content/smart_waste_scanner_dataset
```

Train

Example with EfficientNet-B0:

```python
!python training/train.py \
    --arch efficientnet_b0 \
    --epochs 30 \
    --batch-size 32 \
    --lr 3e-4 \
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

If Colab is interrupted during training, you can resume, for example:

```python
!python training/train.py \
    --arch efficientnet_b0 \
    --epochs 30 \
    --batch-size 32 \
    --lr 3e-4 \
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
```
# ================================================================
# SMART WASTE SCANNER
# ================================================================

from google.colab import drive
from pathlib import Path
import os
import sys
import subprocess
import hashlib

drive.mount("/content/drive")

PROJECT_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/Smart waste scanner"
).resolve()

if not PROJECT_DIR.is_dir():
    raise FileNotFoundError(
        f"Không tìm thấy project: {PROJECT_DIR}"
    )

os.chdir(PROJECT_DIR)

print("=" * 72, flush=True)
print("SMART WASTE SCANNER - COLAB", flush=True)
print("PROJECT:", PROJECT_DIR, flush=True)
print("=" * 72, flush=True)

launcher = PROJECT_DIR / "launcher.py"

launcher_text = launcher.read_text(
    encoding="utf-8",
    errors="ignore",
)

EXPECTED = 'LAUNCHER_BUILD = "2026-08-15-status-v2"'

if EXPECTED not in launcher_text:
    raise RuntimeError(
        "\n❌ launcher.py chưa phải bản status-v2.\n"
        "Hãy dán đè launcher.py từ patch mới nhất rồi chạy lại."
    )

print("✅ Launcher build: 2026-08-15-status-v2", flush=True)

required = [
    PROJECT_DIR / "requirements.txt",
    PROJECT_DIR / ".env",
    PROJECT_DIR / "models" / "best_model.pt",
    PROJECT_DIR / "data" / "dataset" / "dataset.zip",
    PROJECT_DIR / "training" / "check_paths.py",
]

missing = [p for p in required if not p.exists()]

if missing:
    print("\n❌ Thiếu file:")
    for p in missing:
        print(" -", p)
    raise RuntimeError("Project chưa đầy đủ.")

print("✅ Các file bắt buộc đã có.", flush=True)

requirements = PROJECT_DIR / "requirements.txt"

req_hash = hashlib.sha256(
    requirements.read_bytes()
).hexdigest()

marker = Path("/content/.smartwaste_requirements")

need_install = True

if marker.exists():
    if marker.read_text().strip() == req_hash:
        need_install = False

if need_install:
    print("\n📦 Đang cài dependencies...", flush=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            "requirements.txt",
        ],
        cwd=str(PROJECT_DIR),
        check=True,
    )

    marker.write_text(req_hash)

    print("✅ Dependencies OK.", flush=True)

else:
    print("✅ Dependencies đã có, bỏ qua pip install.", flush=True)

print("\n" + "=" * 72, flush=True)
print("CHECK PATHS", flush=True)
print("=" * 72, flush=True)

result = subprocess.run(
    [
        sys.executable,
        "-u",
        "training/check_paths.py",
    ],
    cwd=str(PROJECT_DIR),
)

if result.returncode != 0:
    raise RuntimeError(
        "❌ Path configuration chưa đúng."
    )

print("✅ PATHS OK", flush=True)

print("\n" + "=" * 72, flush=True)
print("START SMART WASTE SCANNER", flush=True)
print("=" * 72, flush=True)

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

process = subprocess.Popen(
    [
        sys.executable,
        "-u",
        "launcher.py",
        "--ngrok",
        "--port",
        "8000",
    ],
    cwd=str(PROJECT_DIR),
    env=env,

    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,

    text=True,
    bufsize=1,
)

try:
    while True:

        line = process.stdout.readline()

        if line:
            print(line, end="", flush=True)

        if process.poll() is not None:
            # In nốt output còn lại
            remaining = process.stdout.read()

            if remaining:
                print(remaining, end="", flush=True)

            break

except KeyboardInterrupt:

    print(
        "\n\n⏹ Đang dừng Waste Scanner...",
        flush=True,
    )

    process.terminate()

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()

    print("✅ Đã dừng.", flush=True)

    raise

if process.returncode not in (0, None):
    raise RuntimeError(
        f"Launcher kết thúc với code {process.returncode}"
    )
```

Synchronization between local and Colab

The local project and the project on Drive use the same source code. Environment differences are handled automatically or through `.env`:

```text
Local
- dataset.zip: data/dataset/dataset.zip
- extract:     data/dataset/_extracted
- model:       models/best_model.pt
- database:    data/waste_scanner.db
- collected:   data/collected

Google Colab
- dataset.zip: .../Drive/.../data/dataset/dataset.zip
- extract:     /content/smart_waste_scanner_dataset
- model:       models/best_model.pt on Drive
- database:    data/waste_scanner.db on Drive
- collected:   data/collected on Drive
```

When you modify the code, you only need to synchronize the source files between local and the project folder on Google Drive. There is no need to upload the dataset again or retrain the model if they have not changed.