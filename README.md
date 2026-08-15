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
import os
import sys
import time
import signal
import socket
import subprocess
from pathlib import Path

PROJECT_DIR = Path("/content/drive/MyDrive/Colab Notebooks/Smart waste scanner")

PORT = 8000
USE_NGROK = True
RELOAD = False
STARTUP_TIMEOUT = 600

LAUNCHER_BUILD = "2026-08-15-status-v3-port-fix"

print("=" * 72)
print("SMART WASTE SCANNER - COLAB")
print(f"PROJECT: {PROJECT_DIR}")
print("=" * 72)
print(f"Launcher build: {LAUNCHER_BUILD}")

if not Path("/content/drive/MyDrive").exists():
    from google.colab import drive
    drive.mount("/content/drive")
else:
    print("Google Drive da duoc mount.")

if not PROJECT_DIR.exists():
    raise FileNotFoundError(f"Khong tim thay project:\n{PROJECT_DIR}")

os.chdir(PROJECT_DIR)

print(f"Working directory: {os.getcwd()}")

def port_is_busy(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)

    try:
        result = sock.connect_ex(("127.0.0.1", port))
        return result == 0
    finally:
        sock.close()

def get_port_pids(port: int):
    pids = set()

    try:
        result = subprocess.run(
            ["bash", "-lc", f"lsof -ti:{port} 2>/dev/null || true"],
            capture_output=True,
            text=True,
            timeout=10
        )

        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
    except Exception:
        pass

    if not pids:
        try:
            result = subprocess.run(
                ["bash", "-lc", f"fuser {port}/tcp 2>/dev/null || true"],
                capture_output=True,
                text=True,
                timeout=10
            )

            for token in result.stdout.replace(":", " ").split():
                token = token.strip()
                if token.isdigit():
                    pids.add(int(token))
        except Exception:
            pass

    return sorted(pids)

def get_process_info(pid: int):
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid=,ppid=,cmd="],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return ""

def wait_port_free(port: int, timeout: float = 10):
    deadline = time.time() + timeout

    while time.time() < deadline:
        if not port_is_busy(port):
            return True
        time.sleep(0.25)

    return not port_is_busy(port)

def kill_port_processes(port: int):
    if not port_is_busy(port):
        print(f"Port {port} dang trong.")
        return True

    print("=" * 72)
    print(f"PORT {port} DANG BI CHIEM")
    print("=" * 72)

    pids = get_port_pids(port)

    if not pids:
        subprocess.run(
            ["bash", "-lc", f"fuser -k {port}/tcp 2>/dev/null || true"],
            capture_output=True,
            text=True
        )

        time.sleep(2)
        return not port_is_busy(port)

    print(f"Tim thay PID: {pids}")

    for pid in pids:
        info = get_process_info(pid)
        if info:
            print(info)

    for pid in pids:
        if pid == os.getpid():
            continue

        try:
            os.kill(pid, signal.SIGTERM)
            print(f"SIGTERM PID {pid}")
        except ProcessLookupError:
            pass
        except PermissionError:
            pass

    if wait_port_free(port, timeout=5):
        print(f"Port {port} da duoc giai phong.")
        return True

    remaining = get_port_pids(port)

    for pid in remaining:
        if pid == os.getpid():
            continue

        try:
            os.kill(pid, signal.SIGKILL)
            print(f"SIGKILL PID {pid}")
        except ProcessLookupError:
            pass
        except PermissionError:
            pass

    return wait_port_free(port, timeout=5)

print()
print("=" * 72)
print("CHECK PORT")
print("=" * 72)

if not kill_port_processes(PORT):
    subprocess.run(
        ["bash", "-lc", f"lsof -i:{PORT} || true"]
    )
    raise RuntimeError(f"Port {PORT} van dang bi chiem.")

print()
print("=" * 72)
print("CHECK OLD NGROK")
print("=" * 72)

try:
    result = subprocess.run(
        ["pgrep", "-f", "ngrok"],
        capture_output=True,
        text=True
    )

    ngrok_pids = []

    for line in result.stdout.splitlines():
        line = line.strip()

        if line.isdigit():
            pid = int(line)
            if pid != os.getpid():
                ngrok_pids.append(pid)

    if ngrok_pids:
        for pid in ngrok_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass

        time.sleep(1)
        print("Da dung ngrok cu.")
    else:
        print("Khong co ngrok cu.")

except Exception as e:
    print(f"Khong kiem tra duoc ngrok: {e}")

possible_launchers = [
    "launcher.py",
    "run.py",
    "app.py",
    "main.py",
]

existing = [
    f for f in possible_launchers
    if (PROJECT_DIR / f).exists()
]

print()
print("=" * 72)
print("CHECK PATHS")
print("=" * 72)

if existing:
    print("Python entry files:", ", ".join(existing))

LAUNCHER_FILE = PROJECT_DIR / "launcher.py"

if not LAUNCHER_FILE.exists():
    fallback_candidates = [
        PROJECT_DIR / "run.py",
        PROJECT_DIR / "app.py",
        PROJECT_DIR / "main.py",
    ]

    LAUNCHER_FILE = next(
        (p for p in fallback_candidates if p.exists()),
        None
    )

if LAUNCHER_FILE is None:
    raise FileNotFoundError(
        "Khong tim thay launcher.py, run.py, app.py hoac main.py."
    )

print(f"Launcher file: {LAUNCHER_FILE}")

cmd = [
    sys.executable,
    "-u",
    str(LAUNCHER_FILE),
]

try:
    help_result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_FILE),
            "--help"
        ],
        capture_output=True,
        text=True,
        timeout=20
    )

    help_text = help_result.stdout + "\n" + help_result.stderr
except Exception:
    help_text = ""

if "--port" in help_text:
    cmd += ["--port", str(PORT)]

if USE_NGROK and "--ngrok" in help_text:
    cmd += ["--ngrok"]

if RELOAD and "--reload" in help_text:
    cmd += ["--reload"]

if "--timeout" in help_text:
    cmd += ["--timeout", str(STARTUP_TIMEOUT)]

print()
print("=" * 72)
print("START SMART WASTE SCANNER")
print("=" * 72)
print(f"WASTE SCANNER LAUNCHER | build={LAUNCHER_BUILD}")
print(f"Python     : {sys.executable}")
print(f"Project    : {PROJECT_DIR}")
print("Colab      : True")
print(f"Port       : {PORT}")
print(f"Ngrok      : {USE_NGROK}")
print(f"Reload     : {RELOAD}")
print("=" * 72)
print()
print("Command:")
print(" ".join(map(str, cmd)))
print()

process = subprocess.Popen(
    cmd,
    cwd=str(PROJECT_DIR),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    universal_newlines=True,
    env={
        **os.environ,
        "PYTHONUNBUFFERED": "1",
    }
)

try:
    for line in iter(process.stdout.readline, ""):
        if not line and process.poll() is not None:
            break

        if line:
            print(line, end="", flush=True)

except KeyboardInterrupt:
    try:
        process.terminate()
        process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass

    raise

finally:
    if process.stdout:
        process.stdout.close()

return_code = process.poll()

print()
print("=" * 72)

if return_code == 0:
    print("Smart Waste Scanner launcher da ket thuc binh thuong.")
elif return_code is None:
    print("Smart Waste Scanner dang chay.")
else:
    print(f"Launcher ket thuc voi code {return_code}")

print("=" * 72)

if return_code not in (0, None):
    raise RuntimeError(
        f"Launcher ket thuc voi code {return_code}"
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