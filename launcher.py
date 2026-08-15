from __future__ import annotations

import argparse
import getpass
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
EXAMPLE_ENV_PATH = ROOT_DIR / ".env.example"
load_dotenv(ENV_PATH)


def _set_env_value(lines: list[str], key: str, value: str) -> list[str]:
    prefix = f"{key}="
    replacement = f"{key}={value}"
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = replacement
            return lines
    if lines and lines[-1].strip():
        lines.append("")
    lines.append(replacement)
    return lines


def configure_ngrok() -> int:
    print("Cau hinh ngrok cho Waste Scanner AI")
    print("Authtoken duoc luu cuc bo trong .env; file nay da nam trong .gitignore.")
    token = getpass.getpass("Nhap NGROK_AUTHTOKEN: ").strip()
    if not token:
        print("Khong co token, huy cau hinh.")
        return 1

    domain = input(
        "NGROK_DOMAIN tuy chon (Enter de dung URL ngau nhien mien phi): "
    ).strip()
    basic_auth = getpass.getpass(
        "NGROK_BASIC_AUTH tuy chon username:password (Enter de bo qua): "
    ).strip()

    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    elif EXAMPLE_ENV_PATH.exists():
        lines = EXAMPLE_ENV_PATH.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    lines = _set_env_value(lines, "NGROK_AUTHTOKEN", token)
    lines = _set_env_value(lines, "NGROK_DOMAIN", domain)
    lines = _set_env_value(lines, "NGROK_BASIC_AUTH", basic_auth)
    ENV_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print("Da luu cau hinh vao .env.")
    return 0


def _popen_kwargs() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _listener_pids_windows(port: int) -> list[int]:
    if os.name != "nt":
        return []
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []

    pids: set[int] = set()
    suffix = f":{port}"
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if "LISTENING" not in line.upper():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        local_addr = parts[1]
        if not local_addr.endswith(suffix):
            continue
        try:
            pids.add(int(parts[-1]))
        except ValueError:
            continue
    return sorted(pids)


def _valid_port(value: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Port phai la mot so nguyen tu 1 den 65535.") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port phai nam trong khoang 1..65535.")
    return port


def _port_is_free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _port_owner_message(port: int) -> str:
    pids = _listener_pids_windows(port)
    if pids:
        joined = ", ".join(str(pid) for pid in pids)
        return f"Port {port} dang bi tien trinh PID {joined} chiem."
    return f"Port {port} dang bi mot tien trinh khac chiem."


def require_free_port(port: int) -> None:
    if _port_is_free(port):
        return
    message = _port_owner_message(port)
    if os.name == "nt":
        message += f" Chay 'start.bat clean' de dong tien trinh dang nghe port {port}."
    raise RuntimeError(message)


def kill_port_listener(port: int) -> int:
    if _port_is_free(port):
        print(f"Port {port} dang trong, khong can don.")
        return 0

    if os.name != "nt":
        print(
            f"Port {port} dang bi chiem. Tren macOS/Linux hay dung lsof/fuser de tim va dung tien trinh.",
            file=sys.stderr,
        )
        return 1

    pids = _listener_pids_windows(port)
    if not pids:
        print(_port_owner_message(port), file=sys.stderr)
        return 1

    failed = False
    for pid in pids:
        print(f"Dang dong PID {pid} tren port {port}...")
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        failed = failed or result.returncode != 0

    time.sleep(0.5)
    if not _port_is_free(port):
        print(f"Khong the giai phong port {port}.", file=sys.stderr)
        return 1
    print(f"Da giai phong port {port}.")
    return 1 if failed else 0


def start_server(port: int, reload_enabled: bool, launch_token: str) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    if reload_enabled:
        command.extend(["--reload", "--reload-dir", str(ROOT_DIR / "app")])
    child_env = os.environ.copy()
    child_env["WASTE_SCANNER_LAUNCH_TOKEN"] = launch_token
    return subprocess.Popen(command, cwd=ROOT_DIR, env=child_env, **_popen_kwargs())


def stop_process_tree(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass


def _decode_health_response(response) -> dict[str, object]:
    raw = response.read().decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Health endpoint khong tra ve JSON object.")
    return payload


def wait_for_server(
    port: int,
    expected_token: str | None,
    timeout_seconds: int = 90,
    process: subprocess.Popen[bytes] | None = None,
    *,
    require_ready: bool = False,
) -> dict[str, object]:
    endpoint = "/api/ready" if require_ready else "/api/health"
    url = f"http://127.0.0.1:{port}{endpoint}"
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        if process is not None:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    "FastAPI da dung truoc khi san sang "
                    f"(exit code {return_code}). Xem log Uvicorn phia tren de biet chi tiet."
                )

        payload: dict[str, object] | None = None
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                payload = _decode_health_response(response)
        except urllib.error.HTTPError as exc:
            # /api/ready intentionally returns 503 until the classifier is ready.
            # The HTTP server may already be alive, so inspect its JSON body.
            try:
                payload = _decode_health_response(exc)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as decode_exc:
                last_error = f"HTTP {exc.code}: {decode_exc}"
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            last_error = str(exc)

        if payload is not None:
            if payload.get("app") != "waste-scanner-ai":
                last_error = "Port dang tra ve mot server khac, khong phai Waste Scanner AI."
            else:
                actual_token = str(payload.get("launch_token", ""))
                token_matches = expected_token is None or actual_token == expected_token
                if not token_matches:
                    if actual_token:
                        last_error = "Port dang tra ve mot instance Waste Scanner khac."
                    else:
                        last_error = "Port dang tra ve mot server cu/khac khong co launch token."
                elif require_ready and not bool(payload.get("ready")):
                    classifier = payload.get("classifier")
                    classifier_state = (
                        str(classifier.get("state", "")) if isinstance(classifier, dict) else ""
                    )
                    classifier_error = (
                        str(classifier.get("error", "")) if isinstance(classifier, dict) else ""
                    )
                    if classifier_state in {"error", "retry_available"}:
                        detail = f": {classifier_error}" if classifier_error else ""
                        raise RuntimeError(f"Model AI khong san sang{detail}")
                    last_error = f"Model AI dang khoi dong (state={classifier_state or 'unknown'})."
                else:
                    return payload

        time.sleep(0.5)

    detail = f" ({last_error})" if last_error else ""
    if require_ready:
        raise RuntimeError(f"AI model khong san sang tai {url}{detail}.")
    if expected_token is None:
        raise RuntimeError(f"Khong tim thay Waste Scanner AI dang chay tai {url}{detail}.")
    raise RuntimeError(f"FastAPI moi khong san sang tai {url}{detail}.")


def build_tunnel(port: int):
    try:
        from pyngrok import conf, ngrok
    except ImportError as exc:
        raise RuntimeError(
            "Thieu pyngrok. Hay chay pip install -r requirements.txt truoc."
        ) from exc

    token = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Chua co NGROK_AUTHTOKEN. Chay 'python launcher.py --configure' "
            "hoac them NGROK_AUTHTOKEN vao file .env."
        )

    config = conf.PyngrokConfig(auth_token=token)
    options: dict[str, str] = {}

    domain = os.getenv("NGROK_DOMAIN", "").strip()
    if domain:
        options["domain"] = domain

    basic_auth = os.getenv("NGROK_BASIC_AUTH", "").strip()
    if basic_auth:
        options["auth"] = basic_auth

    try:
        tunnel = ngrok.connect(
            addr=str(port),
            proto="http",
            pyngrok_config=config,
            **options,
        )
    except Exception as exc:
        raise RuntimeError(f"Khong the tao ngrok tunnel: {exc}") from exc
    return ngrok, tunnel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launcher duy nhat cho Waste Scanner AI: local, dev va ngrok."
    )
    parser.add_argument(
        "--configure",
        action="store_true",
        help="Nhap va luu cau hinh ngrok vao .env, sau do thoat.",
    )
    parser.add_argument(
        "--ngrok",
        action="store_true",
        help="Chay FastAPI va cong khai qua ngrok HTTPS.",
    )
    parser.add_argument(
        "--port",
        type=_valid_port,
        default=os.getenv("PORT", "8000"),
        help="Cong FastAPI noi bo, 1..65535 (mac dinh: PORT trong .env hoac 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Bat auto-reload Uvicorn khi phat trien.",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Voi --ngrok: chi tao tunnel khi FastAPI da chay san.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Voi --ngrok: tu dong mo URL public tren trinh duyet may chu.",
    )
    parser.add_argument(
        "--kill-port",
        action="store_true",
        help="Dong tien trinh dang LISTEN tren --port (Windows) va thoat.",
    )
    return parser.parse_args()


def run_local(port: int, reload_enabled: bool) -> int:
    server: subprocess.Popen[bytes] | None = None

    def shutdown_handler(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, shutdown_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        require_free_port(port)
        launch_token = uuid.uuid4().hex
        mode = "development/reload" if reload_enabled else "normal"
        print(f"[Waste Scanner] Che do: {mode}")
        print(f"[Waste Scanner] http://127.0.0.1:{port}")
        print("Nhan Ctrl+C mot lan de dung toan bo server.")
        server = start_server(port, reload_enabled, launch_token)
        wait_for_server(port, launch_token, process=server)
        while server.poll() is None:
            time.sleep(0.5)
        return int(server.returncode or 0)
    except RuntimeError as exc:
        print(f"\nLOI: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nDang dung toan bo Waste Scanner AI...")
        return 0
    finally:
        stop_process_tree(server)
        print("Da dong server va cac tien trinh con.")


def run_ngrok(port: int, reload_enabled: bool, no_server: bool, open_browser: bool) -> int:
    server: subprocess.Popen[bytes] | None = None
    tunnel = None
    ngrok_client = None

    def shutdown_handler(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, shutdown_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        if not no_server:
            require_free_port(port)
            launch_token = uuid.uuid4().hex
            print(f"[Waste Scanner] Khoi dong FastAPI tai http://127.0.0.1:{port}")
            server = start_server(port, reload_enabled, launch_token)
            wait_for_server(port, launch_token, process=server, require_ready=True)
        else:
            print(f"[Waste Scanner] Kiem tra server co san tai http://127.0.0.1:{port}")
            wait_for_server(port, expected_token=None, timeout_seconds=90, require_ready=True)

        print("[ngrok] Dang tao HTTPS tunnel...")
        ngrok_client, tunnel = build_tunnel(port)
        public_url = tunnel.public_url

        print("\n" + "=" * 64)
        print("WASTE SCANNER AI DA ONLINE")
        print(f"Public URL : {public_url}")
        print(f"Local URL  : http://127.0.0.1:{port}")
        print("Inspector  : http://127.0.0.1:4040")
        print("Nhan Ctrl+C de dung server va tunnel.")
        print("=" * 64 + "\n")

        if open_browser:
            webbrowser.open(public_url)

        if server is not None:
            while server.poll() is None:
                time.sleep(1)
            return int(server.returncode or 0)

        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nDang dung Waste Scanner AI va ngrok...")
        return 0
    except RuntimeError as exc:
        print(f"\nLOI: {exc}", file=sys.stderr)
        return 1
    finally:
        if tunnel is not None:
            try:
                if ngrok_client is not None:
                    ngrok_client.disconnect(tunnel.public_url)
            except Exception:
                pass
        try:
            if ngrok_client is not None:
                ngrok_client.kill()
        except Exception:
            pass
        stop_process_tree(server)


def main() -> int:
    args = parse_args()
    if args.kill_port:
        return kill_port_listener(args.port)
    if args.configure:
        return configure_ngrok()
    if args.ngrok:
        return run_ngrok(args.port, args.reload, args.no_server, args.open)
    if args.no_server or args.open:
        print("--no-server va --open chi dung kem --ngrok.", file=sys.stderr)
        return 2
    return run_local(args.port, args.reload)


if __name__ == "__main__":
    raise SystemExit(main())
