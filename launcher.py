from __future__ import annotations

import argparse
import contextlib
import io
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

LAUNCHER_BUILD = "2026-08-15-status-v4-clean-vi"
ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
EXAMPLE_ENV_PATH = ROOT_DIR / ".env.example"
load_dotenv(ENV_PATH)


_LAUNCH_STARTED_AT = time.monotonic()


def _running_in_colab() -> bool:
    return bool(
        os.getenv("COLAB_RELEASE_TAG")
        or os.getenv("COLAB_GPU")
        or os.getenv("COLAB_BACKEND_VERSION")
    )


def _log(message: str, *, prefix: str = "Waste Scanner") -> None:
    elapsed = time.monotonic() - _LAUNCH_STARTED_AT
    print(f"[{prefix}] +{elapsed:6.1f}s | {message}", flush=True)


def _startup_timeout_default() -> int:
    raw = os.getenv("STARTUP_TIMEOUT_SECONDS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value >= 30:
            return value
    # A fresh Colab session may need to read the checkpoint from Drive and,
    # when the persistent OOD bank is missing/stale, build it once from the
    # extracted dataset. Give that first startup enough time.
    return 600 if _running_in_colab() else 180


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
    print("Cấu hình ngrok cho Waste Scanner AI")
    print("Authtoken được lưu cục bộ trong .env; file này đã nằm trong .gitignore.")
    token = getpass.getpass("Nhập NGROK_AUTHTOKEN: ").strip()
    if not token:
        print("Không có token, hủy cấu hình.")
        return 1

    domain = input(
        "NGROK_DOMAIN tùy chọn (Enter để dùng URL ngẫu nhiên miễn phí): "
    ).strip()
    basic_auth = getpass.getpass(
        "NGROK_BASIC_AUTH tùy chọn username:password (Enter để bỏ qua): "
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
    print("Đã lưu cấu hình vào .env.")
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
            pid = int(parts[-1])
        except ValueError:
            continue
        if pid != os.getpid():
            pids.add(pid)
    return sorted(pids)


def _listener_pids_unix(port: int) -> list[int]:
    if os.name == "nt":
        return []

    pids: set[int] = set()

    # lsof is available by default on macOS and on many Linux systems.
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        for token in result.stdout.split():
            if token.isdigit():
                pid = int(token)
                if pid != os.getpid():
                    pids.add(pid)
    except (OSError, subprocess.TimeoutExpired):
        pass

    # fuser is normally available in Google Colab/Linux and is a useful fallback.
    if not pids:
        try:
            result = subprocess.run(
                ["fuser", "-n", "tcp", str(port)],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            output = f"{result.stdout} {result.stderr}"
            for token in output.replace(":", " ").split():
                if token.isdigit():
                    pid = int(token)
                    if pid != os.getpid():
                        pids.add(pid)
        except (OSError, subprocess.TimeoutExpired):
            pass

    return sorted(pids)


def _listener_pids(port: int) -> list[int]:
    return _listener_pids_windows(port) if os.name == "nt" else _listener_pids_unix(port)


def _valid_port(value: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Port phải là một số nguyên từ 1 đến 65535.") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port phải nằm trong khoảng 1..65535.")
    return port


def _port_is_free(port: int) -> bool:
    """Return True when a new Uvicorn listener can bind this port.

    SO_REUSEADDR is intentionally enabled. Without it, Linux/Colab can report
    EADDRINUSE for a recently closed server even when no process is listening,
    because old TCP connections may still be in TIME_WAIT.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            # Windows has different SO_REUSEADDR semantics; exclusive bind is the
            # reliable check for whether Uvicorn can own this listener.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            # Linux/macOS/Colab: allow immediate reuse after a previous server
            # closed while old client connections are still in TIME_WAIT.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _wait_port_free(port: int, timeout_seconds: float = 6.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _port_is_free(port):
            return True
        time.sleep(0.2)
    return _port_is_free(port)


def _port_owner_message(port: int) -> str:
    pids = _listener_pids(port)
    if pids:
        joined = ", ".join(str(pid) for pid in pids)
        return f"Port {port} đang bị tiến trình PID {joined} chiếm."
    return f"Port {port} chưa bind được, nhưng không xác định được PID listener."


def require_free_port(port: int) -> None:
    if _port_is_free(port):
        return
    message = _port_owner_message(port)
    if os.name == "nt":
        message += f" Chạy 'start.bat clean' hoặc dùng --replace-port để đóng instance cũ."
    else:
        message += " Dùng --replace-port để đóng instance cũ và khởi động lại."
    raise RuntimeError(message)


def kill_port_listener(port: int) -> int:
    """Kill processes listening on *port* on Windows, Linux/Colab, or macOS."""
    if _port_is_free(port):
        print(f"Port {port} đang trống, không cần dọn.")
        return 0

    pids = _listener_pids(port)
    if not pids:
        # On Linux/Colab, fuser can still kill the listener even when PID parsing
        # failed. This is intentionally a fallback and is never used on Windows.
        if os.name != "nt":
            try:
                subprocess.run(
                    ["fuser", "-k", "-n", "tcp", str(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
            if _wait_port_free(port, timeout_seconds=3.0):
                print(f"Đã giải phóng port {port}.")
                return 0
        print(_port_owner_message(port), file=sys.stderr)
        return 1

    print(f"Tìm thấy listener trên port {port}: PID {', '.join(map(str, pids))}")

    if os.name == "nt":
        for pid in pids:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode != 0:
                print(f"Không đóng được PID {pid}.", file=sys.stderr)
        if not _wait_port_free(port):
            print(f"Không thể giải phóng port {port}.", file=sys.stderr)
            return 1
        print(f"Đã giải phóng port {port}.")
        return 0

    # POSIX: terminate politely first so Uvicorn/ngrok related cleanup can run.
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"SIGTERM PID {pid}")
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"Không có quyền dừng PID {pid}.", file=sys.stderr)

    if _wait_port_free(port, timeout_seconds=3.0):
        print(f"Đã giải phóng port {port}.")
        return 0

    remaining = _listener_pids(port)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"SIGKILL PID {pid}")
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"Không có quyền kill PID {pid}.", file=sys.stderr)

    if not _wait_port_free(port, timeout_seconds=3.0):
        print(f"Không thể giải phóng port {port}.", file=sys.stderr)
        return 1

    print(f"Đã giải phóng port {port}.")
    return 0


def prepare_port(port: int, *, replace_existing: bool) -> None:
    if _port_is_free(port):
        return
    if not replace_existing:
        require_free_port(port)
    _log(f"Port {port} đang được sử dụng; đang đóng instance cũ...")
    if kill_port_listener(port) != 0:
        raise RuntimeError(f"Không thể giải phóng port {port}.")


def start_server(port: int, reload_enabled: bool, launch_token: str) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--log-level",
        "info",
    ]
    if reload_enabled:
        command.extend(["--reload", "--reload-dir", str(ROOT_DIR / "app")])
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["WASTE_SCANNER_LAUNCH_TOKEN"] = launch_token
    process = subprocess.Popen(command, cwd=ROOT_DIR, env=child_env, **_popen_kwargs())
    _log(f"Đã tạo tiến trình Uvicorn (PID={process.pid}, port={port}).")
    return process


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
        raise ValueError("Health endpoint không trả về JSON object.")
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
    started_at = time.monotonic()
    deadline = started_at + timeout_seconds
    last_error = ""
    last_state = ""
    last_progress_log = 0.0
    http_seen = False
    waiting_fastapi_logged = False

    target = "mô hình AI" if require_ready else "FastAPI"
    _log(f"Đang đợi {target} sẵn sàng...")

    while time.monotonic() < deadline:
        if process is not None:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    "FastAPI đã dừng trước khi sẵn sàng "
                    f"(exit code {return_code}). Xem log Uvicorn phía trên để biết chi tiết."
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

        now = time.monotonic()
        elapsed = now - started_at

        if payload is not None:
            if not http_seen:
                http_seen = True
                _log(f"FastAPI đã phản hồi tại http://127.0.0.1:{port}.")

            if payload.get("app") != "waste-scanner-ai":
                last_error = "Port đang trả về một server khác, không phải Waste Scanner AI."
            else:
                actual_token = str(payload.get("launch_token", ""))
                token_matches = expected_token is None or actual_token == expected_token
                if not token_matches:
                    if actual_token:
                        last_error = "Port đang trả về một instance Waste Scanner khác."
                    else:
                        last_error = "Port đang trả về một server cũ/khác không có launch token."
                else:
                    classifier = payload.get("classifier")
                    classifier_dict = classifier if isinstance(classifier, dict) else {}
                    classifier_state = str(classifier_dict.get("state", ""))
                    classifier_error = str(classifier_dict.get("error", ""))

                    if classifier_state != last_state:
                        last_state = classifier_state
                        if classifier_state == "not_loaded":
                            _log("FastAPI đã online; đang chờ tác vụ nạp AI bắt đầu...")
                        elif classifier_state == "loading":
                            checkpoint = classifier_dict.get("checkpoint")
                            ood_ref = classifier_dict.get("ood_reference")
                            _log("AI đang nạp checkpoint/OOD reference...")
                            if checkpoint:
                                _log(f"Checkpoint: {checkpoint}")
                            if ood_ref:
                                _log(f"OOD reference: {ood_ref}")
                        elif classifier_state == "ready":
                            architecture = classifier_dict.get("architecture") or "unknown"
                            device = classifier_dict.get("device") or "unknown"
                            image_size = classifier_dict.get("image_size") or "?"
                            _log(
                                f"AI READY | arch={architecture} | device={device} | image_size={image_size}"
                            )
                        elif classifier_state in {"error", "retry_available"}:
                            detail = f": {classifier_error}" if classifier_error else ""
                            _log(f"AI nạp lỗi ({classifier_state}){detail}", prefix="ERROR")

                    if require_ready and not bool(payload.get("ready")):
                        if classifier_state in {"error", "retry_available"}:
                            detail = f": {classifier_error}" if classifier_error else ""
                            raise RuntimeError(f"Mô hình AI không sẵn sàng{detail}")
                        last_error = f"Mô hình AI đang khởi động (state={classifier_state or 'unknown'})."

                        # Keep Colab visibly alive during long Drive/OOD work.
                        if now - last_progress_log >= 10.0:
                            remaining = max(0, int(deadline - now))
                            _log(
                                f"Vẫn đang khởi động AI... {elapsed:.0f}s đã trôi qua, "
                                f"còn tối đa {remaining}s."
                            )
                            last_progress_log = now
                    else:
                        return payload

        elif not waiting_fastapi_logged:
            _log("Đang chờ FastAPI phản hồi...")
            waiting_fastapi_logged = True

        time.sleep(0.5)

    detail = f" ({last_error})" if last_error else ""
    if require_ready:
        raise RuntimeError(f"Mô hình AI không sẵn sàng tại {url}{detail}.")
    if expected_token is None:
        raise RuntimeError(f"Không tìm thấy Waste Scanner AI đang chạy tại {url}{detail}.")
    raise RuntimeError(f"FastAPI mới không sẵn sàng tại {url}{detail}.")


def build_tunnel(port: int):
    try:
        from pyngrok import conf, ngrok
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu pyngrok. Hãy cài dependencies trước."
        ) from exc

    token = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Chưa có NGROK_AUTHTOKEN. Chạy 'python launcher.py --configure' "
            "hoặc thêm NGROK_AUTHTOKEN vào file .env."
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
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            tunnel = ngrok.connect(
                addr=str(port),
                proto="http",
                pyngrok_config=config,
                **options,
            )
    except Exception as exc:
        raise RuntimeError(f"Không thể tạo ngrok tunnel: {exc}") from exc
    return ngrok, tunnel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launcher duy nhất cho Waste Scanner AI: local, dev và ngrok."
    )
    parser.add_argument(
        "--configure",
        action="store_true",
        help="Nhập và lưu cấu hình ngrok vào .env, sau đó thoát.",
    )
    parser.add_argument(
        "--ngrok",
        action="store_true",
        help="Chạy FastAPI và công khai qua ngrok HTTPS.",
    )
    parser.add_argument(
        "--port",
        type=_valid_port,
        default=os.getenv("PORT", "8000"),
        help="Cổng FastAPI nội bộ, 1..65535 (mặc định: PORT trong .env hoặc 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Bật auto-reload Uvicorn khi phát triển.",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Với --ngrok: chỉ tạo tunnel khi FastAPI đã chạy sẵn.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Với --ngrok: tự động mở URL public trên trình duyệt máy chủ.",
    )
    parser.add_argument(
        "--kill-port",
        action="store_true",
        help="Đóng tiến trình đang LISTEN trên --port (Windows/Linux/macOS) và thoát.",
    )
    parser.add_argument(
        "--replace-port",
        action="store_true",
        help=(
            "Nếu --port đang có listener, đóng instance cũ rồi khởi động server mới. "
            "Hữu ích khi restart trên Colab/local dev."
        ),
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=_startup_timeout_default(),
        help=(
            "Số giây tối đa đợi mô hình AI sẵn sàng. "
            "Mặc định 600s trên Colab, 180s ở local, hoặc STARTUP_TIMEOUT_SECONDS trong .env."
        ),
    )
    args = parser.parse_args()
    if args.startup_timeout < 30:
        parser.error("--startup-timeout phải >= 30 giây.")
    return args


def run_local(port: int, reload_enabled: bool, replace_port: bool) -> int:
    server: subprocess.Popen[bytes] | None = None

    def shutdown_handler(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, shutdown_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        prepare_port(port, replace_existing=replace_port)
        launch_token = uuid.uuid4().hex
        mode = "development/reload" if reload_enabled else "normal"
        print(f"[Waste Scanner] Chế độ: {mode}")
        print(f"[Waste Scanner] http://127.0.0.1:{port}")
        print("Nhấn Ctrl+C một lần để dừng toàn bộ server.")
        server = start_server(port, reload_enabled, launch_token)
        wait_for_server(port, launch_token, process=server)
        while server.poll() is None:
            time.sleep(0.5)
        return int(server.returncode or 0)
    except RuntimeError as exc:
        print(f"\nLỖI: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nĐang dừng toàn bộ Waste Scanner AI...")
        return 0
    finally:
        stop_process_tree(server)
        print("Đã đóng server và các tiến trình con.")


def run_ngrok(
    port: int,
    reload_enabled: bool,
    no_server: bool,
    open_browser: bool,
    startup_timeout: int,
    replace_port: bool,
) -> int:
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
            prepare_port(port, replace_existing=replace_port)
            launch_token = uuid.uuid4().hex
            _log(f"Khởi động FastAPI tại http://127.0.0.1:{port}")
            server = start_server(port, reload_enabled, launch_token)
            wait_for_server(
                port,
                launch_token,
                timeout_seconds=startup_timeout,
                process=server,
                require_ready=True,
            )
        else:
            _log(f"Kiểm tra server có sẵn tại http://127.0.0.1:{port}")
            wait_for_server(
                port,
                expected_token=None,
                timeout_seconds=startup_timeout,
                require_ready=True,
            )

        _log("AI sẵn sàng. Đang tạo HTTPS tunnel...", prefix="ngrok")
        ngrok_client, tunnel = build_tunnel(port)
        public_url = tunnel.public_url

        print("\n" + "=" * 72, flush=True)
        print("WASTE SCANNER AI ĐÃ ONLINE", flush=True)
        print(f"PUBLIC URL : {public_url}", flush=True)
        print(f"LOCAL URL  : http://127.0.0.1:{port}", flush=True)
        print("INSPECTOR  : http://127.0.0.1:4040", flush=True)
        print("STATUS     : FastAPI + AI model + ngrok READY", flush=True)
        print("Nhấn Stop cell / Ctrl+C để dừng server và tunnel.", flush=True)
        print("=" * 72 + "\n", flush=True)

        if open_browser:
            webbrowser.open(public_url)

        if server is not None:
            while server.poll() is None:
                time.sleep(1)
            return int(server.returncode or 0)

        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nĐang dừng Waste Scanner AI và ngrok...")
        return 0
    except RuntimeError as exc:
        print(f"\nLỖI: {exc}", file=sys.stderr)
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
    # Force immediate notebook-visible output before any startup checks.
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except (AttributeError, ValueError):
        pass
    print("=" * 72, flush=True)
    print(f"WASTE SCANNER LAUNCHER | build={LAUNCHER_BUILD}", flush=True)
    print(f"Python     : {sys.executable}", flush=True)
    print(f"Project    : {ROOT_DIR}", flush=True)
    print(f"Colab      : {_running_in_colab()}", flush=True)
    print("=" * 72, flush=True)
    _log("Launcher đã bắt đầu. Đang đọc tham số khởi động...")
    args = parse_args()
    _log(
        f"Tham số: ngrok={args.ngrok}, port={args.port}, reload={args.reload}, "
        f"replace_port={args.replace_port}, timeout={args.startup_timeout}s"
    )
    if args.kill_port:
        return kill_port_listener(args.port)
    if args.configure:
        return configure_ngrok()
    if args.ngrok:
        return run_ngrok(
            args.port,
            args.reload,
            args.no_server,
            args.open,
            args.startup_timeout,
            args.replace_port,
        )
    if args.no_server or args.open:
        print("--no-server và --open chỉ dùng kèm --ngrok.", file=sys.stderr)
        return 2
    return run_local(args.port, args.reload, args.replace_port)


if __name__ == "__main__":
    raise SystemExit(main())
