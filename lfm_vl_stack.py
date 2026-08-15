#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx==0.28.1",
#   "mypy==2.3.1",
#   "pydantic==2.13.4",
#   "rich==15.0.0",
#   "ruff==0.16.3",
# ]
# ///
"""One-command LFM2.5-VL + Docker Model Runner + Open WebUI bootstrapper.

Run:

    uv run lfm_vl_stack.py

The default ``up`` command validates an Apple-Silicon Mac, starts Docker
Desktop when needed, repairs the Docker Model Runner CLI plugin when safe,
enables the local OpenAI-compatible API, pulls/configures LFM2.5-VL, performs
real text and generated-image inference tests, launches a pinned Open WebUI
container through a grammar-safe compatibility proxy, and verifies streaming end to end.

Useful commands:

    uv run lfm_vl_stack.py doctor
    uv run lfm_vl_stack.py test
    uv run lfm_vl_stack.py status
    uv run lfm_vl_stack.py logs --follow
    uv run lfm_vl_stack.py down
    uv run lfm_vl_stack.py self-check
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import hashlib
import json
import os
import platform
import re
import secrets
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import time
import traceback
import webbrowser
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, TypeVar, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

APP_NAME: Final = "lfm-vl-stack"
APP_VERSION: Final = "2.1.2"
MODEL_REPOSITORY: Final = "hf.co/LiquidAI/LFM2.5-VL-450M-GGUF"
DEFAULT_WEBUI_IMAGE: Final = "ghcr.io/open-webui/open-webui:v0.11.0"
FALLBACK_WEBUI_IMAGE: Final = "ghcr.io/open-webui/open-webui:main"
CONTAINER_NAME: Final = "lfm25-vl-openwebui"
PROXY_CONTAINER_NAME: Final = "lfm25-vl-dmr-compat"
NETWORK_NAME: Final = "lfm25-vl-network"
VOLUME_NAME: Final = "lfm25-vl-openwebui-data"
MANAGED_LABEL: Final = "io.openai.lfm-vl-stack.managed"
CONFIG_LABEL: Final = "io.openai.lfm-vl-stack.config-sha"
VERSION_LABEL: Final = "io.openai.lfm-vl-stack.version"
DEFAULT_DMR_PORT: Final = 12434
DEFAULT_WEBUI_PORT: Final = 3000
PROXY_PORT: Final = 8000
MAX_CAPTURE_CHARS: Final = 24_000
CONFIG_DIR: Final = Path.home() / ".config" / APP_NAME
CACHE_DIR: Final = Path.home() / ".cache" / APP_NAME
STATE_FILE: Final = CONFIG_DIR / "state.json"
SECRET_FILE: Final = CONFIG_DIR / "webui-secret"
LAST_REPORT_FILE: Final = CONFIG_DIR / "last-report.json"
PROXY_SCRIPT_FILE: Final = CONFIG_DIR / "dmr-compat-proxy.py"

PROXY_SOURCE: Final = r'''#!/usr/bin/env python3
"""Tiny OpenAI-compatible proxy that removes grammar-producing fields."""
from __future__ import annotations

import http.client
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

UPSTREAM_BASE = os.environ.get(
    "UPSTREAM_BASE",
    "http://host.docker.internal:12434/engines/v1",
).rstrip("/")
LISTEN_HOST = os.environ.get("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8000"))
UPSTREAM_TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT", "600"))
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(64 * 1024 * 1024)))

# LFM2.5-VL image/text chat does not need these. In llama.cpp they can be
# translated into GBNF grammars, and malformed or empty values can prevent the
# sampler from initializing. Native tool calling and structured output are
# intentionally disabled at this compatibility boundary.
GRAMMAR_FIELDS = frozenset(
    {
        "grammar",
        "grammar_lazy",
        "grammar_triggers",
        "response_format",
        "json_schema",
        "guided_json",
        "guided_grammar",
        "guided_regex",
        "guided_choice",
        "structured_output",
        "structured_outputs",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "functions",
        "function_call",
    }
)
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "host",
    }
)


def sanitize_payload(value: Any) -> tuple[Any, tuple[str, ...]]:
    """Remove grammar/tool controls while preserving text and image content."""
    if not isinstance(value, dict):
        return value, ()
    cleaned = dict(value)
    removed: list[str] = []
    for key in sorted(GRAMMAR_FIELDS):
        if key in cleaned:
            cleaned.pop(key, None)
            removed.append(key)
    for parent_name in ("extra_body", "params"):
        parent = cleaned.get(parent_name)
        if not isinstance(parent, dict):
            continue
        parent_copy = dict(parent)
        for key in sorted(GRAMMAR_FIELDS):
            if key in parent_copy:
                parent_copy.pop(key, None)
                removed.append(f"{parent_name}.{key}")
        cleaned[parent_name] = parent_copy
    return cleaned, tuple(removed)


def upstream_target(incoming_path: str) -> tuple[str, str, int, str]:
    """Return scheme, host, port, and normalized upstream path."""
    parsed = urlsplit(UPSTREAM_BASE)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid UPSTREAM_BASE: {UPSTREAM_BASE!r}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    base_path = parsed.path.rstrip("/")
    path = incoming_path.split("?", 1)[0]
    query = "?" + incoming_path.split("?", 1)[1] if "?" in incoming_path else ""
    for prefix in ("/engines/v1", "/v1"):
        if path == prefix:
            path = "/"
            break
        if path.startswith(prefix + "/"):
            path = path[len(prefix) :]
            break
    if not path.startswith("/"):
        path = "/" + path
    return parsed.scheme, parsed.hostname, port, base_path + path + query


class ProxyHandler(BaseHTTPRequestHandler):
    """Forward OpenAI-compatible calls to Docker Model Runner."""

    protocol_version = "HTTP/1.0"
    server_version = "lfm-dmr-compat/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("proxy: " + (fmt % args) + "\n")

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            payload = json.dumps({"status": "ok"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._forward(None, ())

    def do_POST(self) -> None:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self.send_error(413, "Request body too large")
            return
        body = self.rfile.read(length)
        removed: tuple[str, ...] = ()
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        path_without_query = self.path.split("?", 1)[0].rstrip("/")
        is_chat = path_without_query.endswith("/chat/completions")
        if body and content_type == "application/json" and is_chat:
            try:
                decoded: Any = json.loads(body)
            except json.JSONDecodeError:
                self.send_error(400, "Request body is not valid JSON")
                return
            decoded, removed = sanitize_payload(decoded)
            if removed:
                self.log_message("removed grammar-producing fields: %s", ",".join(removed))
            body = json.dumps(decoded, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self._forward(body, removed)

    def _forward(self, body: bytes | None, removed: tuple[str, ...]) -> None:
        connection: http.client.HTTPConnection | http.client.HTTPSConnection | None = None
        try:
            scheme, host, port, target = upstream_target(self.path)
            connection_type = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
            connection = connection_type(host, port, timeout=UPSTREAM_TIMEOUT)
            headers = {
                name: value
                for name, value in self.headers.items()
                if name.lower() not in HOP_BY_HOP
            }
            headers["Host"] = host
            headers["Connection"] = "close"
            if body is not None:
                headers["Content-Length"] = str(len(body))
            connection.request(self.command, target, body=body, headers=headers)
            response = connection.getresponse()
            self.send_response(response.status, response.reason)
            for name, value in response.getheaders():
                if name.lower() not in HOP_BY_HOP:
                    self.send_header(name, value)
            self.send_header("Connection", "close")
            self.send_header("X-LFM-Compat-Proxy", "1")
            if removed:
                self.send_header("X-LFM-Compat-Removed", ",".join(removed))
            self.end_headers()
            while True:
                chunk = response.read1(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            if not self.wfile.closed:
                try:
                    self.send_error(502, f"Upstream request failed: {type(exc).__name__}: {exc}")
                except (BrokenPipeError, ConnectionResetError):
                    pass
        finally:
            if connection is not None:
                connection.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    upstream_target("/v1/models")
    server = Server((LISTEN_HOST, LISTEN_PORT), ProxyHandler)
    print(f"lfm-dmr-compat listening on {LISTEN_HOST}:{LISTEN_PORT} -> {UPSTREAM_BASE}", flush=True)
    server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
'''


class CommandName(StrEnum):
    """Supported top-level commands."""

    UP = "up"
    DOCTOR = "doctor"
    TEST = "test"
    STATUS = "status"
    LOGS = "logs"
    DOWN = "down"
    SELF_CHECK = "self-check"


class ProfileName(StrEnum):
    """Performance/quality presets."""

    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


class CheckLevel(StrEnum):
    """Result severity."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class Profile:
    """Resolved model configuration for a preset."""

    quantization: str
    context_size: int
    image_size: int
    description: str


PROFILES: Final[Mapping[ProfileName, Profile]] = {
    ProfileName.FAST: Profile(
        quantization="Q4_0",
        context_size=4096,
        image_size=512,
        description="Lowest latency and memory use",
    ),
    ProfileName.BALANCED: Profile(
        quantization="Q4_K_M",
        context_size=8192,
        image_size=1024,
        description="Balanced speed, quality, and context",
    ),
    ProfileName.QUALITY: Profile(
        quantization="Q8_0",
        context_size=8192,
        image_size=1024,
        description="Near-original weights with comfortable 16 GB headroom",
    ),
}


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Captured subprocess result."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def combined_output(self) -> str:
        """Return non-empty stdout/stderr sections."""
        return "\n".join(part.strip() for part in (self.stdout, self.stderr) if part.strip())


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One recorded validation result."""

    name: str
    level: CheckLevel
    detail: str
    duration_seconds: float = 0.0
    hint: str | None = None


class StackError(RuntimeError):
    """Expected operational failure with remediation context."""

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        details: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details


class CommandError(StackError):
    """A subprocess failed or timed out."""

    def __init__(self, message: str, result: CommandResult, *, hint: str | None = None) -> None:
        super().__init__(message, hint=hint, details=tail_text(result.combined_output))
        self.result = result


class Settings(BaseModel):
    """Validated command-line settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: CommandName = CommandName.UP
    profile: ProfileName = ProfileName.QUALITY
    model: str | None = None
    context_size: int | None = Field(default=None, ge=512, le=32768)
    image_size: int | None = Field(default=None, ge=256, le=4096)
    port: int = Field(default=DEFAULT_WEBUI_PORT, ge=1, le=65535)
    dmr_port: int = Field(default=DEFAULT_DMR_PORT, ge=1, le=65535)
    webui_image: str = Field(default=DEFAULT_WEBUI_IMAGE, min_length=3)
    auth: bool = False
    lan: bool = False
    no_open: bool = False
    skip_self_check: bool = False
    skip_pull: bool = False
    refresh_image: bool = False
    strict_vision: bool = False
    strict_port: bool = False
    no_fallback: bool = False
    allow_unsupported_host: bool = False
    startup_timeout: int = Field(default=240, ge=30, le=1800)
    command_timeout: int = Field(default=120, ge=10, le=1800)
    allow_ui_config: bool = False
    follow: bool = False
    purge_data: bool = False
    yes: bool = False
    verbose: bool = False

    @property
    def profile_spec(self) -> Profile:
        """Return the selected preset."""
        return PROFILES[self.profile]

    @property
    def effective_context_size(self) -> int:
        """Return user override or preset context size."""
        return self.context_size or self.profile_spec.context_size

    @property
    def effective_image_size(self) -> int:
        """Return user override or preset image resize bound."""
        return self.image_size or self.profile_spec.image_size

    @property
    def requested_model(self) -> str:
        """Return an explicit model or preset-derived GGUF reference."""
        if self.model:
            return self.model.strip()
        return f"{MODEL_REPOSITORY}:{self.profile_spec.quantization}"

    @property
    def effective_auth(self) -> bool:
        """Force authentication whenever the UI is exposed to the LAN."""
        return self.auth or self.lan

    @property
    def bind_host(self) -> str:
        """Bind locally unless LAN access was explicitly requested."""
        return "0.0.0.0" if self.lan else "127.0.0.1"

    def model_candidates(self) -> tuple[str, ...]:
        """Return ordered model pull candidates with safe fallbacks."""
        primary = self.requested_model
        if self.model or self.no_fallback:
            return (primary,)
        quantizations = {
            ProfileName.QUALITY: ("Q8_0", "Q4_K_M", "Q4_0"),
            ProfileName.BALANCED: ("Q4_K_M", "Q4_0", "Q8_0"),
            ProfileName.FAST: ("Q4_0", "Q4_K_M", "Q8_0"),
        }[self.profile]
        return tuple(f"{MODEL_REPOSITORY}:{quantization}" for quantization in quantizations)


class State(BaseModel):
    """Durable state for later status/test commands."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    app_version: str
    created_at: str
    model_ref: str
    model_id: str
    context_size: int
    image_size: int
    profile: ProfileName
    webui_image: str
    webui_port: int
    dmr_port: int
    auth_enabled: bool
    lan_enabled: bool
    config_sha: str


T = TypeVar("T")


def utc_now() -> str:
    """Return a stable UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def tail_text(value: str, limit: int = MAX_CAPTURE_CHARS) -> str:
    """Bound stored/displayed command output."""
    if len(value) <= limit:
        return value
    return f"… {len(value) - limit} characters omitted …\n{value[-limit:]}"


def first_nonempty_line(value: str) -> str:
    """Return the first non-blank line."""
    return next((line.strip() for line in value.splitlines() if line.strip()), "")


def human_bytes(value: int) -> str:
    """Format a byte count in binary units."""
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


def redact_text(value: str, secrets_to_hide: Sequence[str] = ()) -> str:
    """Remove likely credentials from diagnostics and verbose command output."""
    redacted = value
    for secret in sorted((item for item in secrets_to_hide if item), key=len, reverse=True):
        redacted = redacted.replace(secret, "<redacted>")
    patterns = (
        (r'''(?i)(WEBUI_SECRET_KEY[=:]\s*)[^\s,\]}"']+''', r"\1<redacted>"),
        (r"(?i)(authorization:\s*bearer\s+)[^\s,\]}\"']+", r"\1<redacted>"),
        (r'''(?i)(OPENAI_API_KEY[=:]\s*)[^\s,\]}"']+''', r"\1<redacted>"),
    )
    for pattern, replacement in patterns:
        redacted = re.sub(pattern, replacement, redacted)
    return redacted


def atomic_write_text(path: Path, value: str, *, mode: int = 0o600) -> None:
    """Atomically replace a small state/configuration file."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_object(value: str, *, context: str) -> dict[str, Any]:
    """Parse a JSON object with a helpful failure."""
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError as exc:
        raise StackError(f"Could not parse JSON from {context}.", details=value) from exc
    if not isinstance(parsed, dict):
        raise StackError(f"Expected a JSON object from {context}.", details=value)
    return cast(dict[str, Any], parsed)


def nested_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    """Return a nested mapping if present."""
    candidate = value.get(key)
    if isinstance(candidate, dict):
        return cast(dict[str, Any], candidate)
    return None


def normalize_model_id(value: str) -> str:
    """Normalize API/CLI model identifiers for matching."""
    normalized = value.strip().lower()
    normalized = normalized.removeprefix("https://")
    normalized = normalized.removeprefix("http://")
    normalized = normalized.removeprefix("hf.co/")
    normalized = normalized.removeprefix("huggingface.co/")
    normalized = normalized.replace("_", "-")
    return re.sub(r"[^a-z0-9:/.-]+", "", normalized)


def choose_model_id(requested: str, available: Sequence[str]) -> str | None:
    """Resolve a requested model across minor Docker API naming differences."""
    if not available:
        return None
    wanted = normalize_model_id(requested)
    exact = [item for item in available if normalize_model_id(item) == wanted]
    if exact:
        return exact[0]
    wanted_without_host = wanted.removeprefix("hf.co/").removeprefix("huggingface.co/")
    suffix = [
        item
        for item in available
        if normalize_model_id(item).removeprefix("hf.co/").removeprefix("huggingface.co/")
        == wanted_without_host
    ]
    if suffix:
        return suffix[0]
    requested_parts = wanted_without_host.rsplit(":", 1)
    if len(requested_parts) == 2:
        repository, tag = requested_parts
        fuzzy = [
            item
            for item in available
            if repository in normalize_model_id(item)
            and normalize_model_id(item).endswith(f":{tag}")
        ]
        if len(fuzzy) == 1:
            return fuzzy[0]
    return None


def free_tcp_port(host: str, preferred: int, *, strict: bool) -> int:
    """Return the preferred free port or a safe ephemeral fallback."""
    def available(port: int) -> bool:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError:
                return False
            return True

    if available(preferred):
        return preferred
    if strict:
        raise StackError(
            f"TCP port {preferred} is already in use on {host}.",
            hint="Stop the conflicting service or choose another --port.",
        )
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        selected = int(probe.getsockname()[1])
    if not available(selected):
        raise StackError("Could not reserve a free Open WebUI port.")
    return selected


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    """Create one PNG chunk."""
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)


def solid_png_data_uri(width: int = 256, height: int = 256) -> str:
    """Generate a tiny deterministic red PNG without image dependencies."""
    if not 1 <= width <= 512 or not 1 <= height <= 512:
        raise ValueError("PNG dimensions must be between 1 and 512")
    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + (b"\xff\x00\x00" * width)
    pixels = row * height
    png = signature + png_chunk(b"IHDR", header) + png_chunk(b"IDAT", zlib.compress(pixels, 9)) + png_chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def extract_assistant_text(payload: Mapping[str, Any]) -> str:
    """Extract assistant text from an OpenAI-compatible chat response."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(cast(str, item["text"]))
        return "\n".join(parts).strip()
    return ""


def load_state() -> State | None:
    """Load durable stack state, treating invalid state as absent."""
    try:
        raw = STATE_FILE.read_text(encoding="utf-8")
        return State.model_validate_json(raw)
    except (OSError, ValidationError, ValueError):
        return None


def save_state(state: State) -> None:
    """Persist state atomically."""
    atomic_write_text(STATE_FILE, state.model_dump_json(indent=2) + "\n")


def stable_secret() -> str:
    """Create or read the stable Open WebUI encryption/signing secret."""
    try:
        existing = SECRET_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        existing = ""
    if len(existing) >= 32:
        try:
            os.chmod(SECRET_FILE, 0o600)
        except OSError:
            pass
        return existing
    value = secrets.token_urlsafe(48)
    atomic_write_text(SECRET_FILE, value + "\n")
    return value


def parse_version(value: str) -> tuple[int, ...]:
    """Extract a comparable numeric version prefix."""
    match = re.search(r"\d+(?:\.\d+)+", value)
    if not match:
        return ()
    return tuple(int(part) for part in match.group(0).split("."))


def backend_is_running(status_output: str, backend: str) -> bool:
    """Return whether ``docker model status`` reports a backend as running.

    Docker Desktop releases have emitted at least two human-readable formats:
    a legacy ``backend: running`` list and a newer whitespace-delimited table.
    Keep this parser deliberately narrow: match only a backend name at the start
    of a line followed by an optional colon and the exact status ``running``.
    """
    pattern = rf"(?im)^\s*{re.escape(backend)}(?:\s*:\s*|\s+)running\b"
    return re.search(pattern, status_output) is not None


def command_text(argv: Sequence[str]) -> str:
    """Shell-quote a command only for display."""
    return " ".join(shlex.quote(part) for part in argv)


class Runner:
    """Timeout-bounded subprocess execution with captured output."""

    def __init__(self, console: Console, *, default_timeout: int, verbose: bool) -> None:
        self.console = console
        self.default_timeout = default_timeout
        self.verbose = verbose
        self.secrets: list[str] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: int | None = None,
        check: bool = True,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        input_text: str | None = None,
    ) -> CommandResult:
        """Run a command without a shell and return bounded text output."""
        command = tuple(str(part) for part in argv)
        if not command:
            raise ValueError("argv must not be empty")
        if self.verbose:
            self.console.print(f"[dim]$ {redact_text(command_text(command), self.secrets)}[/dim]")
        process_env = os.environ.copy()
        if env is not None:
            process_env.update(env)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                input=input_text,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout or self.default_timeout,
                check=False,
                env=process_env,
                cwd=cwd,
            )
        except FileNotFoundError as exc:
            result = CommandResult(command, 127, "", str(exc), time.monotonic() - started)
            raise CommandError(f"Command was not found: {command[0]}", result) from exc
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            result = CommandResult(command, 124, stdout, stderr, time.monotonic() - started)
            raise CommandError(
                f"Command timed out after {timeout or self.default_timeout}s: {command[0]}",
                result,
            ) from exc
        result = CommandResult(
            command,
            completed.returncode,
            tail_text(completed.stdout),
            tail_text(completed.stderr),
            time.monotonic() - started,
        )
        if self.verbose and result.combined_output:
            self.console.print(f"[dim]{redact_text(tail_text(result.combined_output, 4000), self.secrets)}[/dim]")
        if check and result.returncode != 0:
            raise CommandError(
                f"Command failed with exit code {result.returncode}: {command_text(command)}",
                result,
            )
        return result


class Reporter:
    """Rich output plus machine-readable check collection."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.results: list[CheckResult] = []

    def _record(
        self,
        name: str,
        level: CheckLevel,
        detail: str,
        *,
        duration_seconds: float = 0.0,
        hint: str | None = None,
    ) -> None:
        icons = {
            CheckLevel.PASS: "[bold green]✓[/bold green]",
            CheckLevel.WARN: "[bold yellow]![/bold yellow]",
            CheckLevel.FAIL: "[bold red]✗[/bold red]",
            CheckLevel.SKIP: "[dim]–[/dim]",
        }
        self.results.append(CheckResult(name, level, detail, duration_seconds, hint))
        suffix = f" [dim]({duration_seconds:.1f}s)[/dim]" if duration_seconds >= 0.05 else ""
        self.console.print(f"{icons[level]} [bold]{name}[/bold]: {detail}{suffix}")
        if hint:
            self.console.print(f"  [dim]Hint: {hint}[/dim]")

    def passed(self, name: str, detail: str, *, duration_seconds: float = 0.0) -> None:
        """Record a success."""
        self._record(name, CheckLevel.PASS, detail, duration_seconds=duration_seconds)

    def warn(self, name: str, detail: str, *, hint: str | None = None) -> None:
        """Record a non-fatal warning."""
        self._record(name, CheckLevel.WARN, detail, hint=hint)

    def failed(self, name: str, detail: str, *, hint: str | None = None) -> None:
        """Record a failure."""
        self._record(name, CheckLevel.FAIL, detail, hint=hint)

    def skipped(self, name: str, detail: str) -> None:
        """Record a skipped check."""
        self._record(name, CheckLevel.SKIP, detail)

    def step(self, name: str, action: Callable[[], T], detail: Callable[[T], str]) -> T:
        """Run an action and report its duration."""
        started = time.monotonic()
        try:
            result = action()
        except BaseException:
            raise
        self.passed(name, detail(result), duration_seconds=time.monotonic() - started)
        return result

    def save(self, *, command: CommandName, exit_code: int) -> None:
        """Persist the latest check report atomically."""
        payload = {
            "app": APP_NAME,
            "version": APP_VERSION,
            "generated_at": utc_now(),
            "command": command.value,
            "exit_code": exit_code,
            "checks": [asdict(item) for item in self.results],
        }
        atomic_write_text(LAST_REPORT_FILE, json.dumps(payload, indent=2, default=str) + "\n")


class Stack:
    """Orchestrate the local multimodal stack."""

    def __init__(self, settings: Settings, console: Console) -> None:
        self.settings = settings
        self.console = console
        self.reporter = Reporter(console)
        self.runner = Runner(
            console,
            default_timeout=settings.command_timeout,
            verbose=settings.verbose,
        )
        self.docker: str | None = None
        self.model_ref: str | None = None
        self.model_id: str | None = None
        self.webui_port = settings.port
        self.webui_image = settings.webui_image

    def execute(self) -> None:
        """Dispatch the requested command."""
        dispatch: Mapping[CommandName, Callable[[], None]] = {
            CommandName.UP: self.up,
            CommandName.DOCTOR: self.doctor,
            CommandName.TEST: self.test,
            CommandName.STATUS: self.status,
            CommandName.LOGS: self.logs,
            CommandName.DOWN: self.down,
            CommandName.SELF_CHECK: lambda: self.self_check(force=True),
        }
        dispatch[self.settings.command]()

    def up(self) -> None:
        """Create or reconcile the complete stack and verify it end to end."""
        self.console.print(
            Panel.fit(
                "[bold]LFM2.5-VL local vision stack[/bold]\n"
                f"Profile: [cyan]{self.settings.profile.value}[/cyan] — "
                f"{self.settings.profile_spec.description}",
                border_style="cyan",
            )
        )
        self.self_check(force=False)
        self.check_host()
        self.ensure_docker()
        self.ensure_model_runner()
        model_ref, model_id = self.ensure_model()
        self.model_ref = model_ref
        self.model_id = model_id
        self.configure_model(model_ref)
        self.smoke_test_text(model_id)
        self.smoke_test_vision(model_id)
        self.check_metal_acceleration()
        self.ensure_openwebui(model_id, model_ref)
        self.show_ready()

    def doctor(self) -> None:
        """Validate and repair prerequisites, then deeply test existing state."""
        self.console.print(Panel.fit("[bold]LFM2.5-VL stack doctor[/bold]", border_style="cyan"))
        self.self_check(force=False)
        self.check_host()
        self.ensure_docker()
        self.ensure_model_runner()
        self.check_existing_stack(deep=True)

    def test(self) -> None:
        """Run live direct-model and container connectivity tests."""
        self.console.print(Panel.fit("[bold]LFM2.5-VL live tests[/bold]", border_style="cyan"))
        self.self_check(force=False)
        self.ensure_docker()
        self.ensure_model_runner()
        model_ref, model_id = self.ensure_model()
        self.model_ref = model_ref
        self.model_id = model_id
        self.smoke_test_text(model_id)
        self.smoke_test_vision(model_id)
        self.check_metal_acceleration()
        saved_state = load_state()
        self.check_existing_stack(deep=False)
        proxy = self.inspect_container(PROXY_CONTAINER_NAME)
        if proxy is not None and container_running(proxy):
            self.verify_compat_proxy_streaming_chat(model_id)
        elif saved_state is None:
            inspect = self.inspect_container(CONTAINER_NAME)
            if inspect is not None and container_running(inspect):
                self.reporter.warn(
                    "Grammar-safe streaming",
                    "Compatibility proxy is missing; run the default up command to repair Open WebUI",
                )

    def status(self) -> None:
        """Show current state without pulling or mutating resources."""
        self.console.print(Panel.fit("[bold]LFM2.5-VL stack status[/bold]", border_style="cyan"))
        self.discover_docker(required=False)
        if self.docker is None:
            raise StackError(
                "Docker CLI was not found.",
                hint="Install Docker Desktop, then run the default up command.",
            )
        daemon = self.run_docker("info", "--format", "{{.ServerVersion}}", check=False, timeout=15)
        if daemon.returncode != 0:
            raise StackError(
                "Docker daemon is not reachable.",
                hint="Start Docker Desktop, then run the default up command.",
                details=daemon.combined_output,
            )
        self.reporter.passed("Docker daemon", f"Server {daemon.stdout.strip() or 'running'}")
        model_status = self.run_docker("model", "status", check=False, timeout=30)
        if model_status.returncode != 0:
            raise StackError(
                "Docker Model Runner is unavailable.",
                hint="Run this script with no command to enable or repair it.",
                details=model_status.combined_output,
            )
        self.reporter.passed(
            "Docker Model Runner",
            first_nonempty_line(model_status.stdout) or "Running",
        )
        if self.dmr_api_available():
            self.reporter.passed("Model Runner API", f"127.0.0.1:{self.settings.dmr_port}")
        else:
            self.reporter.warn(
                "Model Runner API",
                f"CLI is running but TCP API is unavailable on {self.settings.dmr_port}",
                hint="Run the default up command to repair TCP access.",
            )
        self.check_existing_stack(deep=False)

    def logs(self) -> None:
        """Print or follow managed service logs."""
        self.discover_docker(required=True)
        ui = self.inspect_container(CONTAINER_NAME)
        proxy = self.inspect_container(PROXY_CONTAINER_NAME)
        if ui is None and proxy is None:
            raise StackError(
                "Neither managed Open WebUI nor its compatibility proxy exists.",
                hint="Run the script with no command first.",
            )
        if ui is not None:
            self.require_managed_container(ui)
        if proxy is not None:
            self.require_managed_container(proxy)
        assert self.docker is not None
        proxy_logs = self.run_docker(
            "logs", "--tail", "200", PROXY_CONTAINER_NAME, check=False, timeout=30
        ) if proxy is not None else None
        if proxy_logs is not None:
            self.console.print(
                Panel(
                    redact_text(proxy_logs.combined_output or "No compatibility-proxy logs.", self.runner.secrets),
                    title="DMR compatibility proxy",
                )
            )
        if self.settings.follow:
            if ui is None:
                self.console.print("[dim]Following compatibility-proxy logs; press Ctrl-C to stop.[/dim]")
                subprocess.run(
                    self.docker_command("logs", "--tail", "200", "--follow", PROXY_CONTAINER_NAME),
                    check=False,
                )
            else:
                self.console.print("[dim]Following Open WebUI logs; press Ctrl-C to stop.[/dim]")
                subprocess.run(
                    self.docker_command("logs", "--tail", "200", "--follow", CONTAINER_NAME),
                    check=False,
                )
            return
        if ui is not None:
            ui_logs = self.run_docker("logs", "--tail", "200", CONTAINER_NAME, check=False, timeout=30)
            self.console.print(
                Panel(
                    redact_text(ui_logs.combined_output or "No Open WebUI logs.", self.runner.secrets),
                    title="Open WebUI",
                )
            )
        model_logs = self.run_docker("model", "logs", check=False, timeout=30)
        self.console.print(
            Panel(
                redact_text(tail_text(model_logs.combined_output) or "No Model Runner logs.", self.runner.secrets),
                title="Docker Model Runner",
            )
        )

    def down(self) -> None:
        """Stop managed containers and optionally purge durable data."""
        if self.settings.purge_data and not self.settings.yes:
            raise StackError(
                "Refusing to delete Open WebUI data without --yes.",
                hint="Use: down --purge-data --yes",
            )
        self.discover_docker(required=True)
        targets = (
            (CONTAINER_NAME, "Open WebUI"),
            (PROXY_CONTAINER_NAME, "DMR compatibility proxy"),
        )
        for name, display_name in targets:
            inspect = self.inspect_container(name)
            if inspect is None:
                self.reporter.skipped(display_name, "Managed container is already absent")
                continue
            self.require_managed_container(inspect)
            if self.settings.purge_data:
                started = time.monotonic()
                self.run_docker("rm", "--force", name, timeout=60)
                self.reporter.passed(
                    f"Remove {display_name}",
                    "Removed",
                    duration_seconds=time.monotonic() - started,
                )
                continue
            if not container_running(inspect):
                self.reporter.skipped(display_name, "Managed container is already stopped")
                continue
            started = time.monotonic()
            self.run_docker("stop", "--time", "15", name, timeout=30)
            self.reporter.passed(
                f"Stop {display_name}",
                "Stopped",
                duration_seconds=time.monotonic() - started,
            )
        if self.settings.purge_data:
            self.purge_volume_and_state()
        else:
            self.reporter.passed("Persistence", "Model, chats, configuration, and volume retained")

    def purge_volume_and_state(self) -> None:
        """Remove managed persistence after explicit destructive confirmation."""
        result = self.run_docker("volume", "rm", VOLUME_NAME, check=False, timeout=60)
        if result.returncode == 0:
            self.reporter.passed("Remove Open WebUI volume", "Removed")
        else:
            message = result.combined_output.lower()
            if "no such volume" in message or "not found" in message:
                self.reporter.skipped("Remove Open WebUI volume", "Already absent")
            else:
                raise StackError(
                    "Could not remove the Open WebUI data volume.",
                    hint="Check whether another container is still using the volume.",
                    details=result.combined_output,
                )
        network = self.run_docker("network", "rm", NETWORK_NAME, check=False, timeout=60)
        if network.returncode == 0:
            self.reporter.passed("Remove managed network", NETWORK_NAME)
        else:
            message = network.combined_output.lower()
            if "no such network" in message or "not found" in message:
                self.reporter.skipped("Remove managed network", "Already absent")
            else:
                raise StackError(
                    "Could not remove the managed Docker network.",
                    hint="Check whether another container is still attached to it.",
                    details=network.combined_output,
                )
        STATE_FILE.unlink(missing_ok=True)
        SECRET_FILE.unlink(missing_ok=True)
        PROXY_SCRIPT_FILE.unlink(missing_ok=True)
        self.reporter.passed("Local state", "Removed state, managed secret, and proxy source")

    def self_check(self, *, force: bool) -> None:
        """Compile, lint, type-check, and exercise internal helpers."""
        if self.settings.skip_self_check and not force:
            self.reporter.skipped("Script self-check", "Disabled with --skip-self-check")
            return
        script_path = Path(__file__).resolve()
        ruff = shutil.which("ruff")
        if ruff is None:
            raise StackError("ruff was not installed by uv from the PEP 723 metadata.")
        ruff_version = self.runner.run((ruff, "--version"), timeout=20).stdout.strip()
        mypy_version = self.runner.run((sys.executable, "-m", "mypy", "--version"), timeout=20).stdout.strip()
        digest_material = b"\0".join(
            (
                script_path.read_bytes(),
                ruff_version.encode(),
                mypy_version.encode(),
                sys.version.encode(),
                platform.platform().encode(),
            )
        )
        digest = hashlib.sha256(digest_material).hexdigest()
        cache_file = CACHE_DIR / f"self-check-{digest}.ok"
        if cache_file.exists() and not force:
            self.reporter.passed("Script self-check", f"Cached strict checks ({digest[:10]})")
            return

        def checks() -> str:
            source = script_path.read_text(encoding="utf-8")
            compile(source, str(script_path), "exec")
            mypy_config = CACHE_DIR / "mypy.ini"
            atomic_write_text(mypy_config, "[mypy]\n", mode=0o600)
            self.runner.run(
                (
                    ruff,
                    "check",
                    "--isolated",
                    "--no-cache",
                    "--select",
                    "E4,E7,E9,F",
                    str(script_path),
                ),
                timeout=60,
            )
            self.runner.run(
                (
                    sys.executable,
                    "-m",
                    "mypy",
                    "--strict",
                    "--config-file",
                    str(mypy_config),
                    "--cache-dir",
                    str(CACHE_DIR / "mypy"),
                    "--python-version",
                    "3.12",
                    "--show-error-codes",
                    str(script_path),
                ),
                timeout=180,
            )
            run_internal_tests()
            CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            atomic_write_text(cache_file, f"{utc_now()}\n")
            return f"compile + Ruff + mypy --strict + internal tests ({digest[:10]})"

        self.reporter.step("Script self-check", checks, lambda detail: detail)

    def check_host(self) -> None:
        """Validate target hardware and common performance hazards."""
        if sys.version_info < (3, 12):
            raise StackError("Python 3.12 or newer is required.", hint="Run the script through uv.")
        self.reporter.passed("Python", platform.python_version())
        system = platform.system()
        machine = platform.machine().lower()
        supported = system == "Darwin" and machine in {"arm64", "aarch64"}
        if supported:
            self.reporter.passed("Host platform", f"{system} {machine}; native Apple Silicon path")
        elif self.settings.allow_unsupported_host:
            self.reporter.warn(
                "Host platform",
                f"{system} {machine} is outside the intended Apple Silicon target",
            )
        else:
            raise StackError(
                f"Unsupported host: {system} {machine}.",
                hint="This script targets native Apple Silicon macOS. Use --allow-unsupported-host only for deliberate testing.",
            )
        if system == "Darwin":
            translated = self.runner.run(("sysctl", "-in", "sysctl.proc_translated"), check=False, timeout=10)
            if translated.stdout.strip() == "1":
                raise StackError(
                    "Python is running through Rosetta instead of native arm64.",
                    hint="Install native arm64 uv and rerun from an arm64 terminal.",
                )
            memory = self.read_mac_memory()
            if memory >= 14 * 1024**3:
                self.reporter.passed("Unified memory", f"{human_bytes(memory)} detected")
            elif memory >= 8 * 1024**3:
                self.reporter.warn(
                    "Unified memory",
                    f"{human_bytes(memory)} detected; consider --profile fast if macOS is under pressure",
                )
            else:
                raise StackError(
                    "Less than 8 GiB of unified memory was detected.",
                    hint="This stack is not sized for this machine.",
                )
            self.reporter.passed("macOS", platform.mac_ver()[0] or "unknown")
            self.check_power_settings()
        free_disk = shutil.disk_usage(Path.home()).free
        if free_disk >= 8 * 1024**3:
            self.reporter.passed("Free disk space", human_bytes(free_disk))
        elif free_disk >= 3 * 1024**3:
            self.reporter.warn("Free disk space", f"Only {human_bytes(free_disk)} remains")
        else:
            raise StackError(
                "Less than 3 GiB of free disk space remains.",
                hint="Free disk space before pulling images and models.",
            )
        if self.settings.lan and not self.settings.effective_auth:
            raise StackError("LAN exposure requires authentication.")
        if self.settings.lan:
            self.reporter.warn(
                "LAN mode",
                "Open WebUI will listen on all interfaces and authentication is forced on",
            )
        else:
            self.reporter.passed("Network policy", "Open WebUI will bind to loopback only")

    def read_mac_memory(self) -> int:
        """Read physical memory from macOS sysctl."""
        result = self.runner.run(("sysctl", "-n", "hw.memsize"), timeout=10)
        try:
            return int(result.stdout.strip())
        except ValueError as exc:
            raise StackError("Could not parse macOS memory size.", details=result.stdout) from exc

    def check_power_settings(self) -> None:
        """Warn about battery and low-power modes that reduce sustained speed."""
        battery = self.runner.run(("pmset", "-g", "batt"), check=False, timeout=10)
        text = battery.combined_output
        if "AC Power" in text:
            self.reporter.passed("Power source", "AC power")
        elif "Battery Power" in text:
            self.reporter.warn("Power source", "Battery power may reduce sustained inference speed")
        else:
            self.reporter.warn("Power source", "Could not determine power source")
        custom = self.runner.run(("pmset", "-g", "custom"), check=False, timeout=10)
        low_power_values = re.findall(r"(?im)^\s*lowpowermode\s+(\d+)\s*$", custom.stdout)
        if "1" in low_power_values:
            self.reporter.warn(
                "Low Power Mode",
                "Enabled in at least one configured macOS power profile",
                hint="Disable Low Power Mode for maximum sustained throughput.",
            )
        else:
            self.reporter.passed("Low Power Mode", "No configured profile reports lowpowermode=1")

    def discover_docker(self, *, required: bool) -> None:
        """Find Docker CLI in standard native and Docker Desktop locations."""
        candidates = (
            shutil.which("docker"),
            "/opt/homebrew/bin/docker",
            "/usr/local/bin/docker",
            "/Applications/Docker.app/Contents/Resources/bin/docker",
        )
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                self.docker = str(Path(candidate).resolve())
                return
        if required:
            raise StackError(
                "Docker CLI was not found.",
                hint="Install or update Docker Desktop for Mac, then rerun this script.",
            )

    def docker_command(self, *args: str) -> tuple[str, ...]:
        """Build a Docker command after discovery."""
        if self.docker is None:
            raise StackError("Docker CLI has not been discovered.")
        return (self.docker, *args)

    def run_docker(
        self,
        *args: str,
        timeout: int | None = None,
        check: bool = True,
    ) -> CommandResult:
        """Run Docker with standard timeout/capture behavior."""
        return self.runner.run(self.docker_command(*args), timeout=timeout, check=check)

    def ensure_docker(self) -> None:
        """Find/start Docker Desktop and repair the model CLI plugin when safe."""
        self.discover_docker(required=True)
        assert self.docker is not None
        client = self.run_docker("version", "--format", "{{.Client.Version}}", check=False, timeout=20)
        if client.returncode != 0:
            raise StackError("Docker CLI could not report its version.", details=client.combined_output)
        self.reporter.passed("Docker CLI", client.stdout.strip() or "available")
        daemon = self.run_docker("info", "--format", "{{.ServerVersion}}", check=False, timeout=20)
        if daemon.returncode != 0:
            if platform.system() != "Darwin":
                raise StackError("Docker daemon is unavailable.", details=daemon.combined_output)
            application = Path("/Applications/Docker.app")
            if not application.exists():
                raise StackError(
                    "Docker Desktop is not installed in /Applications.",
                    hint="Install current Docker Desktop for Mac.",
                )
            launched = self.runner.run(("open", "-a", "Docker"), check=False, timeout=20)
            if launched.returncode != 0:
                raise StackError("Docker Desktop could not be launched.", details=launched.combined_output)
            self.wait_for_docker()
            daemon = self.run_docker("info", "--format", "{{.ServerVersion}}", timeout=30)
        server_version = daemon.stdout.strip() or "running"
        self.reporter.passed("Docker daemon", f"Server {server_version}")
        parsed = parse_version(server_version)
        if parsed and parsed < (28, 0):
            self.reporter.warn(
                "Docker version",
                f"Server {server_version} may be too old for the newest Model Runner features",
                hint="Update Docker Desktop before troubleshooting model errors.",
            )
        context = self.run_docker("context", "show", check=False, timeout=15)
        if context.returncode == 0:
            self.reporter.passed("Docker context", context.stdout.strip() or "default")
        self.ensure_model_plugin()

    def wait_for_docker(self) -> None:
        """Poll Docker Desktop until its daemon responds."""
        deadline = time.monotonic() + self.settings.startup_timeout
        last = CommandResult((), 1, "", "No attempts made", 0.0)
        with self.console.status("[bold cyan]Starting Docker Desktop…[/bold cyan]"):
            while time.monotonic() < deadline:
                last = self.run_docker("info", check=False, timeout=15)
                if last.returncode == 0:
                    return
                time.sleep(2)
        raise StackError(
            "Docker Desktop did not become ready before the startup timeout.",
            hint="Open Docker Desktop and resolve any displayed startup error.",
            details=last.combined_output,
        )

    def ensure_model_plugin(self) -> None:
        """Validate docker model and apply Docker's documented symlink repair."""
        probe = self.run_docker("model", "--help", check=False, timeout=20)
        if probe.returncode == 0:
            self.reporter.passed("Docker Model CLI", "docker model is available")
            return
        source_candidates = (
            Path("/Applications/Docker.app/Contents/Resources/cli-plugins/docker-model"),
            Path("/Applications/Docker.app/Contents/Resources/bin/docker-model"),
        )
        source = next((path for path in source_candidates if path.is_file()), None)
        if source is None:
            raise StackError(
                "Docker Model Runner CLI plugin is missing.",
                hint="Update Docker Desktop; Model Runner requires a recent release.",
                details=probe.combined_output,
            )
        destination_dir = Path.home() / ".docker" / "cli-plugins"
        destination = destination_dir / "docker-model"
        destination_dir.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() and destination.resolve() == source.resolve():
                pass
            else:
                raise StackError(
                    f"A conflicting Docker plugin already exists at {destination}.",
                    hint="Move it aside manually, then rerun; the script will not overwrite unknown files.",
                )
        else:
            destination.symlink_to(source)
            self.reporter.passed("Docker Model CLI repair", f"Created documented symlink at {destination}")
        retry = self.run_docker("model", "--help", check=False, timeout=20)
        if retry.returncode != 0:
            raise StackError(
                "docker model is still unavailable after the safe plugin repair.",
                hint="Restart or update Docker Desktop.",
                details=retry.combined_output,
            )
        self.reporter.passed("Docker Model CLI", "docker model is available")

    def dmr_url(self, endpoint: str) -> str:
        """Build a host-side OpenAI-compatible Model Runner URL."""
        return f"http://127.0.0.1:{self.settings.dmr_port}/engines/v1/{endpoint.lstrip('/')}"

    def dmr_api_available(self) -> bool:
        """Check the local unauthenticated Model Runner API without proxies."""
        try:
            response = httpx.get(self.dmr_url("models"), timeout=3.0, trust_env=False)
        except httpx.HTTPError:
            return False
        return response.status_code == httpx.codes.OK

    def ensure_model_runner(self) -> None:
        """Enable Model Runner/TCP API and ensure the llama.cpp Metal backend."""
        status = self.run_docker("model", "status", check=False, timeout=30)
        api_ok = self.dmr_api_available()
        if status.returncode != 0 or not api_ok:
            full = self.run_docker(
                "desktop",
                "enable",
                "model-runner",
                "--tcp",
                str(self.settings.dmr_port),
                "--cors",
                "none",
                check=False,
                timeout=180,
            )
            if full.returncode != 0:
                fallback = self.run_docker(
                    "desktop",
                    "enable",
                    "model-runner",
                    "--tcp",
                    str(self.settings.dmr_port),
                    check=False,
                    timeout=180,
                )
                if fallback.returncode != 0:
                    raise StackError(
                        "Docker Model Runner could not be enabled automatically.",
                        hint="Update Docker Desktop, then enable Model Runner and TCP access in Settings → AI.",
                        details=f"Primary attempt:\n{full.combined_output}\n\nFallback:\n{fallback.combined_output}",
                    )
                self.reporter.warn(
                    "CORS hardening",
                    "This Docker Desktop build accepted TCP enablement but not the --cors option",
                    hint="Keep Model Runner's port local and update Docker Desktop.",
                )
            status = self.wait_for_model_runner()
        else:
            self.reporter.passed("Model Runner TCP API", f"127.0.0.1:{self.settings.dmr_port}")
        if not backend_is_running(status.combined_output, "llama.cpp"):
            install = self.run_docker(
                "model",
                "install-runner",
                "--backend",
                "llama.cpp",
                "--gpu",
                "metal",
                check=False,
                timeout=300,
            )
            if install.returncode != 0:
                reinstall = self.run_docker(
                    "model",
                    "reinstall-runner",
                    "--backend",
                    "llama.cpp",
                    check=False,
                    timeout=300,
                )
                if reinstall.returncode != 0:
                    raise StackError(
                        "The llama.cpp inference engine is not running.",
                        hint="Update Docker Desktop and inspect `docker model logs`.",
                        details=f"Install:\n{install.combined_output}\n\nReinstall:\n{reinstall.combined_output}",
                    )
            status = self.wait_for_model_runner(required_backend="llama.cpp")
        if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
            self.reporter.passed("Inference backend", "llama.cpp running with automatic Apple Metal support")
        else:
            self.reporter.passed("Inference backend", "llama.cpp running")
        if backend_is_running(status.combined_output, "vllm"):
            self.reporter.warn("vLLM backend", "Installed but this GGUF model is intentionally routed to llama.cpp")
        else:
            self.reporter.passed("Backend routing", "GGUF model routes to llama.cpp")

    def wait_for_model_runner(self, *, required_backend: str | None = None) -> CommandResult:
        """Poll CLI status, the TCP API, and optionally one backend."""
        deadline = time.monotonic() + self.settings.startup_timeout
        last = CommandResult((), 1, "", "No attempts made", 0.0)
        backend_required = required_backend is not None
        with self.console.status("[bold cyan]Waiting for Docker Model Runner…[/bold cyan]"):
            while time.monotonic() < deadline:
                last = self.run_docker("model", "status", check=False, timeout=20)
                backend_ok = not backend_required or (
                    required_backend is not None
                    and backend_is_running(last.combined_output, required_backend)
                )
                if last.returncode == 0 and backend_ok and self.dmr_api_available():
                    detail = f"127.0.0.1:{self.settings.dmr_port}"
                    if required_backend is not None:
                        detail += f"; {required_backend} running"
                    self.reporter.passed("Model Runner TCP API", detail)
                    return last
                time.sleep(2)
        requirement = f" and backend {required_backend!r}" if required_backend is not None else ""
        raise StackError(
            f"Docker Model Runner{requirement} did not become healthy before the startup timeout.",
            hint="Inspect Docker Desktop → Settings → AI and run `docker model logs`.",
            details=last.combined_output,
        )

    def local_model_ids(self) -> tuple[str, ...]:
        """List local Model Runner model references."""
        result = self.run_docker("model", "list", "--quiet", timeout=60)
        return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())

    def api_model_ids(self, *, retries: int) -> tuple[str, ...]:
        """Fetch model IDs from the OpenAI-compatible API."""
        last_error = "No response"
        for _attempt in range(retries):
            try:
                response = httpx.get(self.dmr_url("models"), timeout=15.0, trust_env=False)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("response is not an object")
                data = payload.get("data")
                if not isinstance(data, list):
                    raise ValueError("response.data is not a list")
                identifiers = tuple(
                    str(item["id"])
                    for item in data
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                )
                if identifiers:
                    return identifiers
                last_error = "API model list was empty"
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                last_error = str(exc)
            time.sleep(1)
        raise StackError(
            "Could not obtain a usable model list from Docker Model Runner.",
            hint="Run `docker model list` and `docker model logs`.",
            details=last_error,
        )

    def ensure_model(self) -> tuple[str, str]:
        """Select an installed candidate or pull the first working quantization."""
        local = self.local_model_ids()
        candidates = self.settings.model_candidates()
        selected_ref: str | None = None
        for requested in candidates:
            selected_ref = choose_model_id(requested, local)
            if selected_ref is not None:
                if requested != candidates[0]:
                    self.reporter.warn("Model quantization", f"Using installed fallback {selected_ref}")
                break
        errors: list[str] = []
        if selected_ref is None and self.settings.skip_pull:
            raise StackError(
                "No requested LFM2.5-VL model is installed and pulling is disabled.",
                hint="Rerun without --skip-pull.",
            )
        if selected_ref is None:
            for index, candidate in enumerate(candidates):
                started = time.monotonic()
                with self.console.status(f"[bold cyan]Pulling {candidate}…[/bold cyan]"):
                    result = self.run_docker("model", "pull", candidate, check=False, timeout=1800)
                if result.returncode == 0:
                    selected_ref = candidate
                    self.reporter.passed(
                        "Model pull",
                        candidate,
                        duration_seconds=time.monotonic() - started,
                    )
                    if index > 0:
                        self.reporter.warn("Model fallback", f"Preferred pull failed; using {candidate}")
                    break
                errors.append(f"{candidate}:\n{tail_text(result.combined_output, 3000)}")
                self.reporter.warn("Model pull attempt", f"Could not pull {candidate}; trying next fallback")
        if selected_ref is None:
            raise StackError(
                "All model pull candidates failed.",
                hint="Check network access and Docker Model Runner logs.",
                details="\n\n".join(errors),
            )
        api_ids = self.api_model_ids(retries=30)
        selected_id = choose_model_id(selected_ref, api_ids)
        if selected_id is None:
            raise StackError(
                "The model was pulled but is missing from the OpenAI-compatible model list.",
                hint="Run `docker model list` and `docker model logs`, then retry.",
                details=f"Requested: {selected_ref}\nAPI models: {api_ids}",
            )
        self.reporter.passed("Model available", selected_id)
        return selected_ref, selected_id

    def configure_model(self, model_ref: str) -> None:
        """Persist context and model-author sampling defaults in llama.cpp."""
        context = self.settings.effective_context_size
        context_result = self.run_docker(
            "model",
            "configure",
            "--context-size",
            str(context),
            model_ref,
            check=False,
            timeout=120,
        )
        if context_result.returncode != 0:
            raise StackError(
                f"Could not configure the model context to {context}.",
                hint="Update Docker Desktop and inspect `docker model configure --help`.",
                details=context_result.combined_output,
            )
        runtime_result = self.run_docker(
            "model",
            "configure",
            model_ref,
            "--",
            "--temp",
            "0.1",
            "--min-p",
            "0.15",
            "--repeat-penalty",
            "1.05",
            check=False,
            timeout=120,
        )
        if runtime_result.returncode != 0:
            raise StackError(
                "Could not configure the model's llama.cpp sampling defaults.",
                hint="Update Docker Desktop or rerun with a supported Model Runner release.",
                details=runtime_result.combined_output,
            )
        self.reporter.passed(
            "Model configuration",
            f"context={context}; temperature=0.1; min-p=0.15; repeat-penalty=1.05",
        )

    def post_chat(self, payload: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
        """POST one OpenAI-compatible chat request and validate the response object."""
        try:
            response = httpx.post(
                self.dmr_url("chat/completions"),
                json=payload,
                timeout=timeout,
                trust_env=False,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = tail_text(exc.response.text, 6000)
            raise StackError(
                f"Model API returned HTTP {exc.response.status_code}.",
                details=body,
            ) from exc
        except httpx.HTTPError as exc:
            raise StackError("Model API request failed.", details=str(exc)) from exc
        try:
            parsed = response.json()
        except ValueError as exc:
            raise StackError("Model API returned invalid JSON.", details=tail_text(response.text)) from exc
        if not isinstance(parsed, dict):
            raise StackError("Model API returned a non-object JSON response.", details=response.text)
        return cast(dict[str, Any], parsed)

    def smoke_test_text(self, model_id: str) -> None:
        """Run a deterministic text inference smoke test."""
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly the single word READY.",
                }
            ],
            "temperature": 0,
            "max_tokens": 16,
            "stream": False,
        }
        started = time.monotonic()
        response = self.post_chat(payload, timeout=max(180.0, float(self.settings.command_timeout)))
        text = extract_assistant_text(response)
        if not text:
            raise StackError(
                "Text inference returned no assistant content.",
                details=tail_text(json.dumps(response, default=str), 6000),
            )
        if "ready" not in text.lower():
            self.reporter.warn(
                "Text inference",
                f"Model responded successfully but did not follow the exact-word probe: {text[:160]!r}",
            )
        else:
            self.reporter.passed(
                "Text inference",
                f"Received {text[:80]!r}",
                duration_seconds=time.monotonic() - started,
            )

    def smoke_test_vision(self, model_id: str) -> None:
        """Run a real image-input test using an internally generated red PNG."""
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the dominant color in this image? Reply with one color word.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": solid_png_data_uri()},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 24,
            "stream": False,
        }
        started = time.monotonic()
        response = self.post_chat(payload, timeout=max(240.0, float(self.settings.command_timeout)))
        text = extract_assistant_text(response)
        if not text:
            raise StackError(
                "Vision inference returned no assistant content.",
                details=tail_text(json.dumps(response, default=str), 6000),
            )
        if re.search(r"\bred\b", text, flags=re.IGNORECASE):
            self.reporter.passed(
                "Vision inference",
                f"Generated-image test identified red: {text[:120]!r}",
                duration_seconds=time.monotonic() - started,
            )
            return
        message = f"Image request succeeded but the red test image was described as {text[:160]!r}"
        if self.settings.strict_vision:
            raise StackError(message, hint="Try --profile quality or inspect Model Runner logs.")
        self.reporter.warn(
            "Vision inference",
            message,
            hint="Transport and multimodal parsing worked; use --strict-vision to make semantic mismatch fatal.",
        )

    def check_metal_acceleration(self) -> None:
        """Look for Metal/GPU evidence in Model Runner logs after inference."""
        if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
            self.reporter.skipped("Metal acceleration", "Host is not native Apple Silicon macOS")
            return
        logs = self.run_docker("model", "logs", check=False, timeout=30)
        text = logs.combined_output
        metal_patterns = (
            r"(?i)\bmetal\b",
            r"(?i)ggml_metal",
            r"(?i)apple\s+gpu",
            r"(?i)gpu.*offload",
        )
        if logs.returncode == 0 and any(re.search(pattern, text) for pattern in metal_patterns):
            evidence = next(
                (line.strip() for line in text.splitlines() if re.search(r"(?i)metal|apple\s+gpu|gpu.*offload", line)),
                "Metal/GPU log entry detected",
            )
            self.reporter.passed("Metal acceleration", tail_text(evidence, 240))
        else:
            self.reporter.warn(
                "Metal acceleration",
                "Inference worked, but current logs contained no explicit Metal marker",
                hint="Run `docker model logs` and look for Metal/GPU initialization messages.",
            )

    def inspect_network(self, name: str) -> dict[str, Any] | None:
        """Return Docker network inspect data or None when it is absent."""
        result = self.run_docker("network", "inspect", name, check=False, timeout=30)
        if result.returncode != 0:
            combined = result.combined_output.lower()
            if "no such network" in combined or "not found" in combined:
                return None
            raise StackError(f"Could not inspect Docker network {name}.", details=result.combined_output)
        try:
            parsed: object = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise StackError("Docker network inspect returned invalid JSON.", details=result.stdout) from exc
        if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], dict):
            raise StackError("Docker network inspect returned an unexpected structure.", details=result.stdout)
        return cast(dict[str, Any], parsed[0])

    def ensure_managed_network(self) -> None:
        """Create or validate the private network shared by UI and proxy."""
        inspect = self.inspect_network(NETWORK_NAME)
        if inspect is None:
            result = self.run_docker(
                "network",
                "create",
                "--label",
                f"{MANAGED_LABEL}=true",
                "--label",
                f"{VERSION_LABEL}={APP_VERSION}",
                NETWORK_NAME,
                timeout=60,
            )
            self.reporter.passed("Managed network", result.stdout.strip() or NETWORK_NAME)
            return
        if network_labels(inspect).get(MANAGED_LABEL) != "true":
            raise StackError(
                f"Docker network name {NETWORK_NAME!r} is already used by an unmanaged network.",
                hint="Rename or remove that network; the script will not modify unknown resources.",
            )
        self.reporter.passed("Managed network", f"Using existing {NETWORK_NAME}")

    def python_interpreter_for_image(self, image: str) -> str:
        """Find a Python executable inside the selected Open WebUI image."""
        probe = self.run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            image,
            "-c",
            "command -v python3 || command -v python",
            check=False,
            timeout=60,
        )
        interpreter = first_nonempty_line(probe.stdout)
        if probe.returncode != 0 or not interpreter:
            raise StackError(
                f"Could not find Python inside {image} for the compatibility proxy.",
                details=probe.combined_output,
            )
        return interpreter

    def run_python_in_container(self, container: str, code: str, *, timeout: int) -> CommandResult:
        """Run Python in a container while tolerating python/python3 naming differences."""
        attempts: list[tuple[str, CommandResult]] = []
        for interpreter in ("python3", "python"):
            attempt = self.run_docker(
                "exec",
                container,
                interpreter,
                "-c",
                code,
                check=False,
                timeout=timeout,
            )
            attempts.append((interpreter, attempt))
            if attempt.returncode == 0:
                return attempt
        details = "\n\n".join(
            f"{interpreter} probe:\n{attempt.combined_output}" for interpreter, attempt in attempts
        )
        raise StackError(
            f"Could not run the required Python probe in container {container!r}.",
            details=details,
        )

    def proxy_config_sha(self, image: str) -> str:
        """Hash compatibility-proxy configuration and source."""
        payload = {
            "version": APP_VERSION,
            "image": image,
            "network": NETWORK_NAME,
            "port": PROXY_PORT,
            "upstream": f"http://host.docker.internal:{self.settings.dmr_port}/engines/v1",
            "source_sha256": hashlib.sha256(PROXY_SOURCE.encode()).hexdigest(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def ensure_compat_proxy(self, image: str, model_id: str) -> None:
        """Run a local proxy that removes llama.cpp grammar-producing fields."""
        self.ensure_managed_network()
        atomic_write_text(PROXY_SCRIPT_FILE, PROXY_SOURCE, mode=0o644)
        config_sha = self.proxy_config_sha(image)
        existing = self.inspect_container(PROXY_CONTAINER_NAME)
        if existing is not None:
            self.require_managed_container(existing)
            labels = container_labels(existing)
            config = existing.get("Config")
            same_image = str(config.get("Image", "")) == image if isinstance(config, dict) else False
            same_config = labels.get(CONFIG_LABEL) == config_sha
            on_network = NETWORK_NAME in container_network_names(existing)
            if same_config and same_image and on_network and not self.settings.refresh_image:
                if container_running(existing):
                    self.reporter.passed("DMR compatibility proxy", "Existing configuration is current")
                else:
                    self.reporter.step(
                        "Start DMR compatibility proxy",
                        lambda: self.run_docker("start", PROXY_CONTAINER_NAME, timeout=60),
                        lambda _result: "Started existing managed container",
                    )
                self.wait_for_compat_proxy()
                self.verify_compat_proxy_model_list(model_id)
                self.verify_compat_proxy_streaming_chat(model_id)
                return
            self.reporter.warn("Proxy reconciliation", "Compatibility configuration changed; recreating proxy")
            removed = self.run_docker("rm", "--force", PROXY_CONTAINER_NAME, check=False, timeout=60)
            if removed.returncode != 0:
                raise StackError("Could not replace the DMR compatibility proxy.", details=removed.combined_output)
        interpreter = self.python_interpreter_for_image(image)
        command = (
            "run",
            "--detach",
            "--name",
            PROXY_CONTAINER_NAME,
            "--restart",
            "unless-stopped",
            "--label",
            f"{MANAGED_LABEL}=true",
            "--label",
            f"{CONFIG_LABEL}={config_sha}",
            "--label",
            f"{VERSION_LABEL}={APP_VERSION}",
            "--network",
            NETWORK_NAME,
            "--add-host",
            "host.docker.internal:host-gateway",
            "--mount",
            f"type=bind,src={PROXY_SCRIPT_FILE},dst=/opt/lfm/dmr-compat-proxy.py,readonly",
            "--env",
            f"UPSTREAM_BASE=http://host.docker.internal:{self.settings.dmr_port}/engines/v1",
            "--env",
            f"LISTEN_PORT={PROXY_PORT}",
            "--entrypoint",
            interpreter,
            image,
            "-u",
            "/opt/lfm/dmr-compat-proxy.py",
        )
        created = self.run_docker(*command, check=False, timeout=180)
        if created.returncode != 0:
            raise StackError(
                "Could not create the DMR compatibility proxy.",
                hint="Inspect Docker bind-mount permissions and the selected Open WebUI image.",
                details=created.combined_output,
            )
        self.reporter.passed("DMR compatibility proxy", f"Created {PROXY_CONTAINER_NAME}")
        self.wait_for_compat_proxy()
        self.verify_compat_proxy_model_list(model_id)
        self.verify_compat_proxy_streaming_chat(model_id)

    def wait_for_compat_proxy(self) -> None:
        """Poll the private proxy health endpoint until it is ready."""
        deadline = time.monotonic() + self.settings.startup_timeout
        last_error = "No attempts made"
        code = (
            "import urllib.request; "
            f"r=urllib.request.urlopen('http://127.0.0.1:{PROXY_PORT}/health', timeout=5); "
            "print(r.status, r.read().decode())"
        )
        with self.console.status("[bold cyan]Waiting for grammar-safe compatibility proxy…[/bold cyan]"):
            while time.monotonic() < deadline:
                inspect = self.inspect_container(PROXY_CONTAINER_NAME)
                if inspect is None:
                    raise StackError("The DMR compatibility proxy disappeared during startup.")
                if not container_running(inspect):
                    logs = self.run_docker(
                        "logs", "--tail", "300", PROXY_CONTAINER_NAME, check=False, timeout=30
                    )
                    raise StackError("The DMR compatibility proxy exited during startup.", details=logs.combined_output)
                for interpreter in ("python3", "python"):
                    attempt = self.run_docker(
                        "exec",
                        PROXY_CONTAINER_NAME,
                        interpreter,
                        "-c",
                        code,
                        check=False,
                        timeout=10,
                    )
                    if attempt.returncode == 0:
                        self.reporter.passed("Compatibility proxy health", f"Private port {PROXY_PORT}; HTTP 200")
                        return
                    last_error = attempt.combined_output
                time.sleep(1)
        logs = self.run_docker("logs", "--tail", "300", PROXY_CONTAINER_NAME, check=False, timeout=30)
        raise StackError(
            "The DMR compatibility proxy did not become healthy before the startup timeout.",
            details=f"Probe: {last_error}\n\nLogs:\n{logs.combined_output}",
        )

    def verify_compat_proxy_model_list(self, model_id: str) -> None:
        """Verify the proxy preserves the Model Runner model-list response."""
        code = (
            "import json, urllib.request; "
            f"u='http://127.0.0.1:{PROXY_PORT}/v1/models'; "
            "d=json.load(urllib.request.urlopen(u, timeout=15)); "
            "print(json.dumps([x.get('id') for x in d.get('data', [])]))"
        )
        result = self.run_python_in_container(PROXY_CONTAINER_NAME, code, timeout=40)
        try:
            parsed: object = json.loads(result.stdout.strip())
        except json.JSONDecodeError as exc:
            raise StackError("Compatibility proxy returned an invalid model list.", details=result.stdout) from exc
        available = tuple(str(item) for item in parsed) if isinstance(parsed, list) else ()
        if choose_model_id(model_id, available) is None:
            raise StackError(
                "Compatibility proxy cannot see the selected model.",
                details=f"Wanted: {model_id}\nVisible: {available}",
            )
        self.reporter.passed("Proxy → model list", model_id)

    def verify_compat_proxy_streaming_chat(self, model_id: str) -> None:
        """Exercise an Open WebUI-style streaming request with invalid grammar controls."""
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with exactly the single word READY."}],
            "temperature": 0,
            "max_tokens": 16,
            "stream": True,
            "grammar": "root ::= [",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "intentionally_invalid_probe",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"value": {"type": "string", "pattern": "["}},
                    },
                },
            },
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "Compatibility probe only",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        encoded_payload = json.dumps(payload, separators=(",", ":"))
        code = (
            "import json, urllib.request; "
            f"p={encoded_payload!r}.encode(); "
            f"q=urllib.request.Request('http://127.0.0.1:{PROXY_PORT}/v1/chat/completions', data=p, "
            "headers={'Content-Type':'application/json'}, method='POST'); "
            "r=urllib.request.urlopen(q, timeout=240); removed=r.headers.get('X-LFM-Compat-Removed',''); "
            "parts=[]; events=0; "
            "\nfor raw in r:\n"
            " line=raw.decode('utf-8','replace').strip()\n"
            " if not line.startswith('data:'): continue\n"
            " data=line[5:].strip()\n"
            " if data=='[DONE]': break\n"
            " events+=1\n"
            " try: obj=json.loads(data)\n"
            " except json.JSONDecodeError: continue\n"
            " for choice in obj.get('choices',[]):\n"
            "  delta=choice.get('delta') or choice.get('message') or {}\n"
            "  content=delta.get('content','') if isinstance(delta,dict) else ''\n"
            "  if isinstance(content,str): parts.append(content)\n"
            "  elif isinstance(content,list): parts.extend(str(x.get('text','')) for x in content if isinstance(x,dict))\n"
            "\nprint(json.dumps({'status':r.status,'removed':removed.split(',') if removed else [],'events':events,'text':''.join(parts)}))"
        )
        started = time.monotonic()
        result = self.run_python_in_container(
            PROXY_CONTAINER_NAME,
            code,
            timeout=max(300, self.settings.command_timeout),
        )
        output = first_nonempty_line(result.stdout.splitlines()[-1] if result.stdout.splitlines() else "")
        parsed = json_object(output, context="compatibility-proxy streaming probe")
        removed_value = parsed.get("removed")
        removed = {str(item) for item in removed_value} if isinstance(removed_value, list) else set()
        required = {"grammar", "response_format", "tools", "tool_choice", "parallel_tool_calls"}
        missing = sorted(required - removed)
        if missing:
            raise StackError(
                "Compatibility proxy did not remove all grammar-producing request fields.",
                details=f"Missing: {missing}; response: {parsed}",
            )
        text = str(parsed.get("text", "")).strip()
        events = parsed.get("events")
        if not text or not isinstance(events, int) or events <= 0:
            raise StackError(
                "Grammar-safe streaming probe returned no assistant content.",
                details=json.dumps(parsed, indent=2),
            )
        if "ready" not in text.lower():
            self.reporter.warn(
                "Grammar-safe streaming",
                f"Request streamed successfully after sanitization but returned {text[:160]!r}",
            )
        else:
            self.reporter.passed(
                "Grammar-safe streaming",
                f"Removed {', '.join(sorted(required))}; received {text[:80]!r}",
                duration_seconds=time.monotonic() - started,
            )

    def pull_webui_image(self) -> str:
        """Ensure the preferred Open WebUI image exists, with a controlled fallback."""
        requested = self.settings.webui_image
        inspect = self.run_docker("image", "inspect", requested, check=False, timeout=30)
        needs_pull = self.settings.refresh_image or inspect.returncode != 0
        if not needs_pull:
            self.reporter.passed("Open WebUI image", f"Using cached {requested}")
            return requested
        started = time.monotonic()
        with self.console.status(f"[bold cyan]Pulling {requested}…[/bold cyan]"):
            pull = self.run_docker("pull", requested, check=False, timeout=1800)
        if pull.returncode == 0:
            self.reporter.passed(
                "Open WebUI image",
                requested,
                duration_seconds=time.monotonic() - started,
            )
            return requested
        if requested != DEFAULT_WEBUI_IMAGE or self.settings.no_fallback:
            raise StackError(
                f"Could not pull Open WebUI image {requested}.",
                hint="Check registry connectivity or choose another --webui-image.",
                details=pull.combined_output,
            )
        self.reporter.warn(
            "Open WebUI image",
            f"Pinned {requested} could not be pulled; trying {FALLBACK_WEBUI_IMAGE}",
        )
        fallback = self.run_docker("pull", FALLBACK_WEBUI_IMAGE, check=False, timeout=1800)
        if fallback.returncode != 0:
            raise StackError(
                "Both pinned and fallback Open WebUI image pulls failed.",
                details=f"Pinned:\n{pull.combined_output}\n\nFallback:\n{fallback.combined_output}",
            )
        self.reporter.passed("Open WebUI image", FALLBACK_WEBUI_IMAGE)
        return FALLBACK_WEBUI_IMAGE

    def openwebui_environment(self, model_id: str, secret: str) -> dict[str, str]:
        """Build conservative vision-chat defaults for a very small local VLM."""
        size = self.settings.effective_image_size
        capabilities = {
            "vision": True,
            "file_upload": True,
            "file_context": True,
            "builtin_tools": False,
            "web_search": False,
            "image_generation": False,
            "code_interpreter": False,
            "citations": False,
        }
        return {
            "ENABLE_OLLAMA_API": "false",
            "ENABLE_OPENAI_API": "true",
            "OPENAI_API_BASE_URL": f"http://{PROXY_CONTAINER_NAME}:{PROXY_PORT}/v1",
            "OPENAI_API_KEY": "not-needed",
            "WEBUI_AUTH": str(self.settings.effective_auth).lower(),
            "WEBUI_SECRET_KEY": secret,
            "DEFAULT_MODELS": model_id,
            "DEFAULT_MODEL_METADATA": json.dumps(
                {"capabilities": capabilities},
                separators=(",", ":"),
            ),
            "DEFAULT_MODEL_PARAMS": json.dumps(
                {"temperature": 0.1, "function_calling": "legacy"},
                separators=(",", ":"),
            ),
            "FILE_IMAGE_COMPRESSION_WIDTH": str(size),
            "FILE_IMAGE_COMPRESSION_HEIGHT": str(size),
            "ENABLE_PERSISTENT_CONFIG": str(self.settings.allow_ui_config).lower(),
            "ENABLE_AUTOCOMPLETE_GENERATION": "false",
            "ENABLE_FOLLOW_UP_GENERATION": "false",
            "ENABLE_TITLE_GENERATION": "false",
            "ENABLE_TAGS_GENERATION": "false",
            "ENABLE_SEARCH_QUERY_GENERATION": "false",
            "ENABLE_RETRIEVAL_QUERY_GENERATION": "false",
            "WEBUI_NAME": "LFM2.5-VL",
        }

    def openwebui_config_sha(
        self,
        *,
        model_id: str,
        image: str,
        port: int,
        environment: Mapping[str, str],
    ) -> str:
        """Hash managed configuration without storing raw secrets in labels."""
        safe_environment = dict(environment)
        secret = safe_environment.pop("WEBUI_SECRET_KEY", "")
        safe_environment["WEBUI_SECRET_KEY_SHA256"] = hashlib.sha256(secret.encode()).hexdigest()
        payload = {
            "version": APP_VERSION,
            "model_id": model_id,
            "image": image,
            "port": port,
            "bind_host": self.settings.bind_host,
            "environment": safe_environment,
            "volume": VOLUME_NAME,
            "network": NETWORK_NAME,
            "proxy": PROXY_CONTAINER_NAME,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def ensure_openwebui(self, model_id: str, model_ref: str) -> None:
        """Reconcile Open WebUI plus its grammar-safe Model Runner proxy."""
        image = self.pull_webui_image()
        self.webui_image = image
        self.ensure_compat_proxy(image, model_id)
        secret = stable_secret()
        self.runner.secrets.append(secret)
        existing = self.inspect_container(CONTAINER_NAME)
        existing_port: int | None = None
        if existing is not None:
            self.require_managed_container(existing)
            existing_port = container_host_port(existing)
        preferred_port = self.settings.port
        if existing_port is not None and preferred_port == DEFAULT_WEBUI_PORT and not self.settings.strict_port:
            port = existing_port
        else:
            if existing is not None and existing_port == preferred_port:
                port = preferred_port
            else:
                port = free_tcp_port(self.settings.bind_host, preferred_port, strict=self.settings.strict_port)
        self.webui_port = port
        if port != preferred_port:
            self.reporter.warn("Open WebUI port", f"Port {preferred_port} was busy; selected {port}")
        else:
            self.reporter.passed("Open WebUI port", str(port))
        environment = self.openwebui_environment(model_id, secret)
        config_sha = self.openwebui_config_sha(
            model_id=model_id,
            image=image,
            port=port,
            environment=environment,
        )
        if existing is not None:
            labels = container_labels(existing)
            same_config = labels.get(CONFIG_LABEL) == config_sha
            config = existing.get("Config")
            same_image = str(config.get("Image", "")) == image if isinstance(config, dict) else False
            if same_config and same_image and not self.settings.refresh_image:
                if not container_running(existing):
                    self.reporter.step(
                        "Start Open WebUI",
                        lambda: self.run_docker("start", CONTAINER_NAME, timeout=60),
                        lambda _result: "Started existing managed container",
                    )
                else:
                    self.reporter.passed("Open WebUI container", "Existing configuration is current")
                self.wait_for_openwebui(port)
                self.verify_openwebui_container(port)
                self.verify_container_model_connection(model_id)
                self.save_runtime_state(model_ref, model_id, image, port, config_sha)
                return
            self.reporter.warn(
                "Open WebUI reconciliation",
                "Managed configuration changed; recreating container while retaining chats",
            )
            removed = self.run_docker("rm", "--force", CONTAINER_NAME, check=False, timeout=90)
            if removed.returncode != 0:
                raise StackError(
                    "Could not replace the existing managed Open WebUI container.",
                    details=removed.combined_output,
                )
        self.run_docker("volume", "create", VOLUME_NAME, timeout=60)
        command: list[str] = [
            "run",
            "--detach",
            "--name",
            CONTAINER_NAME,
            "--restart",
            "unless-stopped",
            "--label",
            f"{MANAGED_LABEL}=true",
            "--label",
            f"{CONFIG_LABEL}={config_sha}",
            "--label",
            f"{VERSION_LABEL}={APP_VERSION}",
            "--network",
            NETWORK_NAME,
            "--publish",
            f"{self.settings.bind_host}:{port}:8080",
            "--add-host",
            "host.docker.internal:host-gateway",
            "--volume",
            f"{VOLUME_NAME}:/app/backend/data",
        ]
        for key, value in sorted(environment.items()):
            command.extend(("--env", f"{key}={value}"))
        command.append(image)
        created = self.run_docker(*command, check=False, timeout=180)
        if created.returncode != 0:
            raise StackError(
                "Open WebUI container could not be created.",
                hint="Inspect Docker disk space, registry access, and managed network state.",
                details=created.combined_output,
            )
        self.reporter.passed("Open WebUI container", f"Created {CONTAINER_NAME}")
        try:
            self.wait_for_openwebui(port)
            self.verify_openwebui_container(port)
            self.verify_container_model_connection(model_id)
        except BaseException:
            cleanup = self.run_docker("rm", "--force", CONTAINER_NAME, check=False, timeout=60)
            if cleanup.returncode == 0:
                self.reporter.warn(
                    "Failed-container cleanup",
                    "Removed unhealthy Open WebUI container; persistent volume and proxy were retained",
                )
            raise
        self.save_runtime_state(model_ref, model_id, image, port, config_sha)

    def save_runtime_state(
        self,
        model_ref: str,
        model_id: str,
        image: str,
        port: int,
        config_sha: str,
    ) -> None:
        """Persist successful stack state."""
        previous = load_state()
        created_at = previous.created_at if previous is not None else utc_now()
        state = State(
            app_version=APP_VERSION,
            created_at=created_at,
            model_ref=model_ref,
            model_id=model_id,
            context_size=self.settings.effective_context_size,
            image_size=self.settings.effective_image_size,
            profile=self.settings.profile,
            webui_image=image,
            webui_port=port,
            dmr_port=self.settings.dmr_port,
            auth_enabled=self.settings.effective_auth,
            lan_enabled=self.settings.lan,
            config_sha=config_sha,
        )
        save_state(state)
        self.reporter.passed("State file", str(STATE_FILE))

    def wait_for_openwebui(self, port: int) -> None:
        """Poll Open WebUI's unauthenticated health endpoint."""
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.monotonic() + self.settings.startup_timeout
        last_error = "No attempts made"
        with self.console.status("[bold cyan]Waiting for Open WebUI…[/bold cyan]"):
            while time.monotonic() < deadline:
                inspect = self.inspect_container(CONTAINER_NAME)
                if inspect is None:
                    raise StackError("Open WebUI container disappeared during startup.")
                state = nested_mapping(inspect, "State") or {}
                if state.get("Running") is False:
                    logs = self.run_docker("logs", "--tail", "300", CONTAINER_NAME, check=False, timeout=30)
                    raise StackError(
                        "Open WebUI exited during startup.",
                        details=logs.combined_output,
                    )
                try:
                    response = httpx.get(url, timeout=5.0, trust_env=False)
                    if response.status_code == httpx.codes.OK:
                        self.reporter.passed("Open WebUI health", f"HTTP 200 on port {port}")
                        return
                    last_error = f"HTTP {response.status_code}: {tail_text(response.text, 1000)}"
                except httpx.HTTPError as exc:
                    last_error = str(exc)
                time.sleep(2)
        logs = self.run_docker("logs", "--tail", "300", CONTAINER_NAME, check=False, timeout=30)
        raise StackError(
            "Open WebUI did not become healthy before the startup timeout.",
            hint="Review the diagnostic bundle and Open WebUI logs.",
            details=f"Health: {last_error}\n\nLogs:\n{logs.combined_output}",
        )

    def inspect_container(self, name: str) -> dict[str, Any] | None:
        """Return Docker inspect data for one container, or None if absent."""
        result = self.run_docker("inspect", name, check=False, timeout=30)
        if result.returncode != 0:
            combined = result.combined_output.lower()
            if "no such" in combined or "not found" in combined:
                return None
            raise StackError(f"Could not inspect container {name}.", details=result.combined_output)
        try:
            parsed: object = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise StackError("Docker inspect returned invalid JSON.", details=result.stdout) from exc
        if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], dict):
            raise StackError("Docker inspect returned an unexpected structure.", details=result.stdout)
        return cast(dict[str, Any], parsed[0])

    def require_managed_container(self, inspect: Mapping[str, Any]) -> None:
        """Refuse to alter a name-colliding container not created by this script."""
        name = str(inspect.get("Name", "container")).lstrip("/") or "container"
        if container_labels(inspect).get(MANAGED_LABEL) != "true":
            raise StackError(
                f"Container name {name!r} is already used by an unmanaged container.",
                hint="Rename or remove that container, then rerun the script.",
            )

    def verify_openwebui_container(self, port: int) -> None:
        """Check restart policy, volume, binding, network, and ownership."""
        inspect = self.inspect_container(CONTAINER_NAME)
        if inspect is None:
            raise StackError("Open WebUI container vanished after passing health checks.")
        self.require_managed_container(inspect)
        host_config = nested_mapping(inspect, "HostConfig") or {}
        restart = host_config.get("RestartPolicy")
        restart_name = str(restart.get("Name", "")) if isinstance(restart, dict) else ""
        if restart_name != "unless-stopped":
            raise StackError(f"Unexpected Open WebUI restart policy: {restart_name or 'missing'}")
        self.reporter.passed("Restart policy", restart_name)
        mounts = inspect.get("Mounts")
        mounted = False
        if isinstance(mounts, list):
            mounted = any(
                isinstance(item, dict)
                and item.get("Name") == VOLUME_NAME
                and item.get("Destination") == "/app/backend/data"
                for item in mounts
            )
        if not mounted:
            raise StackError("Open WebUI's persistent data volume is not mounted as expected.")
        self.reporter.passed("Persistent storage", VOLUME_NAME)
        networks = container_network_names(inspect)
        if NETWORK_NAME not in networks:
            raise StackError(
                f"Open WebUI is not attached to the managed network {NETWORK_NAME!r}.",
                hint="Rerun the default up command to recreate the managed container.",
            )
        self.reporter.passed("Managed network", NETWORK_NAME)
        actual_port = container_host_port(inspect)
        if actual_port != port:
            raise StackError(f"Expected Open WebUI host port {port}, Docker reports {actual_port}.")
        host_ip = container_host_ip(inspect)
        if self.settings.lan:
            if host_ip not in {"0.0.0.0", "::"}:
                raise StackError(f"LAN mode expected a wildcard bind, Docker reports {host_ip!r}.")
            self.reporter.warn("Network exposure", f"LAN-accessible on port {port}; authentication forced on")
        else:
            if host_ip not in {"127.0.0.1", "::1"}:
                raise StackError(
                    f"Expected loopback-only binding, Docker reports {host_ip!r}.",
                    hint="Do not use the stack until the unexpected exposure is corrected.",
                )
            self.reporter.passed("Network exposure", f"Loopback only at {host_ip}:{port}")

    def verify_container_model_connection(self, model_id: str) -> None:
        """Verify Open WebUI can see the model through the compatibility proxy."""
        code = (
            "import json, urllib.request; "
            f"u='http://{PROXY_CONTAINER_NAME}:{PROXY_PORT}/v1/models'; "
            "r=urllib.request.urlopen(u, timeout=15); d=json.load(r); "
            "print(json.dumps({'proxy':r.headers.get('X-LFM-Compat-Proxy'),"
            "'ids':[x.get('id') for x in d.get('data', [])]}))"
        )
        successful = self.run_python_in_container(CONTAINER_NAME, code, timeout=40)
        try:
            parsed: object = json.loads(successful.stdout.strip())
        except json.JSONDecodeError as exc:
            raise StackError(
                "Container model probe returned invalid JSON.",
                details=successful.stdout,
            ) from exc
        if not isinstance(parsed, dict):
            raise StackError("Container model probe returned an unexpected structure.", details=successful.stdout)
        identifiers = parsed.get("ids")
        available = tuple(str(item) for item in identifiers) if isinstance(identifiers, list) else ()
        if parsed.get("proxy") != "1":
            raise StackError("Open WebUI's model probe bypassed the compatibility proxy.")
        if choose_model_id(model_id, available) is None:
            raise StackError(
                "Open WebUI can reach the compatibility proxy but cannot see the selected model.",
                details=f"Wanted: {model_id}\nVisible: {available}",
            )
        self.reporter.passed("Open WebUI → proxy → model", model_id)

    def check_existing_stack(self, *, deep: bool) -> None:
        """Inspect an existing managed UI and optionally run live model tests."""
        state = load_state()
        if state is not None:
            self.reporter.passed("State file", f"Model {state.model_id}; port {state.webui_port}")
            self.webui_port = state.webui_port
            self.model_id = state.model_id
            self.model_ref = state.model_ref
        else:
            self.reporter.warn("State file", "No valid managed state; run the default up command")
        proxy = self.inspect_container(PROXY_CONTAINER_NAME)
        proxy_running = False
        if proxy is None:
            self.reporter.warn(
                "DMR compatibility proxy",
                "Not created; old direct Open WebUI requests may trigger llama.cpp grammar failures",
                hint="Run the default up command to install the compatibility repair.",
            )
        else:
            self.require_managed_container(proxy)
            proxy_running = container_running(proxy)
            if proxy_running:
                self.reporter.passed("DMR compatibility proxy", "Running")
            else:
                self.reporter.warn(
                    "DMR compatibility proxy",
                    "Stopped",
                    hint="Run the default up command to start it.",
                )
        inspect = self.inspect_container(CONTAINER_NAME)
        if inspect is None:
            self.reporter.warn("Open WebUI container", "Not created")
            return
        self.require_managed_container(inspect)
        if not container_running(inspect):
            self.reporter.warn(
                "Open WebUI container",
                "Stopped",
                hint="Run the default up command to start it.",
            )
            return
        self.reporter.passed("Open WebUI container", "Running")
        port = container_host_port(inspect)
        if port is None:
            raise StackError("Open WebUI container has no published 8080/tcp port.")
        self.webui_port = port
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=10.0, trust_env=False)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise StackError("Open WebUI health endpoint failed.", details=str(exc)) from exc
        self.reporter.passed("Open WebUI health", f"HTTP 200 on port {port}")
        self.verify_openwebui_container(port)
        if state is not None and proxy_running:
            self.verify_container_model_connection(state.model_id)
            if deep:
                self.smoke_test_text(state.model_id)
                self.smoke_test_vision(state.model_id)
                self.verify_compat_proxy_streaming_chat(state.model_id)
                self.check_metal_acceleration()
        elif deep and state is None:
            self.reporter.warn("Deep model tests", "Skipped because no valid state file identifies the model")
        elif deep and not proxy_running:
            self.reporter.warn("Grammar-safe streaming test", "Skipped because the compatibility proxy is not running")

    def show_ready(self) -> None:
        """Show connection details and open the browser when requested."""
        model_id = self.model_id or self.settings.requested_model
        url = f"http://localhost:{self.webui_port}"
        auth_note = "Create the first admin account" if self.settings.effective_auth else "Local auth disabled"
        table = Table(title="Ready", show_header=False, box=None)
        table.add_row("Open WebUI", f"[bold cyan]{url}[/bold cyan]")
        table.add_row("Model", model_id)
        native_apple = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
        backend = (
            "Open WebUI → grammar-safe proxy → Docker Model Runner → llama.cpp → Metal"
            if native_apple
            else "Open WebUI → grammar-safe proxy → Docker Model Runner → llama.cpp"
        )
        table.add_row("Backend", backend)
        table.add_row("Images", f"Vision enabled; uploads compressed to ≤{self.settings.effective_image_size}px")
        table.add_row("Compatibility", "Text, images, and streaming preserved; native tool/structured-output schemas removed")
        table.add_row("Security", auth_note)
        table.add_row("State", str(STATE_FILE))
        self.console.print(table)
        if not self.settings.no_open:
            if webbrowser.open(url, new=2, autoraise=True):
                self.reporter.passed("Browser", f"Opened {url}")
            else:
                self.reporter.warn("Browser", f"Could not open automatically; visit {url}")

    def collect_diagnostics(self, error: BaseException) -> Path:
        """Write a redacted diagnostic bundle after failure."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = CONFIG_DIR / f"diagnostics-{stamp}.txt"
        sections: list[str] = [
            f"{APP_NAME} {APP_VERSION}",
            f"generated_at: {utc_now()}",
            f"command: {self.settings.command.value}",
            f"platform: {platform.platform()}",
            f"python: {sys.version}",
            f"error: {type(error).__name__}: {error}",
        ]
        if isinstance(error, StackError):
            if error.hint:
                sections.append(f"hint: {error.hint}")
            if error.details:
                sections.append(f"details:\n{error.details}")
        if self.docker is not None:
            commands: tuple[tuple[str, tuple[str, ...]], ...] = (
                ("docker version", self.docker_command("version")),
                ("docker info", self.docker_command("info")),
                ("docker model status", self.docker_command("model", "status")),
                ("docker model list", self.docker_command("model", "list", "--json")),
                ("docker model logs", self.docker_command("model", "logs")),
                ("Open WebUI inspect", self.docker_command("inspect", CONTAINER_NAME)),
                ("Open WebUI logs", self.docker_command("logs", "--tail", "300", CONTAINER_NAME)),
                ("proxy inspect", self.docker_command("inspect", PROXY_CONTAINER_NAME)),
                ("proxy logs", self.docker_command("logs", "--tail", "300", PROXY_CONTAINER_NAME)),
                ("managed network", self.docker_command("network", "inspect", NETWORK_NAME)),
            )
            for title, command in commands:
                try:
                    result = self.runner.run(command, timeout=30, check=False)
                    output = tail_text(result.combined_output, 20_000) or "<empty>"
                except StackError as exc:
                    output = f"diagnostic command failed: {exc}"
                sections.append(f"\n===== {title} =====\n{output}")
        checks_payload = [
            {
                "name": item.name,
                "level": item.level.value,
                "detail": item.detail,
                "duration_seconds": item.duration_seconds,
                "hint": item.hint,
            }
            for item in self.reporter.results
        ]
        sections.append("\n===== check results =====\n" + json.dumps(checks_payload, indent=2))
        content = redact_text("\n".join(sections), self.runner.secrets)
        atomic_write_text(path, content + "\n")
        return path


def container_labels(inspect: Mapping[str, Any]) -> dict[str, str]:
    """Return string container labels from Docker inspect data."""
    config = nested_mapping(inspect, "Config") or {}
    labels = config.get("Labels")
    if not isinstance(labels, dict):
        return {}
    return {str(key): str(value) for key, value in labels.items()}


def container_running(inspect: Mapping[str, Any]) -> bool:
    """Return Docker's running state."""
    state = nested_mapping(inspect, "State") or {}
    return state.get("Running") is True


def container_network_names(inspect: Mapping[str, Any]) -> tuple[str, ...]:
    """Return names of Docker networks attached to a container."""
    network = nested_mapping(inspect, "NetworkSettings") or {}
    networks = network.get("Networks")
    if not isinstance(networks, dict):
        return ()
    return tuple(sorted(str(name) for name in networks))


def network_labels(inspect: Mapping[str, Any]) -> dict[str, str]:
    """Return string labels from Docker network inspect data."""
    labels = inspect.get("Labels")
    if not isinstance(labels, dict):
        return {}
    return {str(key): str(value) for key, value in labels.items()}


def container_port_binding(inspect: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the first 8080/tcp host binding."""
    network = nested_mapping(inspect, "NetworkSettings") or {}
    ports = network.get("Ports")
    if not isinstance(ports, dict):
        return None
    bindings = ports.get("8080/tcp")
    if not isinstance(bindings, list) or not bindings or not isinstance(bindings[0], dict):
        return None
    return cast(dict[str, Any], bindings[0])


def container_host_port(inspect: Mapping[str, Any]) -> int | None:
    """Return the published host port for container port 8080."""
    binding = container_port_binding(inspect)
    if binding is None:
        return None
    raw = binding.get("HostPort")
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


def container_host_ip(inspect: Mapping[str, Any]) -> str:
    """Return the published host IP for container port 8080."""
    binding = container_port_binding(inspect)
    if binding is None:
        return ""
    return str(binding.get("HostIp", ""))


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap and deeply validate LFM2.5-VL with Docker Model Runner "
            "and Open WebUI on an Apple-Silicon Mac. With no command, runs up."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default=CommandName.UP.value,
        choices=[item.value for item in CommandName],
        help="Lifecycle command",
    )
    parser.add_argument("--profile", choices=[item.value for item in ProfileName], default=ProfileName.QUALITY.value)
    parser.add_argument("--model", help="Exact Docker Model Runner model reference; disables profile model selection")
    parser.add_argument("--context-size", type=int, help="Model context tokens (512–32768)")
    parser.add_argument("--image-size", type=int, help="Open WebUI image compression bound in pixels")
    parser.add_argument("--port", type=int, default=DEFAULT_WEBUI_PORT, help="Open WebUI host port")
    parser.add_argument("--dmr-port", type=int, default=DEFAULT_DMR_PORT, help="Docker Model Runner TCP port")
    parser.add_argument("--webui-image", default=DEFAULT_WEBUI_IMAGE, help="Open WebUI container image")
    parser.add_argument("--auth", action="store_true", help="Enable Open WebUI authentication")
    parser.add_argument("--lan", action="store_true", help="Expose Open WebUI on the LAN; forces authentication")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser")
    parser.add_argument("--skip-self-check", action="store_true", help="Skip Ruff/mypy/internal checks during up/test/doctor")
    parser.add_argument("--skip-pull", action="store_true", help="Never pull a missing model")
    parser.add_argument("--refresh-image", action="store_true", help="Pull and recreate Open WebUI even when cached")
    parser.add_argument("--strict-vision", action="store_true", help="Fail if the semantic red-image check is wrong")
    parser.add_argument("--strict-port", action="store_true", help="Fail instead of selecting a free port")
    parser.add_argument("--no-fallback", action="store_true", help="Do not try alternative model/image fallbacks")
    parser.add_argument(
        "--allow-unsupported-host",
        action="store_true",
        help="Permit deliberate non-macOS testing; does not make the stack supported",
    )
    parser.add_argument("--startup-timeout", type=int, default=240, help="Startup polling timeout in seconds")
    parser.add_argument("--command-timeout", type=int, default=120, help="Default subprocess timeout in seconds")
    parser.add_argument(
        "--allow-ui-config",
        action="store_true",
        help="Allow Open WebUI admin settings to override and persist over environment defaults",
    )
    parser.add_argument("--follow", action="store_true", help="Follow logs until interrupted (logs command)")
    parser.add_argument("--purge-data", action="store_true", help="Delete Open WebUI data volume (down command only)")
    parser.add_argument("--yes", action="store_true", help="Confirm destructive --purge-data")
    parser.add_argument("--verbose", action="store_true", help="Show redacted subprocess commands/output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    return parser


def settings_from_argv(argv: Sequence[str] | None = None) -> Settings:
    """Parse argparse output through Pydantic validation."""
    namespace = build_parser().parse_args(argv)
    values = vars(namespace)
    values["command"] = CommandName(values["command"])
    values["profile"] = ProfileName(values["profile"])
    command = cast(CommandName, values["command"])
    if values["purge_data"] and command is not CommandName.DOWN:
        raise StackError("--purge-data is valid only with the down command.")
    if values["follow"] and command is not CommandName.LOGS:
        raise StackError("--follow is valid only with the logs command.")
    try:
        return Settings.model_validate(values)
    except ValidationError as exc:
        raise StackError("Invalid command-line settings.", details=str(exc)) from exc


def run_internal_tests() -> None:
    """Exercise pure helpers and enforce typed function signatures."""
    assert human_bytes(1024) == "1.0 KiB"
    assert normalize_model_id("hf.co/LiquidAI/LFM2.5_VL:Q8_0") == "liquidai/lfm2.5-vl:q8-0"
    available = (
        "hf.co/LiquidAI/LFM2.5-VL-450M-GGUF:Q8_0",
        "other/model:tag",
    )
    assert choose_model_id("LiquidAI/LFM2.5-VL-450M-GGUF:Q8_0", available) == available[0]
    assert choose_model_id("missing/model:tag", available) is None
    uri = solid_png_data_uri(2, 2)
    assert uri.startswith("data:image/png;base64,")
    decoded = base64.b64decode(uri.split(",", 1)[1])
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")
    sample = {"choices": [{"message": {"content": " hello "}}]}
    assert extract_assistant_text(sample) == "hello"
    assert "supersecret" not in redact_text("WEBUI_SECRET_KEY=supersecret")
    redacted_json = redact_text(
        '{"Env":["WEBUI_SECRET_KEY=supersecret","OPENAI_API_KEY=alsosecret"]}'
    )
    assert "supersecret" not in redacted_json and "alsosecret" not in redacted_json
    assert isinstance(json.loads(redacted_json), dict)
    assert parse_version("Docker version 28.1.1, build test") == (28, 1, 1)
    legacy_status = (
        "Docker Model Runner is running\n\n"
        "Status:\n"
        "llama.cpp: running llama.cpp version: c22473b\n"
        "vllm: vLLM binary not found\n"
    )
    table_status = (
        "Docker Model Runner is running\n\n"
        "BACKEND    STATUS         DETAILS\n"
        "llama.cpp  Running        llama.cpp b9879-metal (sha256:abc) 72874f5\n"
        "diffusers  Not Installed\n"
        "mlx        Not Installed  package not installed\n"
        "vllm       Not Installed\n"
    )
    assert backend_is_running(legacy_status, "llama.cpp")
    assert backend_is_running(table_status, "llama.cpp")
    assert not backend_is_running(table_status, "vllm")
    compile(PROXY_SOURCE, "<dmr-compat-proxy>", "exec")
    for field in ("grammar", "response_format", "tools", "tool_choice", "parallel_tool_calls"):
        assert f'"{field}"' in PROXY_SOURCE
    assert "/chat/completions" in PROXY_SOURCE and "X-LFM-Compat-Removed" in PROXY_SOURCE
    proxy_namespace: dict[str, Any] = {"__name__": "lfm_dmr_compat_test"}
    exec(PROXY_SOURCE, proxy_namespace)
    sanitize = cast(Callable[[Any], tuple[Any, tuple[str, ...]]], proxy_namespace["sanitize_payload"])
    clean_value, removed = sanitize(
        {
            "messages": [{"role": "user", "content": "keep me"}],
            "grammar": "invalid",
            "tools": [],
            "extra_body": {"guided_json": {}, "keep": 1},
        }
    )
    assert isinstance(clean_value, dict)
    assert clean_value["messages"][0]["content"] == "keep me"
    assert clean_value["extra_body"] == {"keep": 1}
    assert {"grammar", "tools", "extra_body.guided_json"}.issubset(set(removed))
    defaults = settings_from_argv([])
    assert defaults.command is CommandName.UP
    assert defaults.profile is ProfileName.QUALITY
    fast = settings_from_argv(["--profile", "fast", "--no-open"])
    assert fast.command is CommandName.UP
    assert fast.profile is ProfileName.FAST
    status = settings_from_argv(["status"])
    assert status.command is CommandName.STATUS
    try:
        settings_from_argv(["status", "--purge-data"])
    except StackError:
        pass
    else:
        raise AssertionError("--purge-data must be rejected outside down")
    inspect_fixture: dict[str, Any] = {
        "Config": {"Labels": {MANAGED_LABEL: "true"}},
        "State": {"Running": True},
        "NetworkSettings": {
            "Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "3000"}]},
            "Networks": {NETWORK_NAME: {"NetworkID": "abc"}},
        },
    }
    assert container_running(inspect_fixture)
    assert container_labels(inspect_fixture)[MANAGED_LABEL] == "true"
    assert container_host_ip(inspect_fixture) == "127.0.0.1"
    assert container_host_port(inspect_fixture) == 3000
    assert container_network_names(inspect_fixture) == (NETWORK_NAME,)
    assert network_labels({"Labels": {MANAGED_LABEL: "true"}})[MANAGED_LABEL] == "true"
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    defaulted_lambda_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Lambda)
        and (
            bool(node.args.defaults)
            or any(default is not None for default in node.args.kw_defaults)
        )
    ]
    assert not defaulted_lambda_lines, (
        "Default-argument lambdas are incompatible with the strict mypy gate: "
        f"{defaulted_lambda_lines}"
    )
    missing: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for argument in arguments:
            if argument.arg not in {"self", "cls"} and argument.annotation is None:
                missing.append(f"{node.name}:{argument.arg}@{node.lineno}")
        if node.args.vararg is not None and node.args.vararg.annotation is None:
            missing.append(f"{node.name}:*{node.args.vararg.arg}@{node.lineno}")
        if node.args.kwarg is not None and node.args.kwarg.annotation is None:
            missing.append(f"{node.name}:**{node.args.kwarg.arg}@{node.lineno}")
        if node.returns is None:
            missing.append(f"{node.name}:return@{node.lineno}")
    assert not missing, f"Missing annotations: {missing}"
    metadata = source.split("# ///", 2)
    assert len(metadata) == 3 and "requires-python" in metadata[1] and "dependencies" in metadata[1]


def render_failure(console: Console, error: BaseException) -> None:
    """Render a concise actionable error panel."""
    if isinstance(error, StackError):
        lines = [f"[bold red]{error.message}[/bold red]"]
        if error.hint:
            lines.append(f"\n[bold]How to fix[/bold]\n{error.hint}")
        if error.details:
            lines.append(f"\n[bold]Details[/bold]\n{redact_text(tail_text(error.details, 6000))}")
        console.print(Panel("\n".join(lines), title="Setup failed", border_style="red"))
        return
    console.print(
        Panel(
            f"[bold red]{type(error).__name__}: {error}[/bold red]\n\n{tail_text(traceback.format_exc(), 6000)}",
            title="Unexpected failure",
            border_style="red",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with reports and redacted diagnostics."""
    console = Console()
    try:
        settings = settings_from_argv(argv)
    except (StackError, ValidationError) as exc:
        render_failure(console, exc)
        return 2
    stack = Stack(settings, console)
    exit_code = 0
    try:
        stack.execute()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        exit_code = 130
    except BaseException as exc:
        render_failure(console, exc)
        try:
            diagnostics = stack.collect_diagnostics(exc)
            console.print(f"[dim]Redacted diagnostics: {diagnostics}[/dim]")
        except BaseException as diagnostic_error:
            console.print(f"[dim]Could not write diagnostics: {diagnostic_error}[/dim]")
        exit_code = 1
    finally:
        try:
            stack.reporter.save(command=settings.command, exit_code=exit_code)
        except OSError as exc:
            console.print(f"[dim]Could not save last report: {exc}[/dim]")
    if exit_code == 0:
        warnings = sum(item.level is CheckLevel.WARN for item in stack.reporter.results)
        failures = sum(item.level is CheckLevel.FAIL for item in stack.reporter.results)
        console.print(
            f"\n[bold green]Completed successfully.[/bold green] "
            f"{len(stack.reporter.results)} checks, {warnings} warning(s), {failures} recorded failure(s)."
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
