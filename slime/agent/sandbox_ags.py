"""Tencent AGS sandbox backend via SWE-ReX.

Required environment (fail on ``__aenter__``, not on import):

Secrets — either:

* ``SLIME_AGENT_AGS_ENV_FILE`` — KEY=VALUE file loaded with ``override=False``, or
* ``SLIME_AGENT_AGS_SECRET_ID`` + ``SLIME_AGENT_AGS_SECRET_KEY``

Plus:

* ``SLIME_AGENT_AGS_REGION``
* ``SLIME_AGENT_AGS_DOMAIN``
* ``SLIME_AGENT_AGS_ROLE_ARN``
* ``SLIME_AGENT_AGS_HTTP_ENDPOINT``

Optional: ``SLIME_AGENT_AGS_TOOL_ID`` (empty → create a new SandboxTool),
``SLIME_AGENT_AGS_CPU``, ``SLIME_AGENT_AGS_MEMORY``,
``SLIME_AGENT_AGS_TIMEOUT``, ``SLIME_AGENT_AGS_PORT``,
``SLIME_AGENT_AGS_BOOT_TIMEOUT_SEC``, ``SLIME_AGENT_AGS_RUNTIME_TIMEOUT_SEC``,
``SLIME_AGENT_AGS_MAX_RETRIES``, ``SLIME_AGENT_AGS_RETRY_DELAYS_SEC``,
``SLIME_AGENT_AGS_IMAGE_REGISTRY_TYPE``, mount-related ``SLIME_AGENT_AGS_MOUNT_*`` /
``SLIME_AGENT_AGS_IMAGE_SUBPATH``, and ``SLIME_AGENT_AGS_SWE_REX_ROOT`` (sys.path
for SWE-ReX).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import shlex
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

import aiohttp

from slime.agent.sandbox import ExecResult, FileContent

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_DEFAULT_MAX_RETRIES = 5
_DEFAULT_RETRY_DELAYS_SEC = (15.0, 30.0, 60.0)
_DEFAULT_RETRY_JITTER_RATIO = 0.1
_DEFAULT_RETRY_JITTER_MAX_SEC = 5.0

_REQUIRED_KEYS = (
    "SLIME_AGENT_AGS_SECRET_ID",
    "SLIME_AGENT_AGS_SECRET_KEY",
    "SLIME_AGENT_AGS_REGION",
    "SLIME_AGENT_AGS_DOMAIN",
    "SLIME_AGENT_AGS_ROLE_ARN",
    "SLIME_AGENT_AGS_HTTP_ENDPOINT",
)

# Optional: when unset/empty, SWE-ReX creates a new SandboxTool for the image.
_OPTIONAL_TOOL_ID = "SLIME_AGENT_AGS_TOOL_ID"


def _load_env_file(path: str | Path, *, override: bool = False) -> None:
    """Load a simple KEY=VALUE env file into ``os.environ``."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"SLIME_AGENT_AGS_ENV_FILE not found: {p}")
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("[") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def _require_ags_env() -> dict[str, str]:
    """Return required AGS env values, or raise listing every missing key."""
    env_file = os.environ.get("SLIME_AGENT_AGS_ENV_FILE", "").strip()
    if env_file:
        if Path(env_file).is_file():
            _load_env_file(env_file, override=False)
        else:
            # Placeholder paths from env.example must not block Job-injected secrets.
            logger.warning(
                "[agent.sandbox_ags] SLIME_AGENT_AGS_ENV_FILE not found (%s); "
                "continuing with process env",
                env_file,
            )

    missing = [k for k in _REQUIRED_KEYS if not (os.environ.get(k) or "").strip()]
    if missing:
        raise RuntimeError(
            "Missing required AGS config: " + ", ".join(missing) + ". "
            "Set them directly or via SLIME_AGENT_AGS_ENV_FILE."
        )
    return {k: os.environ[k].strip() for k in _REQUIRED_KEYS}


def _optional(name: str, default: str) -> str:
    return (os.environ.get(name) or "").strip() or default


def _retry_count() -> int:
    raw = _optional("SLIME_AGENT_AGS_MAX_RETRIES", str(_DEFAULT_MAX_RETRIES))
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid SLIME_AGENT_AGS_MAX_RETRIES=%r; using %d", raw, _DEFAULT_MAX_RETRIES)
        return _DEFAULT_MAX_RETRIES


def _retry_delays() -> tuple[float, ...]:
    raw = _optional("SLIME_AGENT_AGS_RETRY_DELAYS_SEC", "15,30,60")
    try:
        delays = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
        if not delays or any(value < 0 for value in delays):
            raise ValueError
        return delays
    except ValueError:
        logger.warning(
            "Invalid SLIME_AGENT_AGS_RETRY_DELAYS_SEC=%r; using 15,30,60",
            raw,
        )
        return _DEFAULT_RETRY_DELAYS_SEC


def _is_transient_request_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            asyncio.TimeoutError,
            aiohttp.ClientConnectionError,
            aiohttp.ClientPayloadError,
        ),
    ):
        return True
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in {408, 425, 429, 500, 502, 503, 504}
    return False


async def _retry_transient_request(
    operation: Callable[[], Awaitable[_T]],
    *,
    max_retries: int,
    delays: tuple[float, ...],
    description: str,
) -> _T:
    """Retry transient transport failures; the operation owns request identity."""
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except BaseException as exc:
            if not _is_transient_request_error(exc) or attempt >= max_retries:
                raise
            base_delay = delays[min(attempt, len(delays) - 1)]
            jitter = random.uniform(
                0.0,
                min(_DEFAULT_RETRY_JITTER_MAX_SEC, base_delay * _DEFAULT_RETRY_JITTER_RATIO),
            )
            sleep_sec = base_delay + jitter
            logger.warning(
                "[agent.sandbox_ags] transient %s failure; retry %d/%d in %.1fs: %s",
                description,
                attempt + 1,
                max_retries,
                sleep_sec,
                exc,
            )
            await asyncio.sleep(sleep_sec)
    raise AssertionError("unreachable")


class AGSSandbox:
    """Async sandbox backed by Tencent AGS through SWE-ReX."""

    def __init__(self, image: str) -> None:
        self.image = image
        self.sandbox_id = ""
        self._deployment: Any = None
        self._rex_command_cls: Any = None
        self._rex_command_response_cls: Any = None

    @staticmethod
    def _import_swerex():
        root = (os.environ.get("SLIME_AGENT_AGS_SWE_REX_ROOT") or "").strip()
        if root and root not in sys.path:
            sys.path.insert(0, root)
        try:
            from swerex.deployment.config import TencentAGSDeploymentConfig, get_deployment
            from swerex.runtime.abstract import Command as RexCommand
            from swerex.runtime.abstract import CommandResponse as RexCommandResponse
        except ImportError as e:
            raise RuntimeError(
                "Failed to import SWE-ReX AGS runtime. "
                "Install SWE-ReX or set SLIME_AGENT_AGS_SWE_REX_ROOT to its src/ path."
            ) from e
        return TencentAGSDeploymentConfig, get_deployment, RexCommand, RexCommandResponse

    def _deployment_kwargs(self, required: dict[str, str]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "secret_id": required["SLIME_AGENT_AGS_SECRET_ID"],
            "secret_key": required["SLIME_AGENT_AGS_SECRET_KEY"],
            "tool_id": _optional(_OPTIONAL_TOOL_ID, ""),
            "region": required["SLIME_AGENT_AGS_REGION"],
            "domain": required["SLIME_AGENT_AGS_DOMAIN"],
            "role_arn": required["SLIME_AGENT_AGS_ROLE_ARN"],
            "http_endpoint": required["SLIME_AGENT_AGS_HTTP_ENDPOINT"],
            "image": self.image,
            "image_registry_type": _optional("SLIME_AGENT_AGS_IMAGE_REGISTRY_TYPE", "enterprise"),
            "timeout": _optional("SLIME_AGENT_AGS_TIMEOUT", "30m"),
            "startup_timeout": float(_optional("SLIME_AGENT_AGS_BOOT_TIMEOUT_SEC", "600")),
            "runtime_timeout": float(_optional("SLIME_AGENT_AGS_RUNTIME_TIMEOUT_SEC", "600")),
            "cpu": _optional("SLIME_AGENT_AGS_CPU", "2"),
            "memory": _optional("SLIME_AGENT_AGS_MEMORY", "4Gi"),
            "port": int(_optional("SLIME_AGENT_AGS_PORT", "8000")),
            "mount_readonly": False,
        }
        mount_name = _optional("SLIME_AGENT_AGS_MOUNT_NAME", "")
        if mount_name:
            kwargs["mount_name"] = mount_name
            kwargs["mount_image"] = _optional("SLIME_AGENT_AGS_MOUNT_IMAGE", "")
            kwargs["mount_image_registry_type"] = _optional(
                "SLIME_AGENT_AGS_MOUNT_IMAGE_REGISTRY_TYPE", "enterprise"
            )
            kwargs["mount_path"] = _optional("SLIME_AGENT_AGS_MOUNT_PATH", "/nix")
            kwargs["image_subpath"] = _optional("SLIME_AGENT_AGS_IMAGE_SUBPATH", "/nix")
        return kwargs

    async def __aenter__(self) -> AGSSandbox:
        required = _require_ags_env()
        config_cls, get_deployment, rex_command_cls, rex_command_response_cls = self._import_swerex()
        self._rex_command_cls = rex_command_cls
        self._rex_command_response_cls = rex_command_response_cls
        self._deployment = get_deployment(config_cls(**self._deployment_kwargs(required)))
        await self._deployment.start()
        self.sandbox_id = str(
            getattr(self._deployment, "instance_id", None)
            or getattr(self._deployment, "sandbox_id", None)
            or ""
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._deployment is None:
            return
        try:
            await self._deployment.stop()
        except Exception as e:
            logger.warning("[agent.sandbox_ags] stop %s failed: %s", self.sandbox_id, e)

    @staticmethod
    def _wrap_cmd(cmd: str, *, user: str, env: dict[str, str] | None) -> str:
        exports = ""
        if env:
            exports = " ".join(f"export {k}={shlex.quote(str(v))};" for k, v in env.items())
        body = f"{exports} {cmd}".strip()
        if user and user != "root":
            return f"runuser -u {shlex.quote(user)} -- bash -lc {shlex.quote(body)}"
        return body

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        if (
            self._deployment is None
            or self._rex_command_cls is None
            or self._rex_command_response_cls is None
        ):
            raise RuntimeError("AGSSandbox is not started")

        # SWE-ReX HTTP client timeout is independent of Command.timeout.
        needed = int(timeout) + 60
        raw = os.environ.get("SWEREX_REQUEST_TIMEOUT", "")
        try:
            current = float(raw) if raw else 120.0
        except ValueError:
            current = 120.0
        if current < needed:
            os.environ["SWEREX_REQUEST_TIMEOUT"] = str(needed)

        command = self._rex_command_cls(
            command=self._wrap_cmd(cmd, user=user, env=env),
            shell=True,
            check=False,
            timeout=timeout,
            merge_output_streams=False,
        )
        if idempotent and _retry_count() > 0:
            res = await self._execute_idempotent_with_retry(command)
        else:
            res = await self._deployment.runtime.execute(command)
        exit_code = int(getattr(res, "exit_code", 0))
        stdout = getattr(res, "stdout", "") or ""
        stderr = getattr(res, "stderr", "") or ""
        if check and exit_code != 0:
            raise RuntimeError(f"ags exec failed (exit={exit_code}): {cmd[:120]}\n{stderr[:400]}")
        return exit_code, stdout, stderr

    async def _execute_idempotent_with_retry(self, command: Any) -> Any:
        """Call AGS ``/execute`` with one request ID across all retry attempts."""
        runtime = self._deployment.runtime
        request_id = str(uuid.uuid4())

        async def request_once() -> Any:
            ensure_token = getattr(runtime, "_ensure_valid_token", None)
            if callable(ensure_token):
                await ensure_token()
            headers = dict(runtime._headers)
            headers["X-Request-ID"] = request_id
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as session:
                async with session.post(
                    f"{runtime._api_url}/execute",
                    json=command.model_dump(),
                    headers=headers,
                ) as response:
                    await runtime._handle_response_errors(response)
                    return self._rex_command_response_cls(**await response.json())

        return await _retry_transient_request(
            request_once,
            max_retries=_retry_count(),
            delays=_retry_delays(),
            description="AGS /execute",
        )

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, Path):
            data = content.read_bytes()
        elif isinstance(content, str):
            data = content.encode("utf-8")
        else:
            data = bytes(content)

        b64 = base64.b64encode(data).decode("ascii")
        quoted = shlex.quote(sandbox_path)
        chunk = 60_000 - (60_000 % 4)

        for offset in range(0, max(len(b64), 1), chunk):
            piece = b64[offset : offset + chunk]
            redirect = ">" if offset == 0 else ">>"
            await self.exec(
                f"mkdir -p $(dirname {quoted}) && "
                f"printf %s {shlex.quote(piece)} | base64 -d {redirect} {quoted}",
                user="root",
                timeout=120,
                check=True,
            )
        if user != "root":
            await self.exec(
                f"chown {shlex.quote(user)}:{shlex.quote(user)} {quoted}",
                user="root",
                timeout=120,
                check=True,
            )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        try:
            ec, out, _ = await self.exec(
                f"cat {shlex.quote(sandbox_path)}", user=user, timeout=120, check=False
            )
            return out if ec == 0 else ""
        except Exception:
            return ""
