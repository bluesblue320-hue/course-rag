"""Exercise the Docker stack, including volume persistence across recreation.

The smoke test deliberately isolates itself from the caller's environment:
every ``docker compose`` invocation uses a generated env file holding only
safe defaults plus a process environment stripped of ``LLM_*`` / ``RAG_*``
variables.  A local root ``.env`` or exported variables therefore can never
enable a real Reranker or carry real LLM credentials into the containers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:8080"

#: Variables removed from the subprocess environment for every Compose call.
#: They would either leak real credentials into containers or change the
#: retrieval behavior this test is supposed to exercise in its default state.
_SENSITIVE_ENV_PREFIXES = ("LLM_", "RAG_")
_SENSITIVE_ENV_KEYS = frozenset({"APP_PORT", "MAX_UPLOAD_BYTES"})

#: Safe values written into the isolated env file passed to Compose, so the
#: project-root ``.env`` (if any) is ignored for interpolation.
ISOLATED_ENV_FILE_CONTENT = """\
APP_PORT=8080
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
LLM_TIMEOUT_SECONDS=30
MAX_UPLOAD_BYTES=10485760
RAG_MIN_RELEVANCE_SCORE=0.35
RAG_RERANKER_ENABLED=false
RAG_RERANKER_MODEL=
RAG_RERANKER_CANDIDATE_TOP_K=15
"""

#: Path of the isolated env file; created in main() and removed afterwards.
_isolated_env_file: Path | None = None


def _sanitized_environment() -> dict[str, str]:
    """Return the process environment without sensitive LLM/RAG variables."""
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_SENSITIVE_ENV_PREFIXES)
        and key not in _SENSITIVE_ENV_KEYS
    }


def _write_isolated_env_file() -> Path:
    """Create a temporary env file with safe defaults for Compose to use."""
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="course-rag-smoke-",
        suffix=".env",
        delete=False,
    )
    with handle:
        handle.write(ISOLATED_ENV_FILE_CONTENT)
    return Path(handle.name)


def run_compose(*arguments: str, capture_output: bool = False) -> str:
    """Run one Docker Compose command from the repository root.

    Every call uses the isolated env file and a sanitized environment, so a
    local ``.env`` or exported variables cannot enable a real Reranker or
    carry real LLM credentials into the containers.
    """
    if _isolated_env_file is None:
        raise RuntimeError("isolated env file has not been created")
    completed = subprocess.run(
        ["docker", "compose", "--env-file", str(_isolated_env_file), *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=capture_output,
        env=_sanitized_environment(),
    )
    return completed.stdout.strip() if capture_output else ""


def request_bytes(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30,
) -> bytes:
    """Send one HTTP request and surface response bodies on failure."""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers or {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {path} returned HTTP {exc.code}: {error_body}"
        ) from exc


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Send one request and decode its JSON object response."""
    request_headers = dict(headers or {})
    request_body = body
    if payload is not None:
        request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    raw = request_bytes(
        base_url,
        path,
        method=method,
        body=request_body,
        headers=request_headers,
    )
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise AssertionError(f"{method} {path} did not return a JSON object")
    return decoded


def wait_until_ready(base_url: str, timeout_seconds: float) -> dict[str, Any]:
    """Wait until Nginx can reach a ready FastAPI backend."""
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            health = request_json(base_url, "/api/health")
            if health.get("status") == "ok" and health.get("retrieval_ready") is True:
                return health
            last_error = AssertionError(f"unexpected health payload: {health}")
        except (OSError, RuntimeError, ValueError, AssertionError) as exc:
            last_error = exc
        time.sleep(3)

    raise TimeoutError(
        f"stack was not ready within {timeout_seconds:.0f}s; last error: {last_error}"
    )


def upload_test_document(base_url: str) -> dict[str, Any]:
    """Upload one deterministic text fixture through the Nginx proxy."""
    boundary = f"----course-rag-{uuid4().hex}"
    filename = "docker-persistence-test.txt"
    content = (
        "Docker 持久化验证口令：蓝鲸石英灯塔。\n"
        "该文档仅用于验证上传文件、元数据和重建后的检索索引。\n"
    ).encode("utf-8")
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: text/plain\r\n"
        "\r\n"
    ).encode("ascii") + content + f"\r\n--{boundary}--\r\n".encode("ascii")

    uploaded = request_json(
        base_url,
        "/api/documents",
        method="POST",
        body=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    if uploaded.get("filename") != filename or uploaded.get("is_builtin") is not False:
        raise AssertionError(f"unexpected upload response: {uploaded}")
    return uploaded


def backend_python(script: str) -> str:
    """Execute a small read/write probe inside the backend container."""
    return run_compose(
        "exec",
        "-T",
        "backend",
        "python",
        "-c",
        script,
        capture_output=True,
    )


def assert_default_rag_settings() -> dict[str, str]:
    """Verify Compose passes the safe default settings to the backend.

    The smoke test always launches Compose with the isolated env file and a
    sanitized process environment, so the container must receive the
    application's built-in defaults regardless of any root ``.env`` or
    exported variables: reranker disabled, candidate pool 15, relevance
    threshold 0.35, and empty LLM credentials.
    """
    expected = {
        "RAG_RERANKER_ENABLED": "false",
        "RAG_RERANKER_CANDIDATE_TOP_K": "15",
        "RAG_MIN_RELEVANCE_SCORE": "0.35",
        "LLM_API_KEY": "",
        "LLM_BASE_URL": "",
        "LLM_MODEL": "",
    }
    raw = backend_python(
        "import json, os; "
        "print(json.dumps({"
        "'RAG_RERANKER_ENABLED': os.environ.get('RAG_RERANKER_ENABLED'), "
        "'RAG_RERANKER_CANDIDATE_TOP_K': os.environ.get('RAG_RERANKER_CANDIDATE_TOP_K'), "
        "'RAG_MIN_RELEVANCE_SCORE': os.environ.get('RAG_MIN_RELEVANCE_SCORE'), "
        "'LLM_API_KEY': os.environ.get('LLM_API_KEY'), "
        "'LLM_BASE_URL': os.environ.get('LLM_BASE_URL'), "
        "'LLM_MODEL': os.environ.get('LLM_MODEL')"
        "}))"
    )
    settings = json.loads(raw)
    normalized = {
        key: (value.lower() if value is not None else value)
        for key, value in settings.items()
    }
    if normalized != expected:
        raise AssertionError(
            f"RAG/LLM settings drifted from the safe defaults: "
            f"{settings} != {expected}"
        )
    return settings


def assert_compose_interpolation() -> None:
    """Verify ``${VAR:-default}`` interpolation expands explicit overrides.

    This only runs ``docker compose config`` with temporary override
    variables on top of the sanitized environment; it never starts
    containers, so no real Reranker is loaded.  The overrides are injected
    into a child process and cannot leak into the caller's environment, and
    any caller-provided LLM/RAG variables are stripped first.
    """
    overrides = {
        "RAG_RERANKER_ENABLED": "true",
        "RAG_RERANKER_MODEL": "test/model",
        "RAG_RERANKER_CANDIDATE_TOP_K": "12",
        "RAG_MIN_RELEVANCE_SCORE": "0.42",
    }
    child_env = {**_sanitized_environment(), **overrides}
    completed = subprocess.run(
        ["docker", "compose", "--env-file", str(_isolated_env_file), "config"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
        env=child_env,
    )
    for key, value in overrides.items():
        expected_line = f"{key}: {value}"
        quoted_line = f'{key}: "{value}"'
        if expected_line not in completed.stdout and quoted_line not in completed.stdout:
            raise AssertionError(
                f"Compose did not expand {key}={value!r}; expected "
                f"{expected_line!r} or {quoted_line!r} in `docker compose config` output"
            )


def cache_stats() -> dict[str, int]:
    """Return the number and total size of cached Hugging Face files."""
    raw = backend_python(
        "import json; from pathlib import Path; "
        "root = Path('/cache/huggingface'); "
        "files = [path for path in root.rglob('*') if path.is_file()]; "
        "print(json.dumps({'file_count': len(files), "
        "'total_bytes': sum(path.stat().st_size for path in files)}))"
    )
    stats = json.loads(raw)
    if stats["file_count"] <= 0 or stats["total_bytes"] <= 0:
        raise AssertionError(f"Hugging Face cache is empty: {stats}")
    return stats


def write_cache_marker(marker_path: str, marker_value: str) -> None:
    """Write a unique marker into the mounted model-cache path."""
    backend_python(
        "from pathlib import Path; "
        f"Path({marker_path!r}).write_text({marker_value!r}, encoding='utf-8')"
    )


def read_cache_marker(marker_path: str) -> str:
    """Read the unique cache marker after container recreation."""
    return backend_python(
        "from pathlib import Path; "
        f"print(Path({marker_path!r}).read_text(encoding='utf-8'))"
    )


def remove_cache_marker(marker_path: str) -> None:
    """Remove the temporary marker after a successful test."""
    backend_python(
        "from pathlib import Path; "
        f"Path({marker_path!r}).unlink(missing_ok=True)"
    )


def assert_document_persisted(
    base_url: str,
    uploaded: dict[str, Any],
) -> dict[str, Any]:
    """Verify metadata and the rebuilt index still contain the upload."""
    document_id = uploaded["document_id"]
    listing = request_json(base_url, "/api/documents")
    matches = [
        document
        for document in listing.get("documents", [])
        if document.get("document_id") == document_id
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"uploaded document {document_id} was not restored: {listing}"
        )

    restored = matches[0]
    for key in ("filename", "size_bytes", "text_length", "chunk_count", "created_at"):
        if restored.get(key) != uploaded.get(key):
            raise AssertionError(
                f"restored metadata differs for {key}: "
                f"{restored.get(key)!r} != {uploaded.get(key)!r}"
            )

    search = request_json(
        base_url,
        "/api/search",
        method="POST",
        payload={"query": "蓝鲸石英灯塔 Docker 持久化验证口令", "top_k": 10},
    )
    result_ids = {
        result.get("document_id")
        for result in search.get("results", [])
        if isinstance(result, dict)
    }
    if document_id not in result_ids:
        raise AssertionError(
            "uploaded document metadata survived, but the rebuilt index "
            f"did not return it: {search}"
        )
    return listing


def assert_frontend(base_url: str) -> None:
    """Verify both the Nginx health route and compiled Vue entry page."""
    nginx_health = request_bytes(base_url, "/nginx-health").decode("utf-8").strip()
    if nginx_health != "ok":
        raise AssertionError(f"unexpected Nginx health response: {nginx_health!r}")
    html = request_bytes(base_url, "/").decode("utf-8", errors="replace").lower()
    if "<!doctype html>" not in html or 'id="app"' not in html:
        raise AssertionError("Nginx did not serve the compiled Vue entry page")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build/start the Compose stack, upload a document, recreate the "
            "containers, and verify data plus model-cache persistence."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="reuse existing local images instead of building them first",
    )
    return parser.parse_args()


def main() -> int:
    global _isolated_env_file
    args = parse_args()
    marker_token = hashlib.sha256(uuid4().bytes).hexdigest()
    marker_path = f"/cache/huggingface/.course-rag-test-{uuid4().hex}"

    _isolated_env_file = _write_isolated_env_file()
    try:
        return _run(args, marker_token, marker_path)
    finally:
        if _isolated_env_file is not None:
            _isolated_env_file.unlink(missing_ok=True)
            _isolated_env_file = None


def _run(args: argparse.Namespace, marker_token: str, marker_path: str) -> int:
    print("[1/7] Verifying Compose interpolation and starting the stack...")
    assert_compose_interpolation()
    up_arguments = ["up", "-d"]
    if not args.skip_build:
        up_arguments.insert(1, "--build")
    run_compose(*up_arguments)

    print("[2/7] Waiting for Nginx and FastAPI readiness...")
    initial_health = wait_until_ready(args.base_url, args.timeout_seconds)
    assert_frontend(args.base_url)
    rag_settings = assert_default_rag_settings()

    print("[3/7] Uploading a persistence test document...")
    uploaded = upload_test_document(args.base_url)

    print("[4/7] Recording the Hugging Face cache and volume marker...")
    cache_before = cache_stats()
    write_cache_marker(marker_path, marker_token)

    print("[5/7] Recreating containers without deleting named volumes...")
    run_compose("down")
    run_compose("up", "-d")
    restarted_health = wait_until_ready(args.base_url, args.timeout_seconds)

    print("[6/7] Verifying restored metadata, file, index, and model cache...")
    assert_frontend(args.base_url)
    listing = assert_document_persisted(args.base_url, uploaded)
    cache_after = cache_stats()
    if read_cache_marker(marker_path) != marker_token:
        raise AssertionError("Hugging Face cache marker did not survive recreation")
    if cache_after["total_bytes"] < cache_before["total_bytes"]:
        raise AssertionError(
            f"model cache shrank unexpectedly: {cache_before} -> {cache_after}"
        )

    print("[7/7] Removing only the temporary test document and marker...")
    request_json(
        args.base_url,
        f"/api/documents/{uploaded['document_id']}",
        method="DELETE",
    )
    remove_cache_marker(marker_path)

    summary = {
        "status": "passed",
        "base_url": args.base_url,
        "initial_health": initial_health,
        "restarted_health": restarted_health,
        "rag_settings": rag_settings,
        "restored_document_id": uploaded["document_id"],
        "restored_document_count": listing["document_count"],
        "cache_before": cache_before,
        "cache_after": cache_after,
        "stack_left_running": True,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Docker persistence test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
