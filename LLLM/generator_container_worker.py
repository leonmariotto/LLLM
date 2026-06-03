"""
Container-side RPC server for a ContainerizedGeneratorWithTool instance.

The host process starts this module in a long-lived container.  The worker
constructs the real generator from an importable factory and executes all
generate calls in-place inside the container.
The point of all this is that tools are executed inside this container, so only
mounted volume can be affected by model.

Endpoints:
    GET /health:
        Health check used by the host proxy during startup. Returns
        {"ok": true} with HTTP 200 once the generator has been constructed
        and the HTTP server is listening.
    POST /generate:
        Execute one GeneratorWithTool.generate call. The JSON request body
        must contain 'messages' and may include generation options such as
        'stop_at_eos', 'max_generated_token', 'cache_length',
        'temperature', 'top_k', and 'top_p'. Successful calls return
        {"result": "<assistant response>"}. Worker-side exceptions are
        serialized as {"error": {"type": ..., "message": ..., "traceback":
        ...}} so the host can raise a useful RuntimeError.
"""

from __future__ import annotations

import importlib
import json
import os
import traceback
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

from loguru import logger

from .generator_with_tool import GeneratorWithTool, ToolMessage

_FACTORY_ENV = "LLLM_GENERATOR_FACTORY"
_FACTORY_KWARGS_ENV = "LLLM_GENERATOR_FACTORY_KWARGS"
_HOST_ENV = "LLLM_GENERATOR_WORKER_HOST"
_PORT_ENV = "LLLM_GENERATOR_WORKER_PORT"


def load_factory(factory_path: str) -> Callable[..., object]:
    """Import a factory.

    Args:
        factory_path: Import path in "module:callable" format. The
            callable portion may be dotted, for example
            "some.module:FactoryClass.create".

    Returns:
        The imported callable object. The caller is responsible for invoking it
        and validating its return type.

    Raises:
        ValueError: If factory_path does not use "module:callable"
            syntax.
        AttributeError: If the module imports but the requested attribute path
            does not exist.
        TypeError: If the imported object is not callable.
        ImportError: If the module cannot be imported.
    """
    if ":" not in factory_path:
        raise ValueError("factory must use 'module:callable' format")
    module_name, qualname = factory_path.split(":", 1)
    if not module_name or not qualname:
        raise ValueError("factory must use 'module:callable' format")
    logger.info("Importing GeneratorWithTool factory module {}", module_name)
    value: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"factory {factory_path!r} is not callable")
    return value


def build_generator_from_env() -> GeneratorWithTool:
    """Build the configured container-side GeneratorWithTool.

    Inputs are read from environment variables:
        LLLM_GENERATOR_FACTORY:
            Required "module:callable" factory path.
        LLLM_GENERATOR_FACTORY_KWARGS:
            Optional JSON object of keyword arguments passed to the factory.

    Returns:
        A fully constructed GeneratorWithTool that will handle all
        /generate requests for this worker process.

    Raises:
        ValueError: If the factory env var is missing, or if the kwargs env var
            is not a JSON object.
        TypeError: If the factory does not return GeneratorWithTool.
        json.JSONDecodeError: If the kwargs env var is not valid JSON.
        ImportError, AttributeError: If the factory path cannot be imported.
    """
    factory_path = os.environ.get(_FACTORY_ENV)
    if not factory_path:
        raise ValueError(f"{_FACTORY_ENV} must be set")
    raw_kwargs = os.environ.get(_FACTORY_KWARGS_ENV, "{}")
    kwargs = json.loads(raw_kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError(f"{_FACTORY_KWARGS_ENV} must be a JSON object")
    logger.info("Loading GeneratorWithTool factory {}", factory_path)
    factory = load_factory(factory_path)
    logger.info("Calling GeneratorWithTool factory {}", factory_path)
    generator = factory(**cast(dict[str, object], kwargs))
    if not isinstance(generator, GeneratorWithTool):
        raise TypeError("factory must return GeneratorWithTool")
    logger.info("GeneratorWithTool loaded in container")
    return generator


def execute_generate_payload(
    generator: GeneratorWithTool,
    payload: dict[str, object],
) -> str:
    """Execute one serialized GeneratorWithTool.generate request.

    Args:
        generator: Container-local generator created by
            :func:`build_generator_from_env`.
        payload: JSON-decoded request body. 'messages' is required and must
            be a list. Optional keys are 'stop_at_eos',
            'max_generated_token', 'cache_length', 'temperature',
            'top_k', and 'top_p'.

    Returns:
        The final assistant response string returned by
        GeneratorWithTool.generate.

    Raises:
        ValueError: If required or optional payload fields have invalid types.
        RuntimeError: Propagates generation failures such as exceeding the tool
            round limit.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    return generator.generate(
        cast(list[ToolMessage], messages),
        stop_at_eos=_optional_bool(payload, "stop_at_eos", True),
        max_generated_token=_optional_int(payload, "max_generated_token", 20),
        cache_length=_optional_int_or_none(payload, "cache_length"),
        temperature=_optional_float(payload, "temperature", 0.0),
        top_k=_optional_int_or_none(payload, "top_k"),
        top_p=_optional_float_or_none(payload, "top_p"),
    )


def make_handler(generator: GeneratorWithTool) -> type[BaseHTTPRequestHandler]:
    """Create an HTTP request handler bound to generator.

    Args:
        generator: The GeneratorWithTool instance used for every
            `POST /generate` request handled by this worker.

    Returns:
        A BaseHTTPRequestHandler subclass suitable for
        ThreadingHTTPServer. The class closes over generator rather
        than constructing a new model per request.
    """

    class GeneratorHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self._write_json(404, {"error": "not found"})
                return
            self._write_json(200, {"ok": True})

        def do_POST(self) -> None:
            if self.path != "/generate":
                self._write_json(404, {"error": "not found"})
                return
            try:
                payload = self._read_json_object()
                logger.info("Container generate request received")
                result = execute_generate_payload(generator, payload)
            except Exception as error:
                logger.exception("Container generate request failed")
                self._write_json(
                    200,
                    {
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    },
                )
                return
            logger.info("Container generate request completed")
            self._write_json(200, {"result": result})

        def log_message(self, format: str, *args: object) -> None:
            logger.debug("HTTP {}", format % args)

        def _read_json_object(self) -> dict[str, object]:
            length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(length)
            payload = cast(object, json.loads(raw.decode("utf-8")))
            if not isinstance(payload, dict):
                raise ValueError("request payload must be a JSON object")
            return cast(dict[str, object], payload)

        def _write_json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return GeneratorHandler


def main() -> int:
    """Serve the configured generator until the container is stopped.

    Environment variables:
        LLLM_GENERATOR_WORKER_HOST:
            Host/interface to bind. Defaults to "127.0.0.1".
        LLLM_GENERATOR_WORKER_PORT:
            Port to bind. Defaults to "8765".

    Returns:
        0 if serve_forever exits normally.

    Raises:
        Any exception from generator construction or server binding. In normal
        container usage those failures make the process exit before the host
        proxy's health check succeeds.
    """
    logger.info("Generator container worker process started")
    host = os.environ.get(_HOST_ENV, "127.0.0.1")
    port = int(os.environ.get(_PORT_ENV, "8765"))
    logger.info("Generator container worker configured host={} port={}", host, port)
    generator = build_generator_from_env()
    logger.info("Binding generator container worker HTTP server")
    server = ThreadingHTTPServer((host, port), make_handler(generator))
    logger.info("Generator container worker listening on {}:{}", host, port)
    server.serve_forever()
    return 0


def _optional_bool(payload: dict[str, object], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _optional_int(payload: dict[str, object], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _optional_int_or_none(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer or null")
    return value


def _optional_float(payload: dict[str, object], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _optional_float_or_none(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number or null")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
