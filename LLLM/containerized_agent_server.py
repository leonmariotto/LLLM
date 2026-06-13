"""
Container-side RPC server for a ContainerizedAgent instance.

The host process starts this module in a long-lived container.  The worker
constructs the real agent from an importable factory and executes all run calls
in-place inside the container.
The point of all this is that tools are executed inside this container, so only
mounted volume can be affected by model.

Endpoints:
    GET /health:
        Health check used by the host proxy during startup. Returns
        {"ok": true} with HTTP 200 once the agent has been constructed and the
        HTTP server is listening.
    POST /run:
        Execute one Agent.run call. The JSON request body must contain
        'prompt'. Successful calls return a serialized AgentResult.
        Worker-side exceptions are
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

from .agent import Agent
from .agent_context import AgentResult, ExecutionContext

_FACTORY_ENV = "LLLM_AGENT_FACTORY"
_FACTORY_KWARGS_ENV = "LLLM_AGENT_FACTORY_KWARGS"
_HOST_ENV = "LLLM_AGENT_WORKER_HOST"
_PORT_ENV = "LLLM_AGENT_WORKER_PORT"


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
    logger.info("Importing Agent factory module {}", module_name)
    value: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"factory {factory_path!r} is not callable")
    return value


def build_agent_from_env() -> Agent:
    """Build the configured container-side Agent.

    Inputs are read from environment variables:
        LLLM_AGENT_FACTORY:
            Required "module:callable" factory path.
        LLLM_AGENT_FACTORY_KWARGS:
            Optional JSON object of keyword arguments passed to the factory.

    Returns:
        A fully constructed Agent that will handle all /run requests for this
        worker process.

    Raises:
        ValueError: If the factory env var is missing, or if the kwargs env var
            is not a JSON object.
        TypeError: If the factory does not return Agent.
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
    logger.info("Loading Agent factory {}", factory_path)
    factory = load_factory(factory_path)
    logger.info("Calling Agent factory {}", factory_path)
    agent = factory(**cast(dict[str, object], kwargs))
    if not isinstance(agent, Agent):
        raise TypeError("factory must return Agent")
    logger.info("Agent loaded in container")
    return agent


def execute_run_payload(
    agent: Agent,
    payload: dict[str, object],
) -> AgentResult:
    """Execute one serialized Agent.run request.

    Args:
        agent: Container-local agent created by :func:`build_agent_from_env`.
        payload: JSON-decoded request body. 'prompt' is required and must be a
            string.

    Returns:
        The AgentResult returned by Agent.run.

    Raises:
        ValueError: If required payload fields have invalid types.
        RuntimeError: Propagates agent failures.
    """
    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    return agent.run(prompt)


def serialize_agent_result(result: AgentResult) -> dict[str, object]:
    """Serialize an AgentResult into a JSON-compatible dictionary."""
    return {
        "output": _jsonable(result.output),
        "status": result.status,
        "context": _serialize_context(result.context),
    }


def _serialize_context(context: ExecutionContext) -> dict[str, object]:
    return {
        "execution_id": context.execution_id,
        "events": [event.model_dump(mode="json") for event in context.events],
        "current_step": context.current_step,
        "state": _jsonable(context.state),
        "final_result": _jsonable(context.final_result),
    }


def _jsonable(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[no-any-return, attr-defined]
    if isinstance(value, dict):
        typed_value = cast(dict[object, object], value)
        return {str(key): _jsonable(item) for key, item in typed_value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in cast(list[object], value)]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in cast(tuple[object, ...], value)]
    return value


def make_handler(agent: Agent) -> type[BaseHTTPRequestHandler]:
    """Create an HTTP request handler bound to agent.

    Args:
        agent: The Agent instance used for every `POST /run` request handled by
            this worker.

    Returns:
        A BaseHTTPRequestHandler subclass suitable for
        ThreadingHTTPServer. The class closes over agent rather than
        constructing a new model per request.
    """

    class AgentHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self._write_json(404, {"error": "not found"})
                return
            self._write_json(200, {"ok": True})

        def do_POST(self) -> None:
            if self.path != "/run":
                self._write_json(404, {"error": "not found"})
                return
            try:
                payload = self._read_json_object()
                logger.info("Container agent run request received")
                result = serialize_agent_result(execute_run_payload(agent, payload))
            except Exception as error:
                logger.exception("Container agent run request failed")
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
            logger.info("Container agent run request completed")
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

    return AgentHandler


def main() -> int:
    """Serve the configured agent until the container is stopped.

    Environment variables:
        LLLM_AGENT_WORKER_HOST:
            Host/interface to bind. Defaults to "127.0.0.1".
        LLLM_AGENT_WORKER_PORT:
            Port to bind. Defaults to "8765".

    Returns:
        0 if serve_forever exits normally.

    Raises:
        Any exception from agent construction or server binding. In normal
        container usage those failures make the process exit before the host
        proxy's health check succeeds.
    """
    logger.info("Agent container worker process started")
    host = os.environ.get(_HOST_ENV, "127.0.0.1")
    port = int(os.environ.get(_PORT_ENV, "8765"))
    logger.info("Agent container worker configured host={} port={}", host, port)
    agent = build_agent_from_env()
    logger.info("Binding agent container worker HTTP server")
    server = ThreadingHTTPServer((host, port), make_handler(agent))
    logger.info("Agent container worker listening on {}:{}", host, port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
