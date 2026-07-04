import asyncio
from collections.abc import Sequence
import inspect
from threading import Event, Lock

import httpx

from LLLM.generator import AssistantOutput, ChatCompletion, ChatMessage, JsonObjectSpec
from LLLM.server import create_app


class BlockingGenerator:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self._state_lock = Lock()
        self.calls = 0
        self.active = 0
        self.max_active = 0

    def generate_completion(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, object]] | None = None,
        max_generated_token: int = 20,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        enable_thinking: bool = True,
        response_format: JsonObjectSpec | None = None,
    ) -> ChatCompletion:
        del (
            messages,
            tools,
            max_generated_token,
            temperature,
            top_k,
            top_p,
            enable_thinking,
            response_format,
        )
        with self._state_lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        assert self.release.wait(timeout=2)
        with self._state_lock:
            self.active -= 1
        return ChatCompletion(
            message=AssistantOutput(content="done"),
            raw_completion="done",
            prompt_tokens=1,
            generated_tokens=1,
            finish_reason="stop",
        )


def test_chat_completion_endpoint_is_async() -> None:
    app = create_app(BlockingGenerator())
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/v1/chat/completions"
    )

    assert inspect.iscoroutinefunction(endpoint)


def test_concurrent_completions_wait_without_overlapping_generation() -> None:
    async def run_requests() -> None:
        generator = BlockingGenerator()
        app = create_app(generator)
        transport = httpx.ASGITransport(app=app)
        request = {
            "model": "lllm",
            "messages": [{"role": "user", "content": "hello"}],
        }

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            first = asyncio.create_task(
                client.post("/v1/chat/completions", json=request)
            )
            assert await asyncio.to_thread(generator.started.wait, 1)

            second = asyncio.create_task(
                client.post("/v1/chat/completions", json=request)
            )
            await asyncio.sleep(0.05)

            assert generator.calls == 1
            assert generator.max_active == 1

            generator.release.set()
            responses = await asyncio.gather(first, second)

        assert all(response.status_code == 200 for response in responses)
        assert generator.calls == 2
        assert generator.max_active == 1

    asyncio.run(run_requests())
