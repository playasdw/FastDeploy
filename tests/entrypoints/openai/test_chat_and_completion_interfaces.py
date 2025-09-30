"""Interface tests for OpenAI chat/completion serving entrypoints."""

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _ensure_fastdeploy_package() -> None:
    if "fastdeploy" not in sys.modules:
        package = ModuleType("fastdeploy")
        package.__path__ = [str(Path(__file__).resolve().parents[3] / "fastdeploy")]
        sys.modules["fastdeploy"] = package


def _install_metrics_stub() -> None:
    if "fastdeploy.metrics.work_metrics" in sys.modules:
        return

    module = ModuleType("fastdeploy.metrics.work_metrics")
    module.work_process_metrics = SimpleNamespace(
        e2e_request_latency=SimpleNamespace(observe=lambda *_args, **_kwargs: None)
    )
    sys.modules["fastdeploy.metrics.work_metrics"] = module


def _ensure_paddle_stub() -> None:
    if "paddle" in sys.modules:
        return

    paddle_stub = ModuleType("paddle")
    paddle_stub.seed = lambda *_args, **_kwargs: None
    paddle_stub.is_compiled_with_xpu = lambda: False
    paddle_stub.Tensor = type("Tensor", (), {})
    paddle_stub.empty = lambda *_args, **_kwargs: None
    paddle_stub.empty_like = lambda *_args, **_kwargs: None
    paddle_stub.device = SimpleNamespace(
        cuda=SimpleNamespace(
            max_memory_reserved=lambda *_args, **_kwargs: 0,
            max_memory_allocated=lambda *_args, **_kwargs: 0,
            memory_reserved=lambda *_args, **_kwargs: 0,
            memory_allocated=lambda *_args, **_kwargs: 0,
        )
    )
    sys.modules["paddle"] = paddle_stub


_ensure_fastdeploy_package()
_install_metrics_stub()
_ensure_paddle_stub()


from fastdeploy.entrypoints.openai.protocol import (  # noqa: E402
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ErrorResponse,
    UsageInfo,
)
from fastdeploy.entrypoints.openai.serving_chat import OpenAIServingChat  # noqa: E402
from fastdeploy.entrypoints.openai.serving_completion import (  # noqa: E402
    OpenAIServingCompletion,
)
from fastdeploy.utils import ParameterError  # noqa: E402


class _DummySemaphore:
    def __init__(self):
        self.acquired = 0

    async def acquire(self):
        self.acquired += 1

    def release(self):
        if self.acquired > 0:
            self.acquired -= 1

    def status(self):
        return {"acquired": self.acquired}


def _build_engine_client():
    engine_client = SimpleNamespace()
    engine_client.is_master = True
    engine_client.semaphore = _DummySemaphore()
    engine_client.format_and_add_data = AsyncMock(return_value=[101, 102, 103])
    engine_client.connection_manager = SimpleNamespace()
    engine_client.connection_manager.initialize = AsyncMock()
    engine_client.connection_manager.get_connection = AsyncMock()
    engine_client.connection_manager.cleanup_request = AsyncMock()
    engine_client.data_processor = SimpleNamespace()
    engine_client.check_model_weight_status = lambda: False
    engine_client.check_health = lambda: (True, "")
    engine_client.reasoning_parser = None
    return engine_client


@pytest.fixture()
def engine_client():
    return _build_engine_client()


@pytest.fixture()
def chat_handler(engine_client):
    return OpenAIServingChat(
        engine_client,
        models=None,
        pid="worker-0",
        ips=None,
        max_waiting_time=1,
        chat_template="chatml",
        enable_mm_output=False,
    )


@pytest.fixture()
def completion_handler(engine_client):
    return OpenAIServingCompletion(engine_client, models=None, pid="worker-0", ips=None, max_waiting_time=1)


def test_create_chat_completion_non_stream(monkeypatch, engine_client, chat_handler):

    fake_response = ChatCompletionResponse(
        id="chatcmpl-test",
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )

    async def fake_full_generator(self, request, request_id, model_name, prompt_token_ids, text_after_process):
        assert request.stream is False
        assert model_name == "test-model"
        assert prompt_token_ids == [101, 102, 103]
        self.engine_client.semaphore.release()
        return fake_response

    monkeypatch.setattr(OpenAIServingChat, "chat_completion_full_generator", fake_full_generator)

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
    )

    response = asyncio.run(chat_handler.create_chat_completion(request))

    assert response == fake_response
    engine_client.format_and_add_data.assert_awaited()
    assert engine_client.semaphore.acquired == 0


def test_create_chat_completion_stream(monkeypatch, engine_client, chat_handler):

    async def fake_stream_generator(self, request, request_id, model_name, prompt_token_ids, text_after_process):
        assert request.stream is True
        assert model_name == "test-model"
        assert prompt_token_ids == [101, 102, 103]
        try:
            yield "data: chunk\n\n"
        finally:
            self.engine_client.semaphore.release()

    monkeypatch.setattr(OpenAIServingChat, "chat_completion_stream_generator", fake_stream_generator)

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )

    async def _collect():
        generator = await chat_handler.create_chat_completion(request)
        return [chunk async for chunk in generator]

    chunks = asyncio.run(_collect())

    assert chunks == ["data: chunk\n\n"]
    engine_client.format_and_add_data.assert_awaited()
    assert engine_client.semaphore.acquired == 0


def test_create_completion_non_stream(monkeypatch, engine_client, completion_handler):

    fake_response = CompletionResponse(
        id="cmpl-test",
        model="test-model",
        choices=[CompletionResponseChoice(index=0, text="hello", finish_reason="stop")],
        usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )

    async def fake_full_generator(
        self,
        request,
        num_choices,
        request_id,
        created_time,
        model_name,
        prompt_batched_token_ids,
        text_after_process_list,
    ):
        assert num_choices == 1
        assert prompt_batched_token_ids == [[101, 102, 103]]
        self.engine_client.semaphore.release()
        return fake_response

    monkeypatch.setattr(OpenAIServingCompletion, "completion_full_generator", fake_full_generator)

    request = CompletionRequest(model="test-model", prompt="hello", stream=False)

    response = asyncio.run(completion_handler.create_completion(request))

    assert response == fake_response
    engine_client.format_and_add_data.assert_awaited()
    assert engine_client.semaphore.acquired == 0


def test_create_completion_stream(monkeypatch, engine_client, completion_handler):

    async def fake_stream_generator(
        self,
        request,
        num_choices,
        request_id,
        created_time,
        model_name,
        prompt_batched_token_ids,
        text_after_process_list,
    ):
        assert request.stream is True
        assert num_choices == 1
        assert prompt_batched_token_ids == [[101, 102, 103]]
        try:
            yield "data: chunk\n\n"
        finally:
            self.engine_client.semaphore.release()

    monkeypatch.setattr(OpenAIServingCompletion, "completion_stream_generator", fake_stream_generator)

    request = CompletionRequest(model="test-model", prompt="hello", stream=True)

    async def _collect():
        generator = await completion_handler.create_completion(request)
        return [chunk async for chunk in generator]

    chunks = asyncio.run(_collect())

    assert chunks == ["data: chunk\n\n"]
    engine_client.format_and_add_data.assert_awaited()
    assert engine_client.semaphore.acquired == 0


def test_create_chat_completion_parameter_error(engine_client, chat_handler):
    engine_client.format_and_add_data.side_effect = ParameterError("messages", "invalid")

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
    )

    response = asyncio.run(chat_handler.create_chat_completion(request))

    assert isinstance(response, ErrorResponse)
    assert response.error.param == "messages"
    assert response.error.type == "invalid_request_error"
    assert engine_client.semaphore.acquired == 0


def test_create_completion_parameter_error(engine_client, completion_handler):
    engine_client.format_and_add_data.side_effect = ParameterError("prompt", "invalid")

    request = CompletionRequest(model="test-model", prompt="hello", stream=False)

    response = asyncio.run(completion_handler.create_completion(request))

    assert isinstance(response, ErrorResponse)
    assert response.error.param == "prompt"
    assert response.error.type == "invalid_request_error"
    assert engine_client.semaphore.acquired == 0
