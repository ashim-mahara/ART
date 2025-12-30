"""OpenAI-compatible server functionality for vLLM."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Coroutine

from openai import AsyncOpenAI
from uvicorn.config import LOGGING_CONFIG
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.logger import _DATE_FORMAT, _FORMAT
from vllm.utils.argparse_utils import FlexibleArgumentParser

from ..dev.openai_server import OpenAIServerConfig


def _wrap_sync_llm(llm: Any) -> Any:
    """
    Wrap a sync LLM to expose attributes expected by the OpenAI server.
    The OpenAI server in vLLM 0.12 expects async engine attributes like vllm_config.
    """

    class SyncLLMWrapper:
        """Wrapper around sync LLM to provide async-like interface for OpenAI server."""

        def __init__(self, llm: Any):
            self._llm = llm
            self._engine = llm.llm_engine

        @property
        def vllm_config(self) -> Any:
            return self._engine.vllm_config

        @property
        def model_config(self) -> Any:
            return self._engine.model_config

        @property
        def engine(self) -> Any:
            return self._engine

        @property
        def llm_engine(self) -> Any:
            return self._engine

        async def add_lora(self, lora_request: Any) -> bool:
            # OpenAI server expects this to be async
            return self._engine.add_lora(lora_request)

        async def add_lora_async(self, lora_request: Any) -> bool:
            return self._engine.add_lora(lora_request)

        async def get_tokenizer(self) -> Any:
            # OpenAI server awaits this
            return self._llm.get_tokenizer()

        async def get_tokenizer_async(self) -> Any:
            return self._llm.get_tokenizer()

        def get_tokenizer_sync(self) -> Any:
            return self._llm.get_tokenizer()

        async def get_supported_tasks(self) -> Any:
            return self._engine.get_supported_tasks()

        async def get_supported_tasks_async(self) -> Any:
            return self._engine.get_supported_tasks()

        async def get_model_config(self) -> Any:
            return self._engine.model_config

        @property
        def errored(self) -> bool:
            return False

        @property
        def dead_error(self) -> Exception | None:
            return None

        @property
        def is_running(self) -> bool:
            return True

        @property
        def is_stopped(self) -> bool:
            return False

        async def is_running_async(self) -> bool:
            return True

        async def errored_async(self) -> bool:
            return False

        def __getattr__(self, name: str) -> Any:
            # Delegate to the underlying LLM or engine
            if hasattr(self._llm, name):
                return getattr(self._llm, name)
            if hasattr(self._engine, name):
                return getattr(self._engine, name)
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    return SyncLLMWrapper(llm)


async def openai_server_task(
    engine: EngineClient,
    config: OpenAIServerConfig,
) -> asyncio.Task[None]:
    """
    Starts an asyncio task that runs an OpenAI-compatible server.

    Args:
        engine: The vLLM engine client.
        config: The configuration for the OpenAI-compatible server.

    Returns:
        A running asyncio task for the OpenAI-compatible server. Cancel the task
        to stop the server.
    """
    # Import patches before importing api_server
    from .patches import (
        patch_listen_for_disconnect,
        patch_tool_parser_manager,
        subclass_chat_completion_request,
    )

    # We must subclass ChatCompletionRequest before importing api_server
    # or logprobs will not always be returned
    subclass_chat_completion_request()
    from vllm.entrypoints.llm import LLM
    from vllm.entrypoints.openai import api_server

    patch_listen_for_disconnect()
    patch_tool_parser_manager()
    set_vllm_log_file(config.get("log_file", "vllm.log"))

    # Wrap sync LLM to expose attributes expected by the OpenAI server
    if isinstance(engine, LLM):
        engine = _wrap_sync_llm(engine)

    # Patch engine.add_lora; hopefully temporary
    # For sync LLM, add_lora is on llm_engine
    if hasattr(engine, "add_lora"):
        add_lora = engine.add_lora
    elif hasattr(engine, "llm_engine") and hasattr(engine.llm_engine, "add_lora"):
        add_lora = engine.llm_engine.add_lora
    else:
        # Fallback: no-op
        async def add_lora(lora_request: Any) -> bool:
            return True

    async def _add_lora(lora_request) -> None:
        # Add missing attributes that vLLM expects but unsloth doesn't provide
        if not hasattr(lora_request, "lora_tensors"):
            lora_request.lora_tensors = None
        if not hasattr(lora_request, "tensorizer_config_dict"):
            lora_request.tensorizer_config_dict = None

        result = add_lora(lora_request)
        # Handle both sync and async add_lora
        if asyncio.iscoroutine(result):
            await result

    engine.add_lora = _add_lora

    @asynccontextmanager
    async def build_async_engine_client(
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[EngineClient]:
        yield engine

    api_server.build_async_engine_client = build_async_engine_client
    openai_server_task = asyncio.create_task(_openai_server_coroutine(config))
    server_args = config.get("server_args", {})
    client = AsyncOpenAI(
        api_key=server_args.get("api_key"),
        base_url=f"http://{server_args.get('host', '0.0.0.0')}:{server_args.get('port', 8000)}/v1",
    )

    async def test_client() -> None:
        while True:
            try:
                async for _ in client.models.list():
                    return
            except:  # noqa: E722
                await asyncio.sleep(0.1)

    test_client_task = asyncio.create_task(test_client())
    try:
        timeout = float(os.environ.get("ART_SERVER_TIMEOUT", 30.0))
        done, _ = await asyncio.wait(
            [openai_server_task, test_client_task],
            timeout=timeout,
            return_when="FIRST_COMPLETED",
        )
        if not done:
            raise TimeoutError(
                f"Unable to reach OpenAI-compatible server within {timeout} seconds. You can increase this timeout by setting the ART_SERVER_TIMEOUT environment variable."
            )
        for task in done:
            task.result()

        return openai_server_task
    except Exception:
        openai_server_task.cancel()
        test_client_task.cancel()
        raise


def _openai_server_coroutine(
    config: OpenAIServerConfig,
) -> Coroutine[Any, Any, None]:
    from vllm.entrypoints.openai import api_server

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    engine_args = config.get("engine_args", {})
    server_args = config.get("server_args", {})
    args = []
    for args_dict in [engine_args, server_args]:
        for key, value in args_dict.items():
            if value is None:
                continue
            key_name = key.replace("_", "-")
            if value is True:
                # Boolean True: just --key
                args.append(f"--{key_name}")
            elif value is False:
                # Boolean False: --no-key for BooleanOptionalAction
                args.append(f"--no-{key_name}")
            elif isinstance(value, list):
                # List values: --key item1 --key item2 ...
                for item in value:
                    if item is not None:
                        args.append(f"--{key_name}={item}")
            else:
                # Scalar values: --key=value
                args.append(f"--{key_name}={value}")
    namespace = parser.parse_args(args)
    assert namespace is not None
    validate_parsed_serve_args(namespace)
    return api_server.run_server(
        namespace,
        log_config=get_uvicorn_logging_config(config.get("log_file", "vllm.log")),
    )


def get_uvicorn_logging_config(path: str) -> dict[str, Any]:
    """
    Returns a Uvicorn logging config that writes to the given path.
    """
    return {
        **LOGGING_CONFIG,
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.FileHandler",
                "filename": path,
            },
            "access": {
                "formatter": "default",
                "class": "logging.FileHandler",
                "filename": path,
            },
        },
    }


def set_vllm_log_file(path: str) -> None:
    """
    Sets the vLLM log file to the given path.
    """

    # Create directory for the log file if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Get the vLLM logger
    vllm_logger = logging.getLogger("vllm")

    # Remove existing handlers
    for handler in vllm_logger.handlers[:]:
        vllm_logger.removeHandler(handler)

    # Create a file handler
    file_handler = logging.FileHandler(path)

    # Use the same formatter as vLLM's default
    formatter = logging.Formatter(_FORMAT, _DATE_FORMAT)
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    vllm_logger.addHandler(file_handler)

    # Set log level to filter out DEBUG messages
    vllm_logger.setLevel(logging.INFO)
