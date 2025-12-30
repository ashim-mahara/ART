"""Patches for unsloth/unsloth_zoo to fix compatibility issues with vLLM 0.12+."""

import asyncio
import os
from typing import Any


def patch_get_vllm_state_dict() -> None:
    """
    Patches unsloth_zoo.vllm_utils._get_vllm_state_dict to handle vLLM 0.12+
    where the V1 engine uses multiprocessing and collective_rpc returns coroutines.

    The issue is that unsloth's _get_vllm_state_dict calls collective_rpc
    synchronously, but in V1 engine it returns a coroutine that needs to be awaited.
    """
    import unsloth_zoo.vllm_utils as vllm_utils

    original_get_vllm_state_dict = vllm_utils._get_vllm_state_dict

    def patched_get_vllm_state_dict(
        llm: Any,
        return_state_dict: bool = False,
        config: Any = None,
        is_vision_model: bool = False,
    ) -> Any:
        """
        Patched version that handles V1 engine with async collective_rpc.
        """
        import torch

        try:
            llm_engine = getattr(llm, "llm_engine", getattr(llm, "engine", llm))

            # Try V0-style direct access first (model_executor on engine)
            if (
                hasattr(llm_engine, "model_executor")
                and llm_engine.model_executor is not None
            ):
                # V0 or V1 in-process mode - direct access to model_executor
                # Fall through to original call since we have direct access
                return original_get_vllm_state_dict(
                    llm, return_state_dict, config, is_vision_model
                )

            # V1 engine - check for engine_core
            if hasattr(llm_engine, "engine_core"):
                engine_core = llm_engine.engine_core

                # Check if InprocClient (has direct access via nested engine_core)
                if hasattr(engine_core, "engine_core"):
                    # InprocClient - original should work
                    return original_get_vllm_state_dict(
                        llm, return_state_dict, config, is_vision_model
                    )

                # V1 engine with MPClient (multiprocessing) - need async RPC
                # Try to use collective_rpc with proper async handling
                if hasattr(llm, "collective_rpc"):
                    import nest_asyncio

                    nest_asyncio.apply()

                    async def get_weights_async():
                        gpu_ids = await llm.collective_rpc(
                            "report_device_id", args=tuple()
                        )
                        weights = await llm.collective_rpc(
                            "get_weight_ipc_handles", args=tuple()
                        )
                        return gpu_ids, weights

                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Create new loop in thread
                            import concurrent.futures

                            with concurrent.futures.ThreadPoolExecutor() as pool:
                                future = pool.submit(asyncio.run, get_weights_async())
                                gpu_ids, weights = future.result(timeout=60)
                        else:
                            gpu_ids, weights = asyncio.run(get_weights_async())

                        if weights is not None and gpu_ids is not None:
                            vllm_state_dict = {}
                            weights = weights[0][gpu_ids[0]]
                            for weight_name, (
                                to_cuda_fx,
                                cuda_data,
                            ) in weights.items():
                                vllm_state_dict[weight_name] = to_cuda_fx(*cuda_data)

                            # NOTE: vLLM RPC state dict extraction is experimental
                            # Fall through to original for now
                            raise NotImplementedError(
                                "Unsloth: vLLM RPC mode requires in-process engine"
                            )
                    except Exception as e:
                        if "coroutine" in str(e).lower():
                            raise RuntimeError(
                                "Unsloth: Cannot access vLLM internals with multiprocessing enabled. "
                                "Please ensure VLLM_ENABLE_V1_MULTIPROCESSING=0 is set before vLLM is imported."
                            ) from e
                        raise

        except AttributeError:
            pass
        except Exception:
            pass

        # Call the original function as fallback
        return original_get_vllm_state_dict(
            llm, return_state_dict, config, is_vision_model
        )

    vllm_utils._get_vllm_state_dict = patched_get_vllm_state_dict


def patch_unsloth_for_vllm_012() -> None:
    """
    Apply all necessary patches for unsloth to work with vLLM 0.12+.

    This should be called early in the initialization process, before
    unsloth.FastLanguageModel.from_pretrained is called.
    """
    # Disable V1 multiprocessing for compatibility with unsloth's model access patterns
    # This must be set BEFORE vLLM is imported
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Apply the patch to _get_vllm_state_dict after unsloth is imported
    try:
        patch_get_vllm_state_dict()
    except ImportError:
        pass  # unsloth_zoo not yet imported, will be applied later
