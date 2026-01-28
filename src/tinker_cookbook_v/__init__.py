from __future__ import annotations

import importlib
from pathlib import Path
import sys


def _ensure_local_tinker_cookbook() -> None:
    src_root = Path(__file__).resolve().parent.parent
    src_root_str = str(src_root)
    if sys.path[0] != src_root_str:
        if src_root_str in sys.path:
            sys.path.remove(src_root_str)
        sys.path.insert(0, src_root_str)

    existing = sys.modules.get("tinker_cookbook")
    if existing is None:
        return
    existing_file = getattr(existing, "__file__", "")
    try:
        existing_path = Path(existing_file).resolve()
    except Exception:
        existing_path = None
    if existing_path is None or not str(existing_path).startswith(src_root_str):
        del sys.modules["tinker_cookbook"]


_ensure_local_tinker_cookbook()

renderers = importlib.import_module("tinker_cookbook.renderers")
tokenizer_utils = importlib.import_module("tinker_cookbook.tokenizer_utils")
image_processing_utils = importlib.import_module(
    "tinker_cookbook.image_processing_utils"
)
hyperparam_utils = importlib.import_module("tinker_cookbook.hyperparam_utils")
utils = importlib.import_module("tinker_cookbook.utils")
misc_utils = importlib.import_module("tinker_cookbook.utils.misc_utils")

sys.modules[__name__ + ".renderers"] = renderers
sys.modules[__name__ + ".tokenizer_utils"] = tokenizer_utils
sys.modules[__name__ + ".image_processing_utils"] = image_processing_utils
sys.modules[__name__ + ".hyperparam_utils"] = hyperparam_utils
sys.modules[__name__ + ".utils"] = utils
sys.modules[__name__ + ".utils.misc_utils"] = misc_utils

__all__ = [
    "renderers",
    "tokenizer_utils",
    "image_processing_utils",
    "hyperparam_utils",
    "utils",
    "misc_utils",
]
