import os
from typing import TYPE_CHECKING

from art.utils.output_dirs import get_model_dir

if TYPE_CHECKING:
    from art.model import TrainableModel


def get_step_from_dir(output_dir: str) -> int:
    print("DEBUG get_step_from_dir: output_dir =", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    print("DEBUG get_step_from_dir: checkpoint_dir =", checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        print("DEBUG get_step_from_dir: checkpoint_dir does not exist, returning 0")
        return 0

    subdirs = os.listdir(checkpoint_dir)
    print("DEBUG get_step_from_dir: subdirs =", subdirs)
    numeric_subdirs = [
        subdir for subdir in subdirs
        if os.path.isdir(os.path.join(checkpoint_dir, subdir)) and subdir.isdigit()
    ]
    print("DEBUG get_step_from_dir: numeric_subdirs =", numeric_subdirs)
    result = max((int(d) for d in numeric_subdirs), default=0)
    print("DEBUG get_step_from_dir: returning", result)
    return result


def get_model_step(model: "TrainableModel", art_path: str) -> int:
    return get_step_from_dir(get_model_dir(model=model, art_path=art_path))
