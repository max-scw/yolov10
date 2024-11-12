from pathlib import Path

from safetensors.torch import load_model
from ultralytics import YOLOv10


def load_yolo_model(
        weights: str | Path,
        model_version: int,
        model_type: str = "",
        task: str = "detect"
):
    # check settings
    model_versions = (3, 5, 6, 8, 9, 10)
    if model_version not in model_versions:
        raise ValueError(f"Available model versions are {model_versions} but you requested: {model_version}.")

    # build path to model config
    path_to_config = Path(f"ultralytics/cfg/models/v{model_version}")
    path_to_model_config = path_to_config /f"yolov{model_version}{model_type}.yaml"
    if not path_to_model_config.exists():
        raise ValueError(
            f"Model type yolov{model_versions}{model_type} is not available. "
            f"The following types are available for YOLOv{model_versions}: {[fl.name for fl in path_to_model_config.glob('*.yaml')]}"
        )

    model = YOLOv10(model=path_to_model_config.as_posix(), task=task)

    weights = Path(weights)
    if weights.exists() and weights.is_file():
        if weights.suffix == ".safetensors":
            load_model(model, weights)
        else:
            raise ValueError(f"Weights were expected to be saved as safetensors, but file was {weights.as_posix()}")

    return model
