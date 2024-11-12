from pathlib import Path

from safetensors.torch import load_model

from ultralytics import YOLOv10
from ultralytics.utils import LOGGER


def load_yolo_model(
        weights: str | Path | None,
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

    if weights and (Path(weights).exists() and Path(weights).is_file()):
        if weights.suffix == ".safetensors":
            load_model(model, weights)
        else:
            raise ValueError(f"Weights were expected to be saved as safetensors, but file was {weights.as_posix()}")

    return model

def load_yolo_from_file(
        weights: str = None,
        model_version: int = None,
        model_type: str = None,
        task: str = "detect"
):

    if (weights is None or weights == "") or ((Path(weights).suffix == ".safetensors") and model_version and model_type):
        LOGGER.info(f"Loading model YOLOv{model_version}{model_type} from safetensors file: {weights}")
        model = load_yolo_model(weights, model_version, model_type, task=task)
    elif (Path(weights).suffix == ".pt") and Path(weights).is_file():
        LOGGER.info(f"Loading model from pickle file: {weights}")
        # load model from pickle file
        model = YOLOv10(model=weights)
    elif Path(weights).suffix == ".pt":
        # assume hugging face repository
        LOGGER.info(f"Download pretrained model: {weights}")
        # load pretrained model
        model = YOLOv10.from_pretrained(weights)
    else:
        raise ValueError("Unexpected file type")

    return model