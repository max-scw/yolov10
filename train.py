import logging
from argparse import ArgumentParser

from pathlib import Path
import yaml

from timeit import default_timer

from utils import set_env_variable, cast_logging_level


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

from utils import set_env_variable, set_logging_level, set_process_title, load_yolo_model


def read_yaml_file(file_path) -> dict | None:
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, help="path to data file, i.e. coco128.yaml")

    # model
    parser.add_argument("--model", type=str, help="path to model file, i.e. yolov8n.pt, yolov8n.yaml")
    parser.add_argument("--model-version", type=int, default=None, help="YOLO version number (3, 5, 6, 8, 9, 10)")
    parser.add_argument("--model-type", type=str, default=None, help="YOLO model type. (e.g b, l, m, n, s, x for YOLO version 10)")
    parser.add_argument("--weights", type=str, default=None, help="initial weights path")  # e.g. jameslahm/yolov10n

    parser.add_argument("--freeze", nargs="+", type=int, default=[0],
                        help="freeze first n layers, or freeze list of layer indices during training")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--task", type=str, default="detect", help="YOLO task, i.e. detect, segment, classify, pose")

    # training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16, help="total batch size for all GPUs")
    parser.add_argument("--optimizer", type=str, default="auto", help="optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]")

    # process
    parser.add_argument("--device", default="", help="device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu")
    parser.add_argument('--workers', type=int, default=4, help="number of worker threads for data loading (per RANK if DDP)")

    parser.add_argument("--name", default="exp", help="experiment name, results saved to 'project/name' directory")
    parser.add_argument("--process-title", type=str, default=None, help="Names the process")
    parser.add_argument("--verbose", action="store_true", help="whether to print verbose output")
    # parser.add_argument("--logging-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--config-dir", type=str, default="settings", help="Path to local config dir, e.g. where the 'settings.yaml' is stored to.")

    parser.add_argument("--save-period", type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument("--resume", action="store_true", help="resume training from last checkpoint")

    # augmentation
    parser.add_argument("--albumentations-p", type=float, default=None,
                        help="Probability to apply data augmentation based on the albumentations package.")
    parser.add_argument("--albumentations-config-file", type=str, default=None,
                        help="Path to config file with specifications for transformation functions to augment the training data.")
    # augmentation / hyperparameter
    parser.add_argument("--scale", type=float, default=0.5, help="Hyperparameter for augmentation: image scale (+/- gain)")
    parser.add_argument("--mosaic", type=float, default=1.0,
                        help="Hyperparameter for augmentation: image mosaic (probability)")
    parser.add_argument("--erasing", type=float, default=0.4,
                        help="Hyperparameter for augmentation: probability of random erasing during classification training (0-1)")

    args = parser.parse_args()

    # set local config dir
    if args.config_dir:
        set_env_variable("YOLO_CONFIG_DIR", args.config_dir)

    # load ultralytics library after setting environment variables
    from ultralytics import YOLOv10
    from ultralytics.utils import LOGGING_NAME

    # set logging level
    logger = set_logging_level(LOGGING_NAME, args.logging_level)

    logger.debug(f"Input arguments train.py: {args}")

    set_process_title(args.process_title)

    default_args = read_yaml_file(Path("ultralytics/cfg/default.yaml"))

    if (Path(args.weights).suffix == ".safetensors") and args.model_version and args.model_type:
        logger.info(f"Loading model YOLOv{args.model_version}{args.model_type} from safetensors file: {args.weights}")
        model = load_yolo_model(args.weights, args.model_version, args.model_type)
    elif Path(args.weights).suffix == ".pt":
        logger.info(f"Loading model from pickle file: {args.weights}")
        # load model from pickle file
        model = YOLOv10(model=args.weights)
    elif args.model:
        if args.weights:
            logger.info(f"Download pretrained model: {args.weights}")
            # load pretrained model
            model = YOLOv10.from_pretrained(args.weights)
        else:
            logger.info(f"Create new model with random weights.")
            # create new model from scratch
            model = YOLOv10(task=args.task)
    else:
        raise ValueError("You must provide either --model or --model-version and --model-type.")
    t2 = default_timer()
    logger.debug(f"Building model took {(t2 - t1) * 1000:.4g} ms")

    # Train the model
    model.train(
        **{ky: vl for ky, vl in args.__dict__.items() if ky in default_args}
    )
