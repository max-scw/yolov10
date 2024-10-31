from argparse import ArgumentParser

from ultralytics import YOLOv10
from ultralytics.models.yolov10.train import YOLOv10DetectionTrainer

from safetensors.torch import load_model

from pathlib import Path
import warnings
import yaml


def load_yolo_model(
        weights: str | Path,
        model_version: int,
        model_type: str = "",
        task: str = "detect"
):
    model_versions = (3, 5, 6, 8, 9, 10, 11)
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


def set_process_title(process_title: str = "") -> None:
    if process_title:
        try:
            from setproctitle import setproctitle
            setproctitle(process_title)
        except ModuleNotFoundError as ex:
            warnings.warn(f"Package 'setproctitle' not installed. Process could not be named.")
        except Exception as ex:
            raise Exception(f"Process could not be named: {ex}")


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
    parser.add_argument("--freeze", nargs="+", type=int, default=[0],
                        help="freeze first n layers, or freeze list of layer indices during training")
    parser.add_argument("--weights", type=str, default=None, help="initial weights path")  # e.g. jameslahm/yolov10n
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--task", type=str, default="detect", help="YOLO task, i.e. detect, segment, classify, pose")

    # training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs")
    parser.add_argument("--optimizer", type=str, default="auto", help="optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]")

    # process
    parser.add_argument("--device", default="", help="device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu")
    parser.add_argument('--workers', type=int, default=4, help="number of worker threads for data loading (per RANK if DDP)")

    parser.add_argument("--name", default="exp", help="experiment name, results saved to 'project/name' directory")
    parser.add_argument("--process-title", type=str, default=None, help="Names the process")
    parser.add_argument("--verbose", type=bool, default=True, help="whether to print verbose output")

    parser.add_argument("--save-period", type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument("--save", action="store_true", help="save train checkpoints and predict results")
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


    set_process_title(args.process_title)

    default_args = read_yaml_file(Path("ultralytics/cfg/default.yaml"))


    if args.model_version and args.model_type:
        model = load_yolo_model(args.weights, args.model_version, args.model_type)
    elif args.model:
        if args.weights:
            # load pretrained model
            model = YOLOv10.from_pretrained(args.weights)
        else:
            # create new model from scratch
            model = YOLOv10(task=args.task)
    else:
        raise ValueError("You must provide either --model or --model-version and --model-type.")

    # Train the model
    model.train(
        **{ky: vl for ky, vl in args.__dict__.items() if ky in default_args}
    )
