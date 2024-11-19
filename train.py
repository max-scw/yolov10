import logging
from argparse import ArgumentParser

from pathlib import Path
import yaml

from timeit import default_timer

from utils import set_env_variable, set_logging_level, set_process_title, load_yolo_from_file


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

    # TODO: argparser utils
    parser.add_argument("--config-dir", type=str, default="settings", help="Path to local config dir, e.g. where the 'settings.yaml' is stored to.")
    parser.add_argument("--logging-level", type=str, default="INFO", help="Set logging level")


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

    # freeze all layers up to the given layer if only one number was provided
    if len(args.freeze) == 1:
        args.freeze = list(range(0, args.freeze[0]))

    t0 = default_timer()
    default_args = read_yaml_file(Path("ultralytics/cfg/default.yaml"))

    model = load_yolo_from_file(args.weights, args.model_version, args.model_type, args.task)

    # Train the model
    model.train(
        **{ky: vl for ky, vl in args.__dict__.items() if ky in default_args}
    )
