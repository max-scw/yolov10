from argparse import ArgumentParser, Namespace
from logging import Logger
import yaml
from pathlib import Path
import re
import warnings

from .env import set_env_variable, cast
from .logging import set_logging_level

from typing import Union, Tuple, List, Dict, Any


def set_process_title(process_title: str = "") -> None:
    if process_title:
        try:
            from setproctitle import setproctitle
            setproctitle(process_title)
        except ModuleNotFoundError as ex:
            warnings.warn(f"Package 'setproctitle' not installed. Process could not be named.")
        except Exception as ex:
            raise Exception(f"Process could not be named: {ex}")


def add_arguments_build_model(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--model-version", type=int, default=None, help="YOLO version number (3, 5, 6, 8, 9, 10)")
    parser.add_argument("--model-type", type=str, default=None, help="YOLO model type. (e.g b, l, m, n, s, x for YOLO version 10)")
    parser.add_argument("--weights", type=str, default=None, help="initial weights path")  # e.g. jameslahm/yolov10n
    return parser


def parse_arguments_defaults(parser: ArgumentParser) -> (Namespace, Logger):

    parser.add_argument("--process-title", type=str, default=None, help="Names the process")
    parser.add_argument("--config-dir", type=str, default="settings", help="Path to local config dir, e.g. where the 'settings.yaml' is stored to.")
    parser.add_argument("--logging-level", type=str, default="INFO", help="Set logging level")

    args = parser.parse_args()

    if isinstance(args.freeze, str):
        args.freeze = cast(args.freeze)

    # set local config dir
    if args.config_dir:
        set_env_variable("YOLO_CONFIG_DIR", args.config_dir)

    # load ultralytics library after setting environment variables
    from ultralytics.utils import LOGGING_NAME

    # set logging level
    logger = set_logging_level(LOGGING_NAME, args.logging_level)

    logger.debug(f"Input arguments train.py: {args}")

    set_process_title(args.process_title)

    return args, logger


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


def build_argument_parser_from_yaml_file(
        config_file: Union[str, Path] = "ultralytics/cfg/default.yaml"
) -> ArgumentParser:
    arguments, config = parse_argument_parser_config_from_file(config_file)

    # build argument parser
    parser = ArgumentParser()
    for arg, kwargs in arguments:
        parser.add_argument(arg, **kwargs)

    return parser, config


def parse_argument_parser_config_from_file(
        config_file: Union[str, Path] = "ultralytics/cfg/default.yaml"
) -> Union[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
    # read default config from YAMl file
    default_args = read_yaml_file(config_file)

    # read entire lines to get hold of the comments
    with open(config_file, "r") as fid:
        lines = fid.readlines()
    # strip linebreaks and empty lines
    lines = [el.strip() for el in lines if len(el) > 5]

    re_comment = re.compile("\s*#\s*")
    i = 0
    arguments = []
    for ky, vl in default_args.items():
        re_kwarg = re.compile(f"{ky}:" + f"\s*{vl}" if vl is not None else "", re.ASCII)
        while True:
            # current line
            ln = lines[i]

            # find the key-value pair
            m1 = re.match(re_kwarg, ln)
            if m1:
                # remaining part of the line
                ln_end = ln[m1.end():]
                # find comment; take it as help message
                m2 = re_comment.match(ln_end)
                arg_help = ln_end[m2.end():] if m2 else ""

                arg_key = ky
                arg_default_value = vl
                arg_type = str if vl is None else type(vl)

                kwargs = dict()
                if arg_type == bool:
                    if vl:
                        arg_key = f"no-{ky}"
                    kwargs["action"] = "store_true"
                else:
                    kwargs["type"] = arg_type
                    kwargs["default"] = arg_default_value
                if help:
                    kwargs["help"] = arg_help
                # argument key
                arg = f"--{arg_key.replace('_', '-')}"

                # append options for this argument
                arguments.append((arg, kwargs))
                # stop while loop
                i += 1
                break
            i += 1
            if i >= len(lines):
                print("stop loop")
                break
        if i >= len(lines):
            print("stop loop")
            break
    return arguments, default_args
