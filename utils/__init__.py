import warnings

from .env import cast, set_env_variable, get_env_variable
from .logging import cast_logging_level, set_logging_level
from .model import load_yolo_model

def set_process_title(process_title: str = "") -> None:
    if process_title:
        try:
            from setproctitle import setproctitle
            setproctitle(process_title)
        except ModuleNotFoundError as ex:
            warnings.warn(f"Package 'setproctitle' not installed. Process could not be named.")
        except Exception as ex:
            raise Exception(f"Process could not be named: {ex}")


__all__ = [
    "set_env_variable",
    "get_env_variable",
    "cast",
    "cast_logging_level",
    "set_logging_level",
    "load_yolo_model",
    "set_process_title"
]