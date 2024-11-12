import logging

from .env import cast, get_env_variable

from typing import Union


def cast_logging_level(var: str, default: int = logging.INFO) -> int:
    """Only casts logging levels to their integer values"""
    # cast string if possible
    if isinstance(var, str):
        var = cast(var)

    options = {
        "debug": logging.DEBUG,  # 10
        "info": logging.INFO,  # 20
        "warning": logging.WARNING,  # 30
        "warn": logging.WARN,  # 30
        "error": logging.ERROR,  # 40
        "critical": logging.CRITICAL,  # 50
        "fatal": logging.FATAL,  # 50
        "notset": logging.NOTSET  # 0
    }
    if isinstance(var, int):
        if var not in options.values():
            return default

    elif isinstance(var, str):
        for ky, val in options.items():
            if var.lower() == ky:
                return val
    else:
        return default
    return var


def get_logging_level(key: str = "LOGGING_LEVEL", default: int = logging.INFO) -> int:
    return cast_logging_level(get_env_variable(key, default))


def set_logging_level(name_of_logger: str, logging_level: Union[int, str]) -> logging.Logger:
    """Sets the logging level of the specified logger and all its handlers"""
    logger = logging.getLogger(name_of_logger)
    logging_level = cast_logging_level(logging_level)
    logger.setLevel(logging_level)
    for el in logger.handlers:
        el.setLevel(logging_level)

    return logger

