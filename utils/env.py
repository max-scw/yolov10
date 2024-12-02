import logging
import os
import re
from ast import literal_eval
import importlib.util

from typing import Union, List, Dict, Any, Union


def import_if_installed(library_name):
    lib = None
    if importlib.util.find_spec(library_name) is not None:
        lib = importlib.import_module(library_name)
        logging.debug(f"{library_name} is installed and has been imported.")
    else:
        logging.debug(f"{library_name} is not installed.")
    return lib


def camel_case_split(identifier):
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", identifier)
    return [m.group(0) for m in matches]


def get_env_variable(key: Union[str, List[str]], default_value, check_for_prefix: bool = False) -> Any:
    prefix = ""
    if check_for_prefix and ("PREFIX" in os.environ):
        prefix = f"{os.environ['PREFIX']}_"

    if isinstance(key, str):
        key = [key]

    for ky in key:
        # name of the environment variable to look for
        nm = f"{prefix}{ky}"
        if nm in os.environ:
            return cast(os.environ[nm])
    return default_value


def get_environment_variables(prefix: str = None, with_prefix: bool = True) -> Dict[str, Any]:
    re_prefix = re.compile(prefix if prefix is not None else "")
    config = dict()
    for ky in os.environ:
        m = re_prefix.match(ky)
        if m:
            nm = ky if with_prefix else ky[m.end():]
            config[nm] = cast(os.environ[ky])
    return config


def default_from_env(key: Union[str, List[str]], default: Any) -> Any:
    return get_env_variable(key, default, check_for_prefix=True)


re_number = re.compile("^[0-9.,]+$")
re_integer = re.compile(r"^\d+$")
re_float = re.compile(r"^((\d+\.(\d+)?)|(\.\d+))$")
re_float_de = re.compile(r"^((\d+,(\d+)?)|(,\d+))$")
re_boolean = re.compile(r"^(true|false)$", re.IGNORECASE | re.ASCII)
re_list_or_tuple_or_dict = re.compile(r"^\s*(\[.*\]|\(.*\)|\{.*\})\s*$", re.ASCII)
re_comma = re.compile(r"^(\".*\")|(\'.*\')$", re.ASCII)


def cast(var: str) -> Union[None, int, float, str, bool]:
    """casting strings to primitive datatypes"""
    if re_number.match(var):
        if re_integer.match(var):  # integer
            var = int(var)
        elif re_float.match(var):  # float
            var = float(var)
        elif re_float_de.match(var):  # float
            var = float(var.replace(",", "."))
    elif var.lower() == "none":
        var = None
    elif re_boolean.match(var):
        var = True if var[0].lower() == "t" else False
    elif re_list_or_tuple_or_dict.match(var):
        var = literal_eval(var)
    elif re_comma.match(var):
        # strip enclosing high comma
        var = var.strip('"').strip('"')
    return var


def set_env_variable(key: str, val) -> bool:
    os.environ[key] = str(val)
    return True
