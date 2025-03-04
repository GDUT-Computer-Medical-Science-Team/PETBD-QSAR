import os.path
import yaml


def __read_from_yml(filename, index):
    if not os.path.exists(filename):
        raise FileNotFoundError("文件不存在")
    if not filename.endswith('yml') and not filename.endswith('yaml'):
        raise FileNotFoundError("输入文件格式错误，指定的文件格式为yml或yaml")
    with open(filename, encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data[index]


def get_denied_organ_names(filename: str) -> list:
    return __read_from_yml(filename, 'denied_organ_names')


def get_target_organ_names(filename: str) -> list:
    return __read_from_yml(filename, 'target_organ_names')


def get_deprecated_organ_names(filename: str) -> dict:
    return __read_from_yml(filename, 'deprecated_organ_names')


def get_time_intervals(filename: str) -> list:
    return __read_from_yml(filename, 'time_intervals')


def get_denied_intervals(filename: str) -> list:
    return __read_from_yml(filename, 'denied_interval_markers')


def get_OCR_error_text(filename: str) -> dict:
    return __read_from_yml(filename, 'OCR_error_text')
