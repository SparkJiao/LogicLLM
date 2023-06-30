r"""
:class:`Logger` 是记录日志的模块，**logger** 封装了 logging 模块的 Logger，
具体使用方式与直接使用 :class:`logging.Logger` 相同，同时也新增一些简单好用的API

使用方式::

    # logger 可以和 logging.Logger 一样使用
    logger.info('your msg')
    logger.error('your msg')

    # logger 新增的API
    # 将日志输出到文件，以及输出的日志等级
    logger.add_file('/path/to/log', level='INFO')
    # 定义在命令行中的显示格式和日志等级
    logger.set_stdout('tqdm', level='WARN')
    # 仅警告一次
    logger.warning_once('your msg')
    # 分布式训练下，仅在 rank 0 输出警告
    logger.rank_zero_warning('your msg')

"""


import logging
import logging.config
from logging import DEBUG, ERROR, INFO, WARNING, CRITICAL, raiseExceptions
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Union
from rich.logging import RichHandler
import datetime
import torch

__all__ = [
    'logger'
]

from .handler import StdoutStreamHandler, TqdmLoggingHandler


ROOT_NAME = 'LOMO'


class LoggerSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(LoggerSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LOMOLogger(logging.Logger, metaclass=LoggerSingleton):
    def __init__(self, name):
        super().__init__(name)
        self._warning_msgs = set()

    def add_file(self, path: Optional[Union[str, Path]] = None, level='AUTO', remove_other_handlers: bool = False,
                 mode: str = "w"):
        """
        将日志输出到 path 中。

        :param path: 若 path 为文件路径（通过 path 是否包含后缀判定 path 是否表示文件名，例如 output.log 会被认为是文件，而
                output 则认为是文件夹）则直接写入到给定文件中；如果判定为文件夹，则是在该文件夹下以 时间戳 创建一个日志文件。
        :param level: 可选 ['INFO', 'WARNING', 'DEBUG', 'ERROR', 'AUTO'], 其中AUTO表示根据环境变量"LOMO_LOG_LEVEL'进行
            设置。
        :param remove_other_handlers: 是否移除其它 handler ，如果移除，则terminal中将不会有 log 输出。
        :param mode: 可选为['w', 'a']，如果传入的 path 是存在的文件，'w' 会覆盖原有内容 'a' 则会在文件结尾处继续添加。
        :return:
        """
        r"""添加日志输出文件和输出级别"""
        if level == 'AUTO':
            level = parse_level()
        return _add_file_handler(self, path, level, remove_other_handlers, mode)

    def set_stdout(self, stdout: str = 'raw', level: str = 'AUTO'):
        """
        设置 log 的 terminal 输出形式。

        :param stdout: 可选['rich', 'naive', 'raw', 'none']。
        :param level: 可选 ['INFO', 'WARNING', 'DEBUG', 'ERROR', 'AUTO'], 其中AUTO表示根据环境变量"LOMO_LOG_LEVEL'进行
            设置。
        :return:
        """
        r"""设置标准输出格式和输出级别"""
        if level == 'AUTO':
            level = parse_level()
        return _set_stdout_handler(self, stdout, level)

    def debug(self, msg, *args, **kwargs):
        """
        Delegate a debug call to the underlying log.
        """
        if self.isEnabledFor(DEBUG):
            kwargs = self._add_rank_info(kwargs)
            self._log(DEBUG, msg, args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Delegate an info call to the underlying log.
        """
        if self.isEnabledFor(INFO):
            kwargs = self._add_rank_info(kwargs)
            self._log(INFO, msg, args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Delegate a warning call to the underlying log.
        """
        if self.isEnabledFor(WARNING):
            kwargs = self._add_rank_info(kwargs)
            self._log(WARNING, msg, args, **kwargs)

    def warning_once(self, msg, *args, **kwargs):
        """
        相同的 warning 内容只会 warning 一次

        :param msg:
        :param args:
        :param kwargs:
        :return:
        """
        if msg not in self._warning_msgs:
            if self.isEnabledFor(WARNING):
                kwargs = self._add_rank_info(kwargs)
                self._log(WARNING, msg, args, **kwargs)
            self._warning_msgs.add(msg)

    def rank_zero_warning(self, msg, *args, once=False, **kwargs):
        """
        只在 rank 0 上 warning 。

        :param msg:
        :param args:
        :param once: 是否只 warning 一次
        :param kwargs:
        :return:
        """
        if os.environ.get('LOCAL_RANK', 0) == 0:
            if once:
                if msg in self._warning_msgs:
                    return
                self._warning_msgs.add(msg)

            if self.isEnabledFor(WARNING):
                kwargs = self._add_rank_info(kwargs)
                self._log(WARNING, msg, args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        if self.isEnabledFor(WARNING):
            kwargs = self._add_rank_info(kwargs)
            self._log(WARNING, msg, args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Delegate an error call to the underlying log.
        """
        if self.isEnabledFor(ERROR):
            kwargs = self._add_rank_info(kwargs)
            self._log(ERROR, msg, args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        """
        Delegate an exception call to the underlying log.
        """
        kwargs = self._add_rank_info(kwargs)
        self.error(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Delegate a critical call to the underlying log.
        """
        if self.isEnabledFor(CRITICAL):
            kwargs = self._add_rank_info(kwargs)
            self._log(CRITICAL, msg, args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        """
        Delegate a log call to the underlying log, after adding
        contextual information from this adapter instance.
        """
        if not isinstance(level, int):
            if raiseExceptions:
                raise TypeError("level must be an integer")
            else:
                return
        if self.isEnabledFor(level):
            kwargs = self._add_rank_info(kwargs)
            self._log(level, msg, args, **kwargs)

    def _add_rank_info(self, kwargs):
        if torch.distributed.is_initialized():
            extra = kwargs.get('extra', {})
            extra.update({"rank": int(os.environ.get('LOCAL_RANK', 0))})
            kwargs["extra"] = extra
        return kwargs

    def setLevel(self, level) -> None:
        """
        设置当前 logger 以及其 handler 的 log 级别

        :param level:
        :return:
        """
        if isinstance(level, str):
            level = level.upper()
        super().setLevel(level)
        for handler in self.handlers:
            handler.setLevel(level)

    def _set_distributed(self):
        """
        在 LOMO 拉起进程的时候，调用一下这个方法，使得能够输出 rank 信息

        :return:
        """
        for handler in self.handlers:
            if isinstance(handler, logging.FileHandler):
                formatter = logging.Formatter(fmt='Rank: %(rank)s - %(asctime)s - %(module)s - [%(levelname)s] - %(message)s',
                                           datefmt='%Y/%m/%d %H:%M:%S')
            else:
                formatter = logging.Formatter('Rank: %(rank)s - %(message)s')
            handler.setFormatter(formatter)


def _get_level(level):
    if not isinstance(level, int):
        level = level.lower()
        level = {'info': logging.INFO, 'debug': logging.DEBUG,
                 'warn': logging.WARN, 'warning': logging.WARNING,
                 'error': logging.ERROR}[level]
    return level


def _add_file_handler(_logger: logging.Logger, path: Optional[Union[str, Path]] = None, level: str = 'INFO',
                      remove_other_handlers: bool = False, mode: str = "w"):
    if path is None:
        path = Path.cwd()
    if isinstance(path, str):
        path = Path(path)
    if not isinstance(path, Path):
        raise TypeError("Parameter `path` can only be `str` or `pathlib.Path` type.")
    if not path.exists():
        head, tail = os.path.splitext(path)
        if tail == '':  # 说明没有后缀，理解为是一个folder
            path.mkdir(parents=True, exist_ok=True)
        else:
            # 主进程会帮助我们创建文件夹，但是由于主从进程几乎是同步的，因此到这里时子进程也会尝试创建文件夹，即使主进程会做这件事情；
            dirname = os.path.dirname(path)
            os.makedirs(dirname, exist_ok=True)
    if path.is_dir():
        path = path.joinpath(os.environ.get('LOGGING_TIME', f"{datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S_%f')}") + '.log')

    if not isinstance(remove_other_handlers, bool):
        raise TypeError("Parameter `remove_other_handlers` can only be `bool` type.")

    if not isinstance(mode, str):
        raise TypeError("Parameter 'evaluate_fn' can only be `str` type.")
    if mode not in {"w", "a"}:
        raise ValueError("Parameter `evaluate_fn` can only be one of these values: ('w', 'a').")

    for h in _logger.handlers:
        if isinstance(h, logging.FileHandler):
            if os.path.abspath(path) == h.baseFilename:
                # file path already added
                return

    # File Handler
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        if os.path.exists(path):
            assert os.path.isfile(path)
            warnings.warn('log already exists in {}'.format(path))

    dirname = os.path.abspath(os.path.dirname(path))
    os.makedirs(dirname, exist_ok=True)

    # 这里只要检测到是分布式训练，我们就将 evaluate_fn 改为 "a"；这样会导致的一个问题在于，如果第二次训练也是分布式训练，logger记录的log不会重新
    #  覆盖掉原文件，而是会接着上一次的 log 继续添加；
    # 这样做主要是为了解决这样的情形所导致的问题：在分布式训练中，进程 1 比 进程 0 先运行到这里，然后使得进程 0 将进程 1 的 log 覆盖掉；
    # if torch.distributed.is_initialized():# and int(os.environ.get(LOMO_GLOBAL_RANK, 0)) != 0:
    #     mode = "a"

    file_handler = logging.FileHandler(path, mode=mode)
    logger.info(f"Writing log to file:{os.path.abspath(path)}")
    file_handler.setLevel(_get_level(level))

    if torch.distributed.is_initialized():
        file_formatter = logging.Formatter(fmt='Rank: %(rank)s - %(asctime)s - %(module)s - [%(levelname)s] - %(message)s',
                                           datefmt='%Y/%m/%d %H:%M:%S')
    else:
        file_formatter = logging.Formatter(fmt='%(asctime)s - %(module)s - [%(levelname)s] - %(message)s',
                                           datefmt='%Y/%m/%d %H:%M:%S')

    file_handler.setFormatter(file_formatter)
    _logger.addHandler(file_handler)

    if remove_other_handlers:
        _need_remove_handlers = []
        for i, h in enumerate(_logger.handlers):
            if not isinstance(h, logging.FileHandler):
                _need_remove_handlers.append(h)
        for handler in _need_remove_handlers:
            _logger.removeHandler(handler)

    return file_handler


def _set_stdout_handler(_logger, stdout='raw', level='INFO'):
    level = _get_level(level)
    supported_stdout = ['none', 'raw', 'tqdm', 'naive', 'rich']
    if stdout not in supported_stdout:
        raise ValueError('stdout must in one of {}'.format(supported_stdout))
    # make sure to initialize _logger only once
    stream_handler = None
    _handlers = (logging.StreamHandler, TqdmLoggingHandler, StdoutStreamHandler, RichHandler)
    for i, h in enumerate(_logger.handlers):
        if isinstance(h, _handlers):
            stream_handler = h
            break
    if stream_handler is not None:
        _logger.removeHandler(stream_handler)
        del stream_handler

    # Stream Handler
    if stdout == 'raw':
        stream_handler = StdoutStreamHandler()
    elif stdout == 'rich':
        stream_handler = RichHandler(level=level, log_time_format="[%X]")
    elif stdout == 'naive':
        stream_handler = logging.StreamHandler(sys.stdout)
    elif stdout == 'tqdm':
        stream_handler = TqdmLoggingHandler(level)
    else:
        stream_handler = None

    if stream_handler is not None:
        if torch.distributed.is_initialized():
            stream_formatter = logging.Formatter('Rank: %(rank)s - %(message)s')
        else:
            stream_formatter = logging.Formatter('%(message)s')
        stream_handler.setLevel(level)
        stream_handler.setFormatter(stream_formatter)
        _logger.addHandler(stream_handler)

    return stream_handler


def _init_logger(path=None, stdout='rich', level='INFO'):
    r"""initialize _logger"""
    level = _get_level(level)

    logger = LOMOLogger(ROOT_NAME)

    logger.propagate = False

    _set_stdout_handler(logger, stdout, level)

    # File Handler
    if path is not None:
        _add_file_handler(logger, path, level)

    logger.setLevel(level)

    return logger


def parse_level():
    level = 'WARNING' if int(os.environ.get('LOCAL_RANK', 0)) != 0 else "INFO"
    return level


logger = _init_logger(path=None, stdout='rich', level=parse_level())
logger.debug("The environment variables are as following:")
logger.debug(os.environ)
