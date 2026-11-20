import contextlib
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Final, Optional, Generator
import colorlog


class RAGLogger(logging.Logger):
    RAG_LOG_NAME: Final[str] = "atlas_rag.log"
    RAG_LOGGER: Final[str] = "atlas_rag_logger"

    def __init__(self, name: str = RAG_LOGGER) -> None:
        super(RAGLogger, self).__init__(name, logging.DEBUG)
        self.__color_fmt = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s] %(levelname)s  %(pathname)s:%(lineno)d - %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
        self.__fmt = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s  %(pathname)s:%(lineno)d - %(message)s"
        )
        self.__console_handler = logging.StreamHandler()
        self.__console_handler.setLevel(logging.DEBUG)
        self.__console_handler.setFormatter(self.__color_fmt)
        self.addHandler(self.__console_handler)
        self.__log_file_path: Optional[Path] = None

    def closs_console_log(self) -> None:
        self.removeHandler(self.__console_handler)

    def open_console_log(self) -> None:
        self.addHandler(self.__console_handler)

    def closs_file_log(self) -> None:
        if self.__log_file_path:
            self.removeHandler(self.__file_handler)

    def open_file_log(self) -> None:
        if self.__log_file_path:
            self.addHandler(self.__file_handler)

    def set_terminator(self, end: str = "\n") -> None:
        self.__console_handler.terminator = end
        if self.__log_file_path:
            self.__file_handler.terminator = end

    @contextlib.contextmanager
    def terminator_manager(self) -> Generator[None, None, None]:
        try:
            terminator_old = self.__console_handler.terminator
            yield
        finally:
            self.__console_handler.terminator = terminator_old

    def set_log_path(
        self,
        log_dir_path: Path,
        when: str = "midnight",
        backup_count: int = 30,
        log_level: int = logging.INFO,
    ) -> None:
        self.__log_file_path = log_dir_path.joinpath(self.RAG_LOG_NAME)
        self.__file_handler = TimedRotatingFileHandler(
            filename=self.__log_file_path,
            when=when,
            backupCount=backup_count,
            encoding="utf-8",
            interval=1,
        )
        self.__file_handler.suffix = "%Y-%m-%d.log"
        self.__file_handler.setLevel(log_level)
        self.__file_handler.setFormatter(self.__fmt)
        self.addHandler(self.__file_handler)

    def get_log_file_path(self) -> Path:
        if self.__log_file_path is None:
            raise FileNotFoundError("your must set log file directory path first.")
        return self.__log_file_path


def transform_log_level(log_level: str) -> int:
    if log_level.lower() == "debug":
        return logging.DEBUG
    elif log_level.lower() == "info":
        return logging.INFO
    elif log_level.lower() == "warning":
        return logging.WARNING
    elif log_level.lower() == "error":
        return logging.ERROR
    elif log_level.lower() == "critical":
        return logging.CRITICAL
    return logging.INFO


rag_logger: RAGLogger = RAGLogger()
