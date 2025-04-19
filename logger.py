import logging
from typing import Optional

class Logger:
    def __init__(
        self,
        name: str = __name__,
        level: int = logging.INFO,
        log_to_file: bool = False,
        file_path: str = "log/app.log",
    ):
        """
        A simple wrapper around Python's logging.Logger.

        Args:
            name:         Logger name (typically __name__ of the module).
            level:        Minimum level to capture (DEBUG, INFO, etc.).
            log_to_file:  Whether to also write logs to a file.
            file_path:    Path to the log file (if log_to_file=True).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        fmt = "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(message)s"
        formatter = logging.Formatter(fmt)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Optional file handler
        if log_to_file:
            fh = logging.FileHandler(file_path)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
