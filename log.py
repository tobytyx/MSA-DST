# -*- coding: utf-8 -*-
import logging
import sys
from typing import cast
try:
    import colorama  # type: ignore
except ImportError:
    colorama = None

try:
    import curses
except ImportError:
    curses = None  # type: ignore


def _stderr_supports_color():
    try:
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            if curses:
                curses.setupterm()
                if curses.tigetnum("colors") > 0:
                    return True
            elif colorama:
                if sys.stderr is getattr(
                    colorama.initialise, "wrapped_stderr", object()
                ):
                    return True
    except Exception:
        pass
    return False


class LogFormatter(logging.Formatter):
    def __init__(self, local_rank, color=True) -> None:
        logging.Formatter.__init__(self, datefmt="%m/%d-%H:%M")
        self.local_rank = local_rank
        if local_rank == -1:
            self._fmt = "%(color)s[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d]%(end_color)s %(message)s"
        else:
            _fmt = "%(color)s[%(levelname)1.1s Local{} %(asctime)s %(module)s:%(lineno)d]%(end_color)s %(message)s"
            self._fmt = _fmt.format(local_rank)
        colors = {
            logging.DEBUG: 4,  # Blue
            logging.INFO: 2,  # Green
            logging.WARNING: 3,  # Yellow
            logging.ERROR: 1,  # Red
        }
        self._colors = {}
        if color and _stderr_supports_color():
            if curses is not None:
                fg_color = curses.tigetstr("setaf") or curses.tigetstr("setf") or b""

                for levelno, code in colors.items():
                    # Convert the terminal control characters from
                    # bytes to unicode strings for easier use with the
                    # logging module.
                    self._colors[levelno] = str(curses.tparm(fg_color, code), "ascii")
                self._normal = str(curses.tigetstr("sgr0"), "ascii")
            else:
                # If curses is not present (currently we'll only get here for
                # colorama on windows), assume hard-coded ANSI color codes.
                for levelno, code in colors.items():
                    self._colors[levelno] = "\033[2;3%dm" % code
                self._normal = "\033[0m"
        else:
            self._normal = ""

    def format(self, record):
        try:
            message = record.getMessage()
            assert isinstance(message, str)
            record.message = message
        except Exception as e:
            record.message = "Bad message (%r): %r" % (e, record.__dict__)

        record.asctime = self.formatTime(record, cast(str, self.datefmt))

        if record.levelno in self._colors:
            record.color = self._colors[record.levelno]
            record.end_color = self._normal
        else:
            record.color = record.end_color = ""

        formatted = self._fmt % record.__dict__

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            lines = [formatted.rstrip()]
            lines.extend(record.exc_text.split("\n"))
            formatted = "\n".join(lines)
        return formatted.replace("\n", "\n    ")

    def get_file_formatter(self):
        if self.local_rank == -1:
            format_str = "[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d]%(message)s"
        else:
            format_str = """[%(levelname)1.1s Local{} %(asctime)s %(module)s:%(lineno)d]%(message)s""".format(
                self.local_rank)
        return logging.Formatter(format_str, datefmt="%m/%d-%H:%M")


def create_logger(log_path=None, logger_name=__name__, local_rank=-1):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = LogFormatter(local_rank)
    # 创建一个handler，用于将日志输出到控制台
    channel = logging.StreamHandler()
    channel.setFormatter(formatter)
    logger.addHandler(channel)

    if log_path:
        # 创建一个handler，用于写入日志文件
        file_handler = logging.FileHandler(filename=log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter.get_file_formatter())
        logger.addHandler(file_handler)
    return logger


def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()
    logger = create_logger("./output/what.log", local_rank=args.local_rank)
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")


if __name__ == "__main__":
    test()
