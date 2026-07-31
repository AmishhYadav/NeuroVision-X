"""Logging setup for NeuroVision-X.

Provides a single entry point, ``setup_logging``, that configures Python's
standard-library root logger for readable console output (and, optionally,
file output) in a way that is safe to call more than once per process.
"""

# NOTE: this module is itself named ``logging.py`` and lives inside the
# ``neurovision.utils`` package. Python 3 only uses absolute imports, so the
# line below resolves to the standard library's ``logging`` module, not to
# this file. It looks like a self-import at a glance, but it is not one.
import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: int | str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Configure the root logger for console (and optionally file) output.

    Args:
        level: Logging level as an int (e.g. ``logging.DEBUG``) or a
            case-insensitive level name (e.g. ``"debug"``).
        log_file: Optional path to a log file. Its parent directory is
            created if it does not already exist. If None, only the console
            handler is attached.

    Returns:
        The configured root logger.

    Raises:
        ValueError: If ``level`` is a string that does not name a known
            logging level.
    """
    if isinstance(level, str):
        level_name = level.upper()
        resolved_level = logging.getLevelName(level_name)
        # logging.getLevelName() has a quirky dual role: for a valid level
        # name it returns the int level, but for an unrecognized name it
        # returns a *string* like "Level FOO" instead of raising. So an int
        # check is how we detect an invalid level name here.
        if not isinstance(resolved_level, int):
            raise ValueError(f"Unknown logging level: {level!r}")
    else:
        resolved_level = level

    root = logging.getLogger()

    # Remove and close any handlers left over from a previous call. Without
    # this, calling setup_logging() twice (e.g. re-running a Kaggle notebook
    # cell) attaches a second set of handlers, and every subsequent log line
    # is printed twice (or N times). We deliberately avoid logging.basicConfig
    # here, since it silently no-ops once any handler is already attached.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(resolved_level)
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Kaggle notebook cells render stdout more predictably than stderr, and
    # a terminal-read training log is the common case, so stdout it is.
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Pin the encoding: FileHandler otherwise uses the platform default,
        # which differs between macOS and the Linux image Kaggle runs.
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return root
