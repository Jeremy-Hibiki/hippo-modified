from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with a specific name and optional file logging.

    Args:
        name (str): Logger name, typically the module's `__name__`.
        log_file (str): Log file name. If None, defaults to "<name>.log" under the logs directory.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    return logger
