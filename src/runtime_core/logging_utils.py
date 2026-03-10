from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            markup=True,
            rich_tracebacks=True,
            show_path=False,
        )
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")
    except Exception:  # noqa: BLE001
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="[%X]",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
