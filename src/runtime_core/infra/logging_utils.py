from __future__ import annotations

# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler, formatter = _build_handler_and_formatter()

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _build_handler_and_formatter() -> tuple[logging.Handler, logging.Formatter]:
    try:
        from rich.logging import RichHandler  # pyright: ignore[reportMissingImports,reportUnknownVariableType]

        handler: logging.Handler = RichHandler(
            markup=True,
            rich_tracebacks=True,
            show_path=False,
        )
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")
        return handler, formatter
    except Exception:  # noqa: BLE001
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="[%X]",
        )
        return handler, formatter
