from __future__ import annotations

import logging
from pathlib import Path

from fapbench.utils import close_logger, configure_logger


def test_close_logger_releases_all_per_run_handlers(tmp_path: Path) -> None:
    loggers: list[logging.Logger] = []
    for index in range(80):
        logger = configure_logger(
            tmp_path / f"run_{index}",
            name=f"fapbench.test.cleanup.{index}",
            console=False,
        )
        logger.info("completed")
        close_logger(logger)
        loggers.append(logger)

    assert all(not logger.handlers for logger in loggers)

    # A closed FileHandler has no live stream/file descriptor.  This is the
    # property required by long sequential publication grids on macOS.
    for logger in loggers:
        assert not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
