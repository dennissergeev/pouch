# -*- coding: utf-8 -*-
"""Logging utilities."""
import sys
from datetime import datetime

from loguru import logger


def create_logger(script_path, subdir="logs"):
    """Create a logger using loguru."""
    logpath = script_path.parent / subdir
    logpath.mkdir(exist_ok=True)
    logger.configure(
        handlers=[
            {"sink": sys.stdout, "level": "INFO"},
            {
                "sink": logpath
                / f"log_{script_path.stem}_{datetime.now():%Y-%m-%d_%H%M%S}.log",
                "level": "DEBUG",
            },
        ]
    )
    return logger
