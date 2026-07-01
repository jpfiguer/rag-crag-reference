"""logging estructurado con structlog"""
from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging() -> None:
    log_format = os.environ.get("LOG_FORMAT", "json").lower()
    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.dev.set_exc_info,
        structlog.processors.format_exc_info,
    ]

    if log_format == "console":
        processors = shared + [structlog.dev.ConsoleRenderer(colors=True)]
    else:
        processors = shared + [
            structlog.processors.EventRenamer("message"),
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level, logging.INFO),
        stream=sys.stdout,
    )
