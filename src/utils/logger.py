"""Logging utilities for TruthSim."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# Remove default handler
logger.remove()

# Global logger instance
_logger_configured = False


def setup_logger(
        level: str = "INFO",
        log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        rotation: str = "10 MB",
        retention: str = "7 days",
        format_string: Optional[str] = None,
) -> None:
    """
    Configure the global logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        log_file: Specific log file name
        rotation: When to rotate log files
        retention: How long to keep old logs
        format_string: Custom format string
    """
    global _logger_configured

    if _logger_configured:
        return

    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True,
    )

    # File handler (if log_dir is specified)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            log_file = "truthsim.log"

        logger.add(
            log_path / log_file,
            format=format_string,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    _logger_configured = True


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance.

    Args:
        name: Optional name for the logger context

    Returns:
        Logger instance
    """
    global _logger_configured

    if not _logger_configured:
        setup_logger()

    if name:
        return logger.bind(name=name)

    return logger


# Convenience functions
def debug(message: str, **kwargs):
    """Log a debug message."""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs):
    """Log an info message."""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs):
    """Log a warning message."""
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs):
    """Log an error message."""
    get_logger().error(message, **kwargs)


def critical(message: str, **kwargs):
    """Log a critical message."""
    get_logger().critical(message, **kwargs)
