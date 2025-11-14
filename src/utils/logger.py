"""
Logging utilities for CulturalGaN project.
Provides structured logging with file and console output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for better readability.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        return super().format(record)


def setup_logger(
    name: str = "CulturalGaN",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    file_logging: bool = True
) -> logging.Logger:
    """
    Set up logger with console and file handlers.

    Args:
        name: Logger name
        log_dir: Directory to save log files (default: logs/)
        level: Logging level
        console: Whether to log to console
        file_logging: Whether to log to file

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Format strings
    console_format = '%(levelname)s | %(message)s'
    file_format = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(console_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_logging:
        if log_dir is None:
            log_dir = "logs"

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger


class TrainingLogger:
    """
    Specialized logger for training runs.
    Tracks metrics and provides formatted output.
    """

    def __init__(
        self,
        logger: logging.Logger,
        log_interval: int = 10
    ):
        """
        Initialize training logger.

        Args:
            logger: Base logger
            log_interval: Log every N steps
        """
        self.logger = logger
        self.log_interval = log_interval
        self.step = 0
        self.epoch = 0

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log start of epoch."""
        self.epoch = epoch
        self.logger.info("=" * 60)
        self.logger.info(f"Epoch {epoch}/{total_epochs}")
        self.logger.info("=" * 60)

    def log_step(self, metrics: dict, step: Optional[int] = None):
        """
        Log training step metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (auto-increments if None)
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1

        if self.step % self.log_interval == 0:
            # Format metrics
            metric_str = " | ".join([
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            ])

            self.logger.info(f"Step {self.step} | {metric_str}")

    def log_epoch_end(self, metrics: dict):
        """
        Log end of epoch summary.

        Args:
            metrics: Dictionary of epoch metrics
        """
        self.logger.info("-" * 60)
        self.logger.info("Epoch Summary:")

        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")

        self.logger.info("=" * 60)

    def log_checkpoint(self, checkpoint_path: str, is_best: bool = False):
        """Log checkpoint saving."""
        if is_best:
            self.logger.info(f"✓ Saved BEST checkpoint: {checkpoint_path}")
        else:
            self.logger.info(f"✓ Saved checkpoint: {checkpoint_path}")

    def log_evaluation(self, results: dict):
        """
        Log evaluation results.

        Args:
            results: Dictionary of evaluation metrics
        """
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 60)

        for key, value in results.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")

        self.logger.info("=" * 60)


# Global logger instance
_global_logger = None


def get_logger(
    name: str = "CulturalGaN",
    log_dir: Optional[str] = None,
    reset: bool = False
) -> logging.Logger:
    """
    Get or create global logger instance.

    Args:
        name: Logger name
        log_dir: Log directory
        reset: Whether to reset existing logger

    Returns:
        Logger instance
    """
    global _global_logger

    if _global_logger is None or reset:
        _global_logger = setup_logger(name=name, log_dir=log_dir)

    return _global_logger
