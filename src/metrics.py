"""Metrics logging utilities for training and evaluation.

This module provides a unified interface for logging metrics to CSV files
during training, evaluation, and hyperparameter tuning.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, TextIO


class MetricsLogger:
    """Context manager for logging metrics to CSV files.

    Handles CSV file creation, header writing, and row appending with automatic
    flushing for real-time monitoring during training.

    Example:
        with MetricsLogger(output_path, ["step", "loss"]) as logger:
            logger.log({"step": 1, "loss": 0.5})
            logger.log({"step": 2, "loss": 0.3})
    """

    def __init__(self, file_path: str | Path, fieldnames: list[str]):
        """Initialize the metrics logger.

        Args:
            file_path: Path where the CSV file will be written.
            fieldnames: List of column names for the CSV file. These define
                the expected keys in dictionaries passed to log().
        """
        self.file_path = Path(file_path)
        self.fieldnames = fieldnames
        self._file_handle: TextIO | None = None
        self._csv_writer: csv.DictWriter | None = None

    def __enter__(self) -> MetricsLogger:
        """Open the CSV file and write the header."""
        self._file_handle = self.file_path.open("w", newline="")
        self._csv_writer = csv.DictWriter(self._file_handle, fieldnames=self.fieldnames)
        self._csv_writer.writeheader()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the CSV file."""
        if self._file_handle is not None:
            self._file_handle.close()

    def log(self, metrics: dict[str, Any]) -> None:
        """Write a row of metrics to the CSV file.

        Args:
            metrics: Dictionary mapping field names to values. Keys must match
                the fieldnames provided during initialization.
        """
        if self._csv_writer is None:
            raise RuntimeError("MetricsLogger must be used as a context manager")

        self._csv_writer.writerow(metrics)
        if self._file_handle is not None:
            self._file_handle.flush()


def create_doubleq_metrics_logger(output_path: str | Path) -> MetricsLogger:
    """Create a metrics logger configured for Double Q-learning training.

    Args:
        output_path: Path where the CSV file will be written.

    Returns:
        MetricsLogger configured with Double Q-learning metric fields.
    """
    fieldnames = [
        "step",
        "episode",
        "epsilon",
        "eval_mean_return",
        "win_rate",
        "draw_rate",
        "loss_rate",
        "mean_len",
    ]
    return MetricsLogger(output_path, fieldnames)


def create_a2c_metrics_logger(output_path: str | Path) -> MetricsLogger:
    """Create a metrics logger configured for A2C-GAE training.

    Args:
        output_path: Path where the CSV file will be written.

    Returns:
        MetricsLogger configured with A2C-GAE metric fields.
    """
    fieldnames = [
        "step",
        "update",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "eval_mean_return",
        "win_rate",
        "draw_rate",
        "loss_rate",
        "mean_len",
    ]
    return MetricsLogger(output_path, fieldnames)
