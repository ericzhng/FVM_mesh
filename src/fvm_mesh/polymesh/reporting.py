# -*- coding: utf-8 -*-
"""
This module provides reporting functions for mesh quality analysis.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .quality import MeshQuality


def format_quality_summary(quality: "MeshQuality") -> str:
    """
    Formats a summary of the computed mesh quality metrics.
    """
    if not quality:
        return "Quality metrics not computed."

    report = []
    report.append(f"\n{'--- Mesh Quality Metrics ---':^80}")
    report.append(_format_metric_table(quality))
    report.append(_format_connectivity_issues(quality))
    return "\n".join(report)


def _format_metric_table(quality: "MeshQuality") -> str:
    """Formats the table of quality metrics."""
    lines = []
    lines.append(f"  {'Metric':<25} {'Min':>15} {'Max':>15} {'Average':>15}")
    lines.append(f"  {'-'*24} {'-'*15} {'-'*15} {'-'*15}")

    lines.append(
        _format_metric_row(
            "Min/Max Volume Ratio",
            np.array(quality.min_max_volume_ratio),
            is_single_value=True,
        )
    )
    lines.append(_format_metric_row("Skewness", quality.cell_skewness_values))
    lines.append(
        _format_metric_row(
            "Non-Orthogonality (deg)", quality.cell_non_orthogonality_values
        )
    )
    lines.append(
        _format_metric_row(
            "Aspect Ratio", quality.cell_aspect_ratio_values, filter_finite=True
        )
    )
    return "\n".join(filter(None, lines))


def _format_metric_row(
    name: str,
    values: np.ndarray,
    is_single_value: bool = False,
    filter_finite: bool = False,
) -> str | None:
    """Formats a single row in the metric table."""
    if is_single_value:
        if values > 0:
            return f"  {name:<25} {values:>15.4f} {'-':>15} {'-':>15}"
        return None

    if values.size > 0 and np.any(values):
        if filter_finite:
            values = values[np.isfinite(values)]
        if values.size > 0:
            min_val, max_val, mean_val = (
                np.min(values),
                np.max(values),
                np.mean(values),
            )
            return f"  {name:<25} {min_val:>15.4f} {max_val:>15.4f} {mean_val:>15.4f}"
    return None


def _format_connectivity_issues(quality: "MeshQuality") -> str:
    """Formats any connectivity issues found."""
    lines = []
    lines.append(f"\n{'--- Connectivity Check ---':^80}")
    if quality.connectivity_issues:
        lines.append("  Issues Found:")
        for issue in quality.connectivity_issues:
            lines.append(f"    - {issue}")
    else:
        lines.append("  No connectivity issues found.")
    return "\n".join(lines)
