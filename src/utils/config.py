"""Centralized project paths and simple configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_raw: Path
    data_processed: Path
    data_fashion: Path
    data_annotations: Path
    outputs_generated: Path
    outputs_case_studies: Path
    outputs_eval: Path
    outputs_figures: Path
    models_checkpoints: Path
    models_configs: Path
    models_pretrained: Path


def get_paths(start: str | Path | None = None) -> ProjectPaths:
    """Return resolved project paths from any starting location inside repo.

    Args:
        start: Any path within the repository. Defaults to current working dir.
    """
    start_path = Path(start or ".").resolve()
    # Find repository root by walking up to the README.md directory
    root = _find_repo_root(start_path)

    data = root / "data"
    outputs = root / "outputs"
    models = root / "models"

    return ProjectPaths(
        root=root,
        data_raw=data / "raw",
        data_processed=data / "processed",
        data_fashion=data / "fashion_reference",
        data_annotations=data / "annotations",
        outputs_generated=outputs / "generated_designs",
        outputs_case_studies=outputs / "case_studies",
        outputs_eval=outputs / "evaluation_results",
        outputs_figures=outputs / "figures",
        models_checkpoints=models / "checkpoints",
        models_configs=models / "configs",
        models_pretrained=models / "pretrained",
    )


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "README.md").exists():
            return parent
    # Fallback: assume current directory is root
    return start


