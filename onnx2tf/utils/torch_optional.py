from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class OptionalPyTorchDependencyError(RuntimeError):
    """Raised when a PyTorch-backed feature is requested without PyTorch."""


@dataclass(frozen=True)
class PyTorchModules:
    torch: Any


_TORCH_MODULES: Optional[PyTorchModules] = None


def _build_missing_dependency_message(feature: str) -> str:
    normalized_feature = str(feature).strip() or "PyTorch-backed onnx2tf feature"
    return (
        f"{normalized_feature} requires optional dependency: torch. "
        'Install it with: pip install "onnx2tf[torch]"'
    )


def require_torch(feature: str) -> PyTorchModules:
    global _TORCH_MODULES
    if _TORCH_MODULES is not None:
        return _TORCH_MODULES

    try:
        import torch
    except ModuleNotFoundError as ex:
        missing_name = str(getattr(ex, "name", "") or "")
        if missing_name == "torch":
            raise OptionalPyTorchDependencyError(
                _build_missing_dependency_message(feature)
            ) from ex
        raise
    except ImportError as ex:
        missing_name = str(getattr(ex, "name", "") or "")
        if missing_name == "torch":
            raise OptionalPyTorchDependencyError(
                _build_missing_dependency_message(feature)
            ) from ex
        raise
    except Exception:
        raise

    _TORCH_MODULES = PyTorchModules(
        torch=torch,
    )
    return _TORCH_MODULES
