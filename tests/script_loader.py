"""Shared loader for the `scripts/` modules under test.

`scripts/` is not an importable package (it holds Hydra CLI entry points, not
library code), so every test file that needs one of its functions used to
inline the same `importlib.util.spec_from_file_location` boilerplate. This
module centralises that pattern in one place.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_script(name: str) -> ModuleType:
    """Loads `scripts/<name>.py` as a module named `<name>_script`.

    Registers the module under `sys.modules[f"{name}_script"]` exactly as
    every test file used to do inline, so dataclasses defined in the script
    and Hydra's config composition behave the same as a normal import.

    Args:
        name: Script filename without the `.py` suffix, e.g. `"localize"`.

    Returns:
        The executed module object.
    """
    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_script"] = module
    spec.loader.exec_module(module)
    return module
