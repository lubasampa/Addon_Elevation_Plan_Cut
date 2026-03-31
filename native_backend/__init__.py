"""Namespaced native backend modules for the Mesh Cut extension."""

from __future__ import annotations

import importlib
import sys


def load_meshcut_parallel():
    """Load the compiled backend and drop short-name aliases.

    Blender extensions warn when compiled modules from an extension remain
    registered as top-level modules in ``sys.modules``. Some extension import
    paths create those aliases automatically, so clean them up immediately after
    import while keeping the fully-qualified package name loaded.
    """
    module = importlib.import_module(f"{__name__}.meshcut_parallel")
    sys.modules.pop("meshcut_parallel", None)
    sys.modules.pop("native_backend.meshcut_parallel", None)
    return module
