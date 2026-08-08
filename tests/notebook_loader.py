"""Load functions defined inside Colab notebooks so they can be unit tested.

The repository stores all of its code in ``.ipynb`` notebooks, so there is no
importable package. This helper parses a notebook, keeps only the safe
top-level nodes of a code cell (imports, function definitions and class
definitions) and executes them in an isolated namespace. Every statement with
side effects (data loading, plotting, model training, ``time.sleep`` ...) is
dropped, which makes the extracted callables cheap and deterministic to test.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

_SAFE_NODES = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def code_cells(notebook: str) -> list[str]:
    """Return the source of every code cell of ``notebook``."""
    path = REPO_ROOT / notebook
    with path.open(encoding="utf-8") as handle:
        content = json.load(handle)
    return [
        "".join(cell["source"])
        for cell in content["cells"]
        if cell.get("cell_type") == "code"
    ]


def _definitions(source: str, names: Optional[Sequence[str]]) -> Optional[ast.Module]:
    """Return a module with the imports/definitions of ``source``.

    ``None`` is returned when the cell defines none of the requested ``names``
    or cannot be parsed (notebooks may contain IPython magics).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    kept = [node for node in tree.body if isinstance(node, _SAFE_NODES)]
    defined = {
        node.name
        for node in kept
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    if names is not None and not defined.intersection(names):
        return None
    return ast.Module(body=kept, type_ignores=[])


def load_definitions(
    notebook: str,
    names: Iterable[str],
    *,
    cell_index: Optional[int] = None,
    inject: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute the definitions of ``notebook`` and return the resulting namespace.

    Args:
        notebook: File name of the notebook, relative to the repository root.
        names: Names that must be present in the returned namespace.
        cell_index: Index within the list of code cells. When omitted every
            cell defining one of ``names`` is executed, so the last definition
            wins -- mirroring how the notebook behaves when run top to bottom.
        inject: Values placed in the namespace before execution, used for the
            globals that a notebook function closes over (thresholds,
            lemmatizers, stop word sets ...).

    Raises:
        KeyError: If a requested name is not defined by the notebook.
    """
    names = list(names)
    namespace: Dict[str, Any] = dict(inject or {})

    cells = code_cells(notebook)
    if cell_index is not None:
        cells = [cells[cell_index]]

    for source in cells:
        module = _definitions(source, None if cell_index is not None else names)
        if module is None:
            continue
        for node in module.body:
            statement = ast.Module(body=[node], type_ignores=[])
            try:
                exec(compile(statement, f"<{notebook}>", "exec"), namespace)  # noqa: S102
            except ImportError:
                # Notebook cells import heavy optional dependencies (tensorflow,
                # seaborn, IPython ...) next to the pure helpers under test.
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    raise

    missing = [name for name in names if name not in namespace]
    if missing:
        raise KeyError(f"{notebook} does not define {missing}")
    return namespace
