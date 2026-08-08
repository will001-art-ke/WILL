#!/usr/bin/env python3
"""Report unit test coverage of the notebooks in this repository.

Standard coverage tools cannot be pointed at ``.ipynb`` files, so this script
uses a function level metric: it lists every callable defined in a notebook and
marks it as covered when a test in ``tests/`` requests it through
``load_definitions``.

Usage::

    python tools/notebook_coverage.py            # table sorted by coverage
    python tools/notebook_coverage.py --uncovered  # only the missing callables
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"


def notebook_functions(path: Path) -> set[str]:
    """Names of the functions and classes defined anywhere in a notebook."""
    with path.open(encoding="utf-8") as handle:
        content = json.load(handle)

    names: set[str] = set()
    for cell in content["cells"]:
        if cell.get("cell_type") != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
    return names


def tested_names() -> dict[str, set[str]]:
    """Map notebook file name to the names loaded by the test suite."""
    loaded: dict[str, set[str]] = {}
    for test_file in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
        constants = {
            node.targets[0].id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name != "load_definitions" or not node.args:
                continue
            argument = node.args[0]
            if isinstance(argument, ast.Name):
                notebook = constants[argument.id]
            else:
                notebook = ast.literal_eval(argument)
            names = ast.literal_eval(node.args[1]) if len(node.args) > 1 else []
            loaded.setdefault(notebook, set()).update(names)
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uncovered", action="store_true", help="list the untested callables"
    )
    args = parser.parse_args()

    covered = tested_names()
    rows = []
    for path in sorted(REPO_ROOT.glob("*.ipynb")):
        defined = notebook_functions(path)
        tested = covered.get(path.name, set()) & defined
        ratio = len(tested) / len(defined) if defined else 1.0
        rows.append((ratio, path.name, defined, tested))

    rows.sort(key=lambda row: (row[0], -len(row[2])))

    width = max(len(row[1]) for row in rows)
    print(f"{'notebook'.ljust(width)}  tested/defined  coverage")
    total_defined = total_tested = 0
    for ratio, name, defined, tested in rows:
        total_defined += len(defined)
        total_tested += len(tested)
        counts = f"{len(tested)}/{len(defined)}"
        print(f"{name.ljust(width)}  {counts.center(14)}  {ratio:>7.0%}")
        if args.uncovered and defined - tested:
            for missing in sorted(defined - tested):
                print(f"{' ' * width}    - {missing}")

    overall = total_tested / total_defined if total_defined else 1.0
    print(f"\nTOTAL: {total_tested}/{total_defined} callables covered ({overall:.0%})")


if __name__ == "__main__":
    main()
