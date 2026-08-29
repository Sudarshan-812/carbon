"""Coverage-ish summary: which ``src/`` modules the test suite exercises.

Static only (no line data): for each ``src/*.py`` count the test files that
import it and the total ``test_*`` functions across the suite.

    .venv/Scripts/python.exe scripts/coverage_report.py
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TESTS = ROOT / "tests"


def _count_test_funcs(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        1 for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    )


def main() -> int:
    test_files = sorted(TESTS.glob("test_*.py"))
    modules = sorted(p.stem for p in SRC.glob("*.py") if p.stem != "__init__")

    references: dict[str, list[str]] = {m: [] for m in modules}
    for test_file in test_files:
        text = test_file.read_text(encoding="utf-8")
        for module in modules:
            if re.search(rf"\bsrc\.{module}\b", text) or re.search(
                rf"\bfrom src import\b[^\n]*\b{module}\b", text
            ):
                references[module].append(test_file.stem.removeprefix("test_"))

    total_funcs = sum(_count_test_funcs(f) for f in test_files)
    print(f"test files       : {len(test_files)}")
    print(f"test functions   : {total_funcs}")
    print(f"src modules       : {len(modules)}")
    print()
    print(f"{'module':16}{'#tests files':>13}  {'status':<6} referencing test files")
    print("-" * 72)

    gaps: list[str] = []
    for module in modules:
        hits = references[module]
        status = "ok" if hits else "GAP"
        if not hits:
            gaps.append(module)
        print(f"{module:16}{len(hits):>13}  {status:<6} {', '.join(sorted(set(hits))) or '-'}")

    print()
    if gaps:
        print("modules without a directly-referencing test file:", ", ".join(gaps))
        return 1
    print("every src module is referenced by at least one test file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
