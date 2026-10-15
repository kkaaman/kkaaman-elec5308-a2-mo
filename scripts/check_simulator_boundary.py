"""RV-05: inspect core imports and smoke-import it with simulator imports blocked."""

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DEMOS = frozenset({"demo_combined_perception", "demo_fused_lane_following"})
FORBIDDEN_ROOTS = frozenset({"carla", "carla_adapter"}) | LEGACY_DEMOS


def source_modules(source_dir):
    """Discover every source file, including compatibility/namespace modules."""
    modules = {}
    for path in sorted(source_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(source_dir).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if parts:
            modules[".".join(parts)] = path
    return modules


def is_forbidden(module):
    parts = module.split(".")
    # Running from the clone root can also expose src as a namespace package.
    if parts[0] == "src" and len(parts) > 1:
        parts = parts[1:]
    return parts[0] in FORBIDDEN_ROOTS


def literal_string(node):
    """Resolve string literals/concatenations, never execute source code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = literal_string(node.left), literal_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def qualified_name(node, bindings):
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = qualified_name(node.value, bindings)
        return f"{parent}.{node.attr}" if parent else None
    return None


def import_references(tree, module, is_package):
    """Yield (line, target) for imports, including statically named loaders.

    Unresolved loader calls fail closed. This is an architecture guard for
    ordinary Python imports, not an evaluator of arbitrary Python programs.
    """
    bindings = {}
    package = module if is_package else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            target = node.module or ""
            if node.level:
                try:
                    target = importlib.util.resolve_name("." * node.level + target, package)
                except (ImportError, ValueError):
                    yield node.lineno, "<invalid relative import>"
                    continue
            yield node.lineno, target
            for alias in node.names:
                full_name = f"{target}.{alias.name}" if target else alias.name
                bindings[alias.asname or alias.name] = full_name
                yield node.lineno, full_name

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        loader = qualified_name(node.func, bindings)
        if loader not in {"importlib.import_module", "__import__", "builtins.__import__"}:
            continue
        arguments = {keyword.arg: keyword.value for keyword in node.keywords}
        name_node = node.args[0] if node.args else arguments.get("name")
        target = literal_string(name_node)
        if target is None:
            yield node.lineno, "<unresolved dynamic import>"
            continue
        if loader == "importlib.import_module" and target.startswith("."):
            package_node = node.args[1] if len(node.args) > 1 else arguments.get("package")
            dynamic_package = (
                package if isinstance(package_node, ast.Name) and package_node.id == "__package__"
                else literal_string(package_node)
            )
            try:
                target = importlib.util.resolve_name(target, dynamic_package)
            except (ImportError, ValueError, AttributeError):
                yield node.lineno, "<unresolved relative dynamic import>"
                continue
        elif loader in {"__import__", "builtins.__import__"}:
            # Relative __import__ depends on globals/fromlist/level. Require the
            # ordinary import syntax instead of silently treating it as absolute.
            level = node.args[4] if len(node.args) > 4 else arguments.get("level")
            if level is not None and not (isinstance(level, ast.Constant) and level.value == 0):
                yield node.lineno, "<unresolved relative dynamic import>"
                continue
        yield node.lineno, target


def boundary_violations(source_dir):
    """Scan all core files, so importing an intermediate shim cannot hide a leak."""
    violations = []
    for module, path in source_modules(source_dir).items():
        if is_forbidden(module):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for line, target in import_references(tree, module, path.name == "__init__.py"):
            if is_forbidden(target) or target.startswith("<"):
                violations.append(f"{path.relative_to(source_dir)}:{line}: {module} -> {target}")
    return sorted(set(violations))


SMOKE_CODE = """
import importlib
import importlib.abc
import json
import sys

forbidden = set(json.loads(sys.argv[1]))
modules = json.loads(sys.argv[2])
attempts = []

class BoundaryImportError(BaseException):
    pass

class SimulatorBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        parts = fullname.split('.')
        root = parts[1] if parts[0] == 'src' and len(parts) > 1 else parts[0]
        if root in forbidden:
            attempts.append(fullname)
            # ImportError can be swallowed by optional-dependency guards.
            raise BoundaryImportError('Forbidden simulator import: ' + fullname)

assert not any(name.split('.', 1)[0] in forbidden for name in sys.modules)
sys.meta_path.insert(0, SimulatorBlocker())
for name in modules:
    importlib.import_module(name)
assert not attempts, 'Forbidden simulator import attempts: ' + ', '.join(attempts)
assert not any(name.split('.', 1)[0] in forbidden for name in sys.modules)
print('Imported', len(modules), 'core modules with simulator imports blocked')
"""


def smoke_imports(source_dir, *, modules=None, cwd=None):
    """Fresh interpreter; no fake SDK, server, model construction or inference."""
    if modules is None:
        modules = [name for name in source_modules(source_dir) if not is_forbidden(name)]
    return subprocess.run(
        [sys.executable, "-c", SMOKE_CODE, json.dumps(sorted(FORBIDDEN_ROOTS)), json.dumps(modules)],
        env=dict(os.environ, PYTHONPATH=str(source_dir.resolve())),
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="also import core in a fresh process")
    args = parser.parse_args()
    source_dir = REPO_ROOT / "src"
    violations = boundary_violations(source_dir)
    if violations:
        print("Simulator boundary violations:", *violations, sep="\n", file=sys.stderr)
        return 1
    print("AST simulator boundary passed")
    if args.smoke:
        result = smoke_imports(source_dir)
        print(result.stdout, end="")
        if result.returncode:
            print(result.stderr, end="", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
