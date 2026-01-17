import ast
import re
from pathlib import Path

CORE = Path(__file__).parents[1] / "src" / "d3x" / "_core"
STUBS = CORE / "__init__.pyi"
CONSTS = CORE / "constants.pyi"


def parse_stub(path: Path) -> ast.Module:
    """Parse a Python stub file and return its AST."""
    try:
        return ast.parse(path.read_text())
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Stub file not found: {path}") from e
    except SyntaxError as e:
        raise SyntaxError(f"Invalid syntax in stub file: {e}") from e


def extract_all(tree: ast.Module) -> list:
    """Extract the `__all__` list from the AST."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        return [
                            elt.value if isinstance(elt, ast.Constant) else elt.id  # type: ignore
                            for elt in node.value.elts
                        ]
    return []


def get_assignment_info(node):
    """Helper to extract name and value from Assign or AnnAssign nodes."""
    if isinstance(node, ast.Assign):
        # Handle target list (e.g., x = y = 1)
        if isinstance(node.targets[0], ast.Name):
            return node.targets[0].id, node.value
    elif isinstance(node, ast.AnnAssign):
        # Handle annotated assignment (e.g., x: float = 1)
        if isinstance(node.target, ast.Name):
            return node.target.id, node.value
    return None, None


def clean_sig(sig: str) -> str:
    """Removes 'self' and generic 'arg0' names for cleaner docs."""
    sig = sig.replace("(self)", "()").replace("(self, ", "(")
    # Replace arg0, arg1 with generic placeholders or just empty if preferred
    sig = re.sub(r"arg\d+", "value", sig)
    return sig


def generate_api() -> str:
    """Generate professional API documentation."""
    core = parse_stub(STUBS)
    exports = extract_all(core)
    lines: list[str] = ["## API Reference\n"]

    # ---------- Imports ----------
    lines += ["### Quick Start", "```python", "from d3x import ("]
    for name in exports:
        lines.append(f"    {name},")
    lines += [")", "```", ""]

    # ---------- Processing ----------
    for node in core.body:
        # 1. HANDLE CLASSES
        if isinstance(node, ast.ClassDef):
            lines.append(f"## Class `{node.name}`")
            doc = ast.get_docstring(node)
            if doc:
                lines.append(f"*{doc}*\n")

            lines.append("| Member | Type | Description |")
            lines.append("|:-------|:-----|:------------|")

            seen_members = set()
            for item in node.body:
                name, m_type, m_doc = None, "Property", ""

                if isinstance(item, ast.FunctionDef):
                    if item.name.startswith("_") or item.name in seen_members:
                        continue
                    name = item.name
                    m_doc = ast.get_docstring(item) or "-"

                    is_prop = any(
                        isinstance(d, ast.Name) and d.id == "property" for d in item.decorator_list
                    )
                    if not is_prop:
                        args = [a.arg for a in item.args.args if a.arg != "self"]
                        name = f"{name}({', '.join(args)})"
                        m_type = "Method"

                    seen_members.add(item.name)

                elif isinstance(item, ast.Assign | ast.AnnAssign):
                    name, _ = get_assignment_info(item)
                    if name and not name.startswith("_") and name not in seen_members:
                        m_type = "Attribute"
                        seen_members.add(name)

                if name:
                    lines.append(f"| `{name}` | {m_type} | {m_doc} |")
            lines.append("")

        # 2. HANDLE STANDALONE FUNCTIONS
        elif isinstance(node, ast.FunctionDef):
            if node.name in exports:
                args = [a.arg for a in node.args.args]
                sig = f"({', '.join(args)})"
                doc = ast.get_docstring(node) or "No description available."
                lines.append(f"### `fn {node.name}{sig}`")
                lines.append(f"{doc}\n")

    return "\n".join(lines)


def generate_constants() -> str:
    """Generate a table of constants with their values and units."""
    consts_tree = parse_stub(CONSTS)
    const_exports = extract_all(consts_tree)

    data = []

    # Iterate through body to map units and collect values
    for idx, node in enumerate(consts_tree.body):
        name, value_node = get_assignment_info(node)

        if name and name in const_exports:
            # 1. Extract Unit from the following node (docstring)
            unit = ""
            if idx + 1 < len(consts_tree.body):
                next_node = consts_tree.body[idx + 1]
                if isinstance(next_node, ast.Expr) and isinstance(next_node.value, ast.Constant):
                    doc = str(next_node.value.value)
                    if "@unit" in doc:
                        unit = doc.replace("@unit", "").strip()

            # 2. Extract Value
            val = value_node.value if isinstance(value_node, ast.Constant) else "Unknown"
            data.append((name, val, unit))

    # Build Markdown Table
    lines = ["## Constants\n", "| Name | Value | Unit |", "|------|-------|------|"]
    for name, val, unit in data:
        lines.append(f"| `{name}` | {val} | {unit} |")

    return "\n".join(lines)


if __name__ == "__main__":
    # --- Check Imports ---
    print("--- CORE STUBS ---\n")
    print(generate_api())

    # --- Check Constants ---
    print("\n--- CONSTANTS ---\n")
    # Reuse the logic to ensure consistency
    print(generate_constants())

    print("\n--- PATHS---\n")
    for p in [CORE, STUBS, CONSTS]:
        print(f"{p.name:<15} : {p.as_posix()}")
