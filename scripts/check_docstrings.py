import ast
import sys
from pathlib import Path

def check_docstrings(path: Path) -> bool:
    """Check if all classes and functions have docstrings."""
    has_all_docstrings = True
    
    def check_node(node, filename):
        nonlocal has_all_docstrings
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            if not ast.get_docstring(node):
                print(f"Missing docstring in {node.name} ({filename}:{node.lineno})")
                has_all_docstrings = False
    
    for python_file in path.rglob("*.py"):
        if python_file.name == "check_docstrings.py":
            continue
            
        try:
            with open(python_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                check_node(node, python_file)
                
        except SyntaxError as e:
            print(f"Syntax error in {python_file}: {e}")
            has_all_docstrings = False
            
    return has_all_docstrings

if __name__ == "__main__":
    src_path = Path("src")
    if not check_docstrings(src_path):
        sys.exit(1) 