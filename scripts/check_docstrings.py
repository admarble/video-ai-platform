import ast
import sys
from pathlib import Path
from typing import List, Tuple

def check_docstring(node: ast.AST) -> Tuple[bool, str]:
    """Check if a node has a proper docstring.
    
    Args:
        node: AST node to check
        
    Returns:
        Tuple of (has_valid_docstring, message)
    """
    if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
        return True, ""
        
    docstring = ast.get_docstring(node)
    if not docstring:
        return False, f"Missing docstring for {node.__class__.__name__}"
    
    return True, ""

def analyze_file(file_path: str) -> List[str]:
    """Analyze a Python file for docstring issues.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        List of docstring issues found
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    tree = ast.parse(content)
    issues = []
    
    for node in ast.walk(tree):
        valid, message = check_docstring(node)
        if not valid:
            issues.append(f"{file_path}: {message}")
    
    return issues

def main():
    """Check docstrings in all Python files in the project."""
    root_dir = Path(__file__).parent.parent
    issues = []
    
    for py_file in root_dir.rglob("*.py"):
        if not any(part.startswith(".") for part in py_file.parts):
            issues.extend(analyze_file(str(py_file)))
    
    if issues:
        print("\n".join(issues))
        sys.exit(1)
    
    print("All docstrings look good!")
    sys.exit(0)

if __name__ == "__main__":
    main() 