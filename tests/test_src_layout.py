import subprocess
import sys
from pathlib import Path


def test_importable_packages_resolve_from_src():
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    package_names = ("models", "training", "evaluation", "utils")
    code = f"""
from importlib import import_module
import sys
from pathlib import Path

sys.path.insert(0, {str(src_root)!r})

modules = [import_module(name) for name in {package_names!r}]

src_root = Path({str(src_root)!r}).resolve()
for module in modules:
    module_path = Path(module.__file__).resolve()
    assert src_root in module_path.parents, module_path

print("ok")
"""
    subprocess.run([sys.executable, "-c", code], cwd=repo_root, check=True)
