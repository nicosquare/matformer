import subprocess
import sys
from pathlib import Path


def test_importable_packages_resolve_from_src():
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    code = f"""
import sys
from pathlib import Path

sys.path.insert(0, {str(src_root)!r})

import evaluation
import models
import training
import utils

src_root = Path({str(src_root)!r}).resolve()
for module in (models, training, evaluation, utils):
    module_path = Path(module.__file__).resolve()
    assert src_root in module_path.parents, module_path

print("ok")
"""
    subprocess.run([sys.executable, "-c", code], cwd=repo_root, check=True)
