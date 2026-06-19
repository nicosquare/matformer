from pathlib import Path


def test_repository_assets_remain_outside_src_layout():
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"

    boundary_paths = (
        "notebooks",
        "logs",
        "outputs",
        "paper",
        "references",
    )

    for relative_path in boundary_paths:
        path = (repo_root / relative_path).resolve()
        assert path.exists(), f"missing expected repository asset: {path}"
        assert path.is_relative_to(repo_root.resolve())
        assert not path.is_relative_to(src_root.resolve())
        assert not (src_root / relative_path).exists()
