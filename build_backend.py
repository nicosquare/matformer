from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import zipfile


NAME = "matformer"
VERSION = "0.0.0"
WHEEL_TAG = "py3-none-any"
DIST_INFO = f"{NAME}-{VERSION}.dist-info"
REPOSITORY_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPOSITORY_ROOT / "src"


def get_requires_for_build_wheel(config_settings=None):  # noqa: D401
    return []


def get_requires_for_build_editable(config_settings=None):  # noqa: D401
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _write_metadata_tree(Path(metadata_directory))


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return _write_metadata_tree(Path(metadata_directory))


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _build_wheel(Path(wheel_directory), editable=False)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _build_wheel(Path(wheel_directory), editable=True)


def _write_metadata_tree(metadata_root: Path) -> str:
    dist_info = metadata_root / DIST_INFO
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_metadata_text(), encoding="utf-8")
    (dist_info / "WHEEL").write_text(_wheel_text(), encoding="utf-8")
    (dist_info / "top_level.txt").write_text("models\ntraining\nevaluation\nutils\n", encoding="utf-8")
    return DIST_INFO


def _build_wheel(wheel_directory: Path, editable: bool) -> str:
    wheel_directory.mkdir(parents=True, exist_ok=True)
    wheel_name = f"{NAME}-{VERSION}-{WHEEL_TAG}.whl"
    wheel_path = wheel_directory / wheel_name
    records: list[tuple[str, str, str]] = []

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _add_bytes(zf, f"{DIST_INFO}/METADATA", _metadata_text().encode("utf-8"), records)
        _add_bytes(zf, f"{DIST_INFO}/WHEEL", _wheel_text().encode("utf-8"), records)
        _add_bytes(
            zf,
            f"{DIST_INFO}/top_level.txt",
            "models\ntraining\nevaluation\nutils\n".encode("utf-8"),
            records,
        )
        if editable:
            _add_bytes(zf, f"{NAME}.pth", f"{SRC_ROOT}\n".encode("utf-8"), records)
        record_lines = [",".join(row) for row in records]
        record_lines.append(f"{DIST_INFO}/RECORD,,")
        zf.writestr(f"{DIST_INFO}/RECORD", "\n".join(record_lines).encode("utf-8"))

    return wheel_name


def _metadata_text() -> str:
    return (
        "Metadata-Version: 2.1\n"
        f"Name: {NAME}\n"
        f"Version: {VERSION}\n"
        "Summary: MatFormer reproduction experiments\n"
    )


def _wheel_text() -> str:
    return (
        "Wheel-Version: 1.0\n"
        "Generator: build_backend\n"
        "Root-Is-Purelib: true\n"
        f"Tag: {WHEEL_TAG}\n"
    )


def _add_bytes(
    zf: zipfile.ZipFile,
    archive_name: str,
    data: bytes,
    records: list[tuple[str, str, str]],
) -> None:
    zf.writestr(archive_name, data)
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    records.append((archive_name, f"sha256={encoded}", str(len(data))))
