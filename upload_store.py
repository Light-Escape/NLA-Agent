"""
Uploaded matrix file storage.

The browser cannot expose a user's real local path.  We therefore persist the
uploaded bytes on the backend and pass a stable file_id back to the agent.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Optional

UPLOAD_URI_PREFIX = "nla-upload://"

_MODULE_DIR = Path(__file__).resolve().parent
_SAFE_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def upload_root() -> Path:
    root = os.getenv("NLA_UPLOAD_DIR", "").strip()
    return Path(root).expanduser().resolve() if root else (_MODULE_DIR / "uploads").resolve()


def _safe_filename(filename: str) -> str:
    name = Path(filename or "matrix.dat").name.strip() or "matrix.dat"
    cleaned = _UNSAFE_FILENAME_RE.sub("_", name)
    return cleaned.strip("._") or "matrix.dat"


def _metadata_path(file_id: str) -> Path:
    return upload_root() / file_id / "metadata.json"


def uploaded_file_uri(file_id: str) -> str:
    return f"{UPLOAD_URI_PREFIX}{file_id}"


def save_uploaded_matrix_file(
    filename: str,
    content: bytes,
    content_type: Optional[str] = None,
) -> dict:
    if not content:
        raise ValueError("上传文件为空")

    file_id = uuid.uuid4().hex
    folder = upload_root() / file_id
    folder.mkdir(parents=True, exist_ok=False)

    safe_name = _safe_filename(filename)
    server_path = folder / safe_name
    server_path.write_bytes(content)

    metadata = {
        "file_id": file_id,
        "uri": uploaded_file_uri(file_id),
        "server_path": str(server_path),
        "original_name": filename,
        "stored_name": safe_name,
        "size": len(content),
        "content_type": content_type or "",
    }
    _metadata_path(file_id).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def _extract_file_id(reference: str) -> Optional[str]:
    ref = (reference or "").strip().strip("`\"'")
    if ref.startswith(UPLOAD_URI_PREFIX):
        ref = ref[len(UPLOAD_URI_PREFIX) :]
    if _SAFE_ID_RE.fullmatch(ref):
        return ref
    match = re.search(r"(?:file_id|fileId)\s*[:=]\s*([a-f0-9]{32})", ref)
    if match:
        return match.group(1)
    match = re.search(r"nla-upload://([a-f0-9]{32})", ref)
    if match:
        return match.group(1)
    return None


def resolve_uploaded_matrix_file(reference: str) -> Optional[Path]:
    file_id = _extract_file_id(reference)
    if not file_id:
        return None

    metadata_file = _metadata_path(file_id)
    if metadata_file.exists():
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            server_path = Path(str(metadata.get("server_path", "")))
            if server_path.exists():
                return server_path
        except Exception:
            pass

    folder = upload_root() / file_id
    if not folder.exists():
        return None
    candidates = [path for path in folder.iterdir() if path.is_file() and path.name != "metadata.json"]
    return candidates[0] if candidates else None
