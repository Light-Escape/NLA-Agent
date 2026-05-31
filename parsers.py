"""
矩阵输入解析模块：负责 CSC / Matrix Market 文件读取与格式转换。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import mmread
from scipy.sparse import csc_matrix, issparse

try:
    from .upload_store import UPLOAD_URI_PREFIX, resolve_uploaded_matrix_file
except ImportError:
    from upload_store import UPLOAD_URI_PREFIX, resolve_uploaded_matrix_file

MAX_MATRIX_ELEMENTS = 10_000
_MODULE_DIR = Path(__file__).resolve().parent
_MATRIX_PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?[^`\"'\s，。；：,;]*?\.(?:mtx\.gz|txt|mtx|csv|npz))",
    re.IGNORECASE,
)


def normalize_file_path(file_path: str) -> str:
    """清理用户输入路径，兼容引号/反引号包裹、自然语言后缀与 Windows 反斜杠。"""
    fp = (file_path or "").strip()
    if (len(fp) >= 2) and ((fp[0] == fp[-1] == '"') or (fp[0] == fp[-1] == "'")):
        fp = fp[1:-1].strip()
    if fp.startswith("`") and fp.endswith("`") and len(fp) >= 2:
        fp = fp[1:-1].strip()
    match = _MATRIX_PATH_RE.search(fp)
    if match:
        fp = match.group("path")
    return fp.replace("\\", "/")


def _matrix_file_candidates(file_path: str) -> list[Path]:
    """解析矩阵文件路径；相对路径同时兼容启动目录与 NLA_Master 目录。"""
    uploaded = resolve_uploaded_matrix_file(file_path)
    if uploaded is not None:
        return [uploaded]

    fp = normalize_file_path(file_path)
    path = Path(fp).expanduser()
    if path.is_absolute():
        return [path]

    candidates: list[Path] = [
        Path.cwd() / path,
        Path.cwd() / "NLA_Master" / path,
        _MODULE_DIR / path,
        _MODULE_DIR.parent / path,
    ]

    # 如果模型把短文件名嵌在自然语言里，或传入了多余目录，继续用 basename 在常见目录兜底。
    if path.name and path.name != str(path):
        candidates.extend(
            [
                Path.cwd() / path.name,
                Path.cwd() / "NLA_Master" / path.name,
                _MODULE_DIR / path.name,
                _MODULE_DIR.parent / path.name,
            ]
        )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _is_browser_path(file_path: str) -> bool:
    fp = (file_path or "").strip().strip("`\"'")
    return fp.startswith("Current Folder/") or fp.startswith("Current Folder\\")


def _is_probably_bare_filename(file_path: str) -> bool:
    fp = normalize_file_path(file_path)
    path = Path(fp)
    return path.name == fp and bool(path.suffix)


def _file_resolution_error(file_path: str, checked_paths: list[Path]) -> dict:
    raw = (file_path or "").strip()
    normalized = normalize_file_path(raw)
    if raw.strip("`\"'").startswith(UPLOAD_URI_PREFIX):
        message = "上传 URI 未解析到后端文件；可能是 file_id 已失效、上传服务使用了不同的 NLA_UPLOAD_DIR，或文件尚未成功上传。"
        next_actions = [
            "检查本轮 <frontend-files> 中的 uploadUri 是否完整",
            "重新上传矩阵文件并使用新的 nla-upload://<file_id>",
            "确认 ADK 后端与上传服务使用相同的 NLA_UPLOAD_DIR",
        ]
        detail = {"reference_type": "upload_uri", "normalized_path": normalized}
    elif _is_browser_path(raw):
        message = "浏览器 Current Folder 路径不是后端可读取路径；矩阵文件必须先上传并使用 nla-upload://<file_id>。"
        next_actions = [
            "从 <frontend-files> 读取 uploadUri / file_id",
            "若没有 uploadUri，请让用户重新上传文件或启动上传服务",
            "不要把 Current Folder 路径传给矩阵读取工具",
        ]
        detail = {"reference_type": "browser_path", "normalized_path": normalized}
    elif _is_probably_bare_filename(raw):
        message = f"未找到裸文件名 {raw}；浏览器文件不会自动出现在后端工作目录。"
        next_actions = [
            "优先使用 <frontend-files> 中的 nla-upload://<file_id>",
            "若用户只提供了文件名，请要求重新上传或提供后端真实可访问路径",
            "不要猜测 uploads 目录、临时目录或标准测试矩阵内容",
        ]
        detail = {"reference_type": "bare_filename", "normalized_path": normalized}
    else:
        message = f"文件不存在: {raw}"
        next_actions = [
            "确认路径是后端进程可访问的真实路径",
            "或改用前端上传得到的 nla-upload://<file_id>",
        ]
        detail = {
            "reference_type": "backend_path",
            "normalized_path": normalized,
            "checked_paths": [str(path) for path in checked_paths],
        }
    return {
        "status": "error",
        "error_type": "file_resolution_error",
        "message": message,
        "fallback_required": True,
        "next_allowed_actions": next_actions,
        "detail": detail,
    }


def resolve_matrix_file_path(file_path: str) -> Path:
    candidates = _matrix_file_candidates(file_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _csc_to_payload(matrix: csc_matrix) -> dict:
    matrix = matrix.tocsc()
    m, n = map(int, matrix.shape)
    nnz = int(matrix.nnz)
    payload: dict = {"format": "csc", "shape": [m, n], "nnz": nnz}
    if nnz <= MAX_MATRIX_ELEMENTS and (n + 1) <= MAX_MATRIX_ELEMENTS:
        payload.update(
            {
                "indptr": matrix.indptr.tolist(),
                "indices": matrix.indices.tolist(),
                "data": matrix.data.astype(float, copy=False).tolist(),
            }
        )
    return payload


def csc_payload_to_matrix(payload: dict) -> csc_matrix:
    """将可序列化的 CSC payload 还原为 scipy.sparse.csc_matrix。"""
    if not isinstance(payload, dict):
        raise ValueError("A_csc 必须是 dict")
    if payload.get("format") != "csc":
        raise ValueError("A_csc.format 必须是 'csc'")
    shape = payload.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(isinstance(x, int) and x >= 0 for x in shape)
    ):
        raise ValueError("A_csc.shape 必须是 [m, n]")
    for key in ("indptr", "indices", "data"):
        if key not in payload:
            raise ValueError(f"A_csc 缺少字段: {key}")
    return csc_matrix(
        (
            np.asarray(payload["data"], dtype=float),
            np.asarray(payload["indices"], dtype=np.int64),
            np.asarray(payload["indptr"], dtype=np.int64),
        ),
        shape=(int(shape[0]), int(shape[1])),
    )


def parse_csc_content(content: str) -> dict:
    """解析 4 行文本 CSC 格式。"""
    lines = [line.strip() for line in (content or "").strip().splitlines()]
    if len(lines) < 4:
        return {
            "status": "error",
            "message": "CSC 文件至少需要 4 行：m n、indptr、indices、data",
        }

    header = np.fromstring(lines[0], sep=" ", dtype=np.int64)
    if header.size != 2:
        return {"status": "error", "message": "第 1 行必须是两个整数：m n"}
    m, n = map(int, header.tolist())
    if m <= 0 or n <= 0:
        return {"status": "error", "message": f"非法矩阵形状: {m}x{n}"}

    indptr = np.fromstring(lines[1], sep=" ", dtype=np.int64)
    indices = np.fromstring(lines[2], sep=" ", dtype=np.int64)
    data = np.fromstring(lines[3], sep=" ", dtype=float)

    if indptr.size != (n + 1):
        return {
            "status": "error",
            "message": f"indptr 长度应为 n+1={n + 1}，当前为 {indptr.size}",
        }
    if indices.size != data.size:
        return {"status": "error", "message": "indices 与 data 长度必须相同"}

    try:
        matrix = csc_matrix((data, indices, indptr), shape=(m, n))
    except Exception as exc:
        return {"status": "error", "message": f"构建 CSC 矩阵失败: {exc}"}

    if m * n > MAX_MATRIX_ELEMENTS:
        return {
            "status": "ok",
            "shape": [m, n],
            "A_csc": _csc_to_payload(matrix),
            "_matrix": matrix,
            "message": (
                f"矩阵规模 {m}x{n} 超过上限 {MAX_MATRIX_ELEMENTS}，"
                "已返回稀疏 CSC 表示。"
            ),
        }

    dense = matrix.toarray()
    return {
        "status": "ok",
        "A_rows": dense.tolist(),
        "shape": [m, n],
        "A_csc": _csc_to_payload(matrix),
        "_matrix": matrix,
        "message": "已解析 CSC，返回 A_rows 与 A_csc。",
    }


def load_matrix_csc_file(file_path: str) -> dict:
    try:
        fp = resolve_matrix_file_path(file_path)
        content = fp.read_text(encoding="utf-8")
        result = parse_csc_content(content)
        if isinstance(result, dict) and result.get("status") == "ok":
            result["resolved_path"] = str(fp)
        return result
    except FileNotFoundError:
        return _file_resolution_error(file_path, _matrix_file_candidates(file_path))
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def load_matrix_csc_content(content: str) -> dict:
    try:
        return parse_csc_content(content)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def load_matrix_mtx_gz(file_path: str) -> dict:
    try:
        fp = resolve_matrix_file_path(file_path)
        matrix = mmread(fp)
        if issparse(matrix):
            csc = matrix.tocsc().astype(float, copy=False)
            m, n = map(int, csc.shape)
            return {
                "status": "ok",
                "shape": [m, n],
                "A_csc": _csc_to_payload(csc),
                "_matrix": csc,
                "resolved_path": str(fp),
                "message": f"已读取 Matrix Market 稀疏矩阵 ({m}x{n}, nnz={csc.nnz})",
            }

        dense = np.asarray(matrix, dtype=float)
        m, n = map(int, dense.shape)
        if m * n > MAX_MATRIX_ELEMENTS:
            return {
                "status": "error",
                "message": (
                    f"矩阵过大（{m}x{n}={m * n}），超过上限 {MAX_MATRIX_ELEMENTS}。"
                ),
            }
        return {
            "status": "ok",
            "A_rows": dense.tolist(),
            "shape": [m, n],
            "_matrix": dense,
            "resolved_path": str(fp),
            "message": "已读取 Matrix Market 稠密矩阵并转换为 A_rows。",
        }
    except FileNotFoundError:
        return _file_resolution_error(file_path, _matrix_file_candidates(file_path))
    except Exception as exc:
        msg = str(exc)
        if "Unterminated string" in msg or "JSONDecodeError" in msg:
            msg += "（Windows 路径建议使用 /，或将 \\ 写成 \\\\）"
        return {"status": "error", "message": msg}


def matrix_to_dense(
    A_rows: Optional[list[list[float]]] = None,
    A_csc: Optional[dict] = None,
) -> np.ndarray:
    """内部工具：统一把输入矩阵转成 dense ndarray。"""
    if A_rows is not None:
        arr = np.asarray(A_rows, dtype=float)
        if arr.ndim != 2:
            raise ValueError("A_rows 必须是二维列表")
        return arr
    if A_csc is not None:
        return csc_payload_to_matrix(A_csc).toarray()
    raise ValueError("A_rows 与 A_csc 至少提供一个")
