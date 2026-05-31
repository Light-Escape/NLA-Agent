"""
MATLAB 风格工作区：跨轮变量持久化、对象句柄、受控读取与审计日志。
"""

from __future__ import annotations

import hashlib
import reprlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix, issparse

MAX_CONTEXT_ELEMENTS = 64
MAX_DENSE_CONVERSION_ELEMENTS = 1_000_000
MAX_SLICE_ELEMENTS = 100
MAX_SLICE_CALLS_PER_VAR = 8
MAX_CUMULATIVE_SLICE_ELEMENTS = 256
MAX_SLICE_COVERAGE_RATIO = 0.25
MAX_AUDIT_RECORDS = 200
MAX_WORKSPACE_SET_ELEMENTS = 64


def _infer_shape(value: Any) -> list[int]:
    """尽量推断常见容器的 shape（列表/元组/NumPy 数组）。"""
    if isinstance(value, dict):
        shape = value.get("shape")
        if isinstance(shape, (list, tuple)) and all(isinstance(x, int) for x in shape):
            return [int(x) for x in shape]
        for key in ("A_rows", "x", "value", "data", "singular_values", "eigenvalues"):
            if key in value:
                return _infer_shape(value[key])

    if hasattr(value, "shape"):
        try:
            shape = getattr(value, "shape")
            return [int(x) for x in shape]
        except Exception:
            pass

    if isinstance(value, (list, tuple)):
        if not value:
            return [0]
        sub_shape = _infer_shape(value[0])
        return [len(value)] + sub_shape
    return []


def _element_count(shape: list[int]) -> int:
    if not shape:
        return 1
    total = 1
    for dim in shape:
        total *= max(int(dim), 0)
    return int(total)


def _matrix_format(value: Any) -> str | None:
    if isinstance(value, dict):
        if value.get("matrix_handle") is True:
            return str(value.get("format") or value.get("storage_type") or "matrix")
        if value.get("format") == "csc":
            return "csc"
        if "A_csc" in value:
            return "csc"
        if "A_rows" in value:
            return "dense"
    if issparse(value):
        return "csc"
    if hasattr(value, "ndim") and hasattr(value, "shape"):
        try:
            return "dense" if int(getattr(value, "ndim")) == 2 else None
        except Exception:
            return None
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return "dense"
    return None


def _is_matrix_like(value: Any) -> bool:
    shape = _infer_shape(value)
    return len(shape) == 2 and _matrix_format(value) is not None


def _is_large_value(value: Any) -> bool:
    if _is_matrix_like(value):
        return True
    shape = _infer_shape(value)
    return _element_count(shape) > MAX_CONTEXT_ELEMENTS


def _first_scalar(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("A_rows", "x", "value", "data", "singular_values", "eigenvalues"):
            if key in value:
                found = _first_scalar(value[key])
                if found is not None:
                    return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_scalar(item)
            if found is not None:
                return found
        return None
    if hasattr(value, "flat"):
        try:
            iterator = value.flat
            return next(iter(iterator), None)
        except Exception:
            return None
    return value


def _infer_dtype(value: Any) -> str:
    if isinstance(value, dict):
        if isinstance(value.get("dtype"), str):
            return str(value["dtype"])
        if value.get("format") == "csc" or "A_csc" in value:
            return "float64"
    if issparse(value):
        return str(value.dtype)
    if hasattr(value, "dtype"):
        try:
            return str(value.dtype)
        except Exception:
            pass
    scalar = _first_scalar(value)
    if scalar is None:
        return "unknown"
    if isinstance(scalar, bool):
        return "logical"
    if isinstance(scalar, int):
        return "int"
    if isinstance(scalar, float):
        return "double"
    if isinstance(scalar, complex):
        return "complex"
    if isinstance(scalar, str):
        return "string"
    return type(scalar).__name__


def _storage_type(value: Any) -> str:
    if _is_matrix_like(value):
        fmt = _matrix_format(value)
        return "sparse" if fmt == "csc" else "dense"
    if issparse(value):
        return "sparse"
    if isinstance(value, dict) and any(key in value for key in ("u", "v", "vh", "q", "r", "lu")):
        return "summary_only" if _is_large_value(value) else "dense"
    shape = _infer_shape(value)
    if len(shape) == 0:
        return "scalar"
    if len(shape) == 1:
        return "dense"
    return "summary_only" if _is_large_value(value) else "dense"


def _kind(value: Any) -> str:
    if _is_matrix_like(value):
        return "sparse_matrix" if _storage_type(value) == "sparse" else "matrix"
    shape = _infer_shape(value)
    if isinstance(value, dict):
        if any(key in value for key in ("u", "v", "vh", "q", "r", "lu", "singular_values")):
            return "factorization"
        if "x" in value or "residual_norm_2" in value:
            return "result"
        if "status" in value:
            return "diagnostic"
    if len(shape) == 1 and shape:
        return "vector"
    if not shape:
        return "scalar"
    return "result"


def _class_name(value: Any, shape: list[int]) -> str:
    kind = _kind(value)
    if kind == "sparse_matrix":
        return "sparse matrix"
    if kind == "matrix":
        return "matrix"
    if kind == "vector":
        return "vector"
    if kind == "factorization":
        return "factorization"
    if kind == "result":
        return "result struct"
    if isinstance(value, dict) and value.get("matrix_handle") is True:
        return "matrix handle"
    if len(shape) == 2:
        return "matrix"
    return type(value).__name__


def _format_shape(shape: list[int]) -> str:
    if not shape:
        return "1x1"
    if len(shape) == 1:
        return f"{shape[0]}x1"
    return "x".join(str(x) for x in shape)


def _matrix_stats(value: Any) -> dict:
    shape = _infer_shape(value)
    fmt = _matrix_format(value) or "matrix"
    dtype = _infer_dtype(value)
    stats: dict[str, Any] = {
        "matrix_handle": True,
        "shape": shape,
        "format": fmt,
        "dtype": dtype,
    }
    m, n = (shape + [0, 0])[:2]
    total = int(m) * int(n)

    try:
        if isinstance(value, dict) and "A_csc" in value:
            csc_payload = value["A_csc"]
            stats["nnz"] = int(csc_payload.get("nnz", len(csc_payload.get("data", []))))
            stats["format"] = "csc"
        elif isinstance(value, dict) and value.get("format") == "csc":
            stats["nnz"] = int(value.get("nnz", len(value.get("data", []))))
            stats["format"] = "csc"
        elif issparse(value):
            stats["nnz"] = int(value.nnz)
            stats["format"] = "csc"
        elif isinstance(value, dict) and "A_rows" in value:
            stats["nnz"] = int(np.count_nonzero(np.asarray(value["A_rows"], dtype=float)))
        else:
            stats["nnz"] = int(np.count_nonzero(np.asarray(value, dtype=float)))
    except Exception:
        pass

    if total and isinstance(stats.get("nnz"), int):
        stats["density"] = float(stats["nnz"] / total)
    return stats


def _fingerprint(value: Any) -> str:
    shape = _infer_shape(value)
    payload = f"{_kind(value)}|{shape}|{_infer_dtype(value)}|{_preview(value, max_len=160)}"
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:16]


def _summary(value: Any) -> str:
    shape = _infer_shape(value)
    kind = _kind(value)
    storage = _storage_type(value)
    parts = [kind, f"shape={shape or []}", f"dtype={_infer_dtype(value)}", f"storage={storage}"]
    if _is_matrix_like(value):
        stats = _matrix_stats(value)
        if "nnz" in stats:
            parts.append(f"nnz={stats['nnz']}")
        if "density" in stats:
            parts.append(f"density={stats['density']:.6g}")
    return ", ".join(parts)


def _preview(value: Any, max_len: int = 120) -> str:
    if _is_large_value(value):
        return f"ObjectHandle({_summary(value)})"
    raw = reprlib.repr(value)
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 3] + "..."


def _object_handle(name: str, value: Any, meta: dict[str, Any] | None = None) -> dict:
    meta = meta or {}
    handle = {
        "object_handle": True,
        "ref": name,
        "name": name,
        "display_name": meta.get("display_name", name),
        "role": meta.get("role", ""),
        "origin": meta.get("origin", meta.get("source", "Agent")),
        "parent_refs": list(meta.get("parent_refs", [])),
        "alias_of": meta.get("alias_of", ""),
        "created_by_tool": meta.get("created_by_tool", ""),
        "kind": _kind(value),
        "shape": _infer_shape(value),
        "dtype": _infer_dtype(value),
        "storage_type": _storage_type(value),
        "version": int(meta.get("version", 1)),
        "fingerprint": str(meta.get("fingerprint") or _fingerprint(value)),
        "summary": str(meta.get("summary") or _summary(value)),
        "preview_policy": "handle_only" if _is_large_value(value) else "safe",
        "message": "对象内容保存在后端 Workspace；请用 *_ref 参数或受控读取接口引用，不要展开完整数据。",
    }
    if _is_matrix_like(value):
        handle.update(_matrix_stats(value))
        handle["matrix_handle"] = True
    return handle


def _value_for_context(name: str, value: Any, meta: dict[str, Any] | None = None) -> Any:
    if _is_large_value(value):
        return _object_handle(name, value, meta=meta)
    return value


def _to_dense_array(value: Any) -> np.ndarray:
    if isinstance(value, dict) and "A_rows" in value:
        return np.asarray(value["A_rows"], dtype=float)
    if isinstance(value, dict) and "A_csc" in value:
        value = value["A_csc"]
    if isinstance(value, dict) and value.get("format") == "csc":
        for key in ("indptr", "indices", "data"):
            if key not in value:
                raise ValueError(f"CSC 数据缺少字段: {key}")
        shape = value.get("shape")
        matrix = csc_matrix(
            (
                np.asarray(value["data"], dtype=float),
                np.asarray(value["indices"], dtype=np.int64),
                np.asarray(value["indptr"], dtype=np.int64),
            ),
            shape=(int(shape[0]), int(shape[1])),
        )
        return matrix.toarray()
    if issparse(value):
        return value.toarray()
    return np.asarray(value, dtype=float)


def _to_sparse_matrix(value: Any) -> csc_matrix:
    if issparse(value):
        return value.tocsc()
    if isinstance(value, dict) and "A_csc" in value:
        value = value["A_csc"]
    if isinstance(value, dict) and value.get("format") == "csc":
        for key in ("indptr", "indices", "data"):
            if key not in value:
                raise ValueError(f"CSC 数据缺少字段: {key}")
        shape = value.get("shape")
        return csc_matrix(
            (
                np.asarray(value["data"], dtype=float),
                np.asarray(value["indices"], dtype=np.int64),
                np.asarray(value["indptr"], dtype=np.int64),
            ),
            shape=(int(shape[0]), int(shape[1])),
        )
    arr = np.asarray(value["A_rows"] if isinstance(value, dict) and "A_rows" in value else value, dtype=float)
    if arr.ndim != 2:
        raise ValueError("对象不是二维矩阵")
    return csc_matrix(arr)


@dataclass
class WorkspaceStore:
    _vars: dict[str, Any] = field(default_factory=dict)
    _meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    _versions: dict[str, int] = field(default_factory=dict)
    _slice_budget: dict[str, dict[str, Any]] = field(default_factory=dict)
    _audit: list[dict[str, Any]] = field(default_factory=list)

    def _now(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _log(self, action: str, name: str | None = None, detail: dict[str, Any] | None = None) -> None:
        record = {
            "time": self._now(),
            "action": action,
            "name": name or "",
            "version": self._versions.get(name or "", 0),
            "detail": detail or {},
        }
        self._audit.append(record)
        if len(self._audit) > MAX_AUDIT_RECORDS:
            self._audit = self._audit[-MAX_AUDIT_RECORDS:]

    def _protocol_error(self, name: str, message: str, detail: dict[str, Any] | None = None) -> dict:
        payload = {
            "status": "error",
            "error_type": "workspace_protocol_violation",
            "message": message,
            "fallback_required": True,
            "next_allowed_actions": [
                "workspace_summary",
                "workspace_stats",
                "workspace_structure",
                "使用 *_ref 或 array_refs 调用后端计算工具",
                "使用 workspace_alias / workspace_bind_role 管理已有对象的名称和角色",
            ],
        }
        if detail:
            payload["detail"] = detail
        self._log("protocol_violation", name, {"reason": message, **(detail or {})})
        return payload

    def _set_meta(
        self,
        name: str,
        value: Any,
        source: str,
        notes: str,
        *,
        role: str = "",
        origin: str = "",
        parent_refs: list[str] | None = None,
        display_name: str = "",
        created_by_tool: str = "",
        alias_of: str = "",
    ) -> None:
        version = self._versions.get(name, 0) + 1
        self._versions[name] = version
        self._meta[name] = {
            "source": (source or "Agent").strip() or "Agent",
            "updatedAt": self._now(),
            "notes": (notes or "").strip(),
            "version": version,
            "fingerprint": _fingerprint(value),
            "summary": _summary(value),
            "role": (role or "").strip(),
            "origin": (origin or source or "Agent").strip(),
            "parent_refs": [str(ref) for ref in (parent_refs or [])],
            "display_name": (display_name or name).strip(),
            "created_by_tool": (created_by_tool or "").strip(),
            "alias_of": (alias_of or "").strip(),
        }

    def _variable_detail(self, name: str) -> dict:
        value = self._vars[name]
        meta = self._meta.get(name, {})
        shape = _infer_shape(value)
        cls = _class_name(value, shape)
        detail = {
            "name": name,
            "ref": name,
            "display_name": meta.get("display_name", name),
            "role": meta.get("role", ""),
            "origin": meta.get("origin", meta.get("source", "Agent")),
            "parent_refs": list(meta.get("parent_refs", [])),
            "alias_of": meta.get("alias_of", ""),
            "created_by_tool": meta.get("created_by_tool", ""),
            "type": type(value).__name__,
            "kind": _kind(value),
            "className": cls,
            "shape": shape,
            "size": _format_shape(shape),
            "dtype": _infer_dtype(value),
            "storage_type": _storage_type(value),
            "isSparse": cls == "sparse matrix",
            "isComplex": _infer_dtype(value) == "complex",
            "source": meta.get("source", "Agent"),
            "updatedAt": meta.get("updatedAt", ""),
            "notes": meta.get("notes", ""),
            "version": int(meta.get("version", self._versions.get(name, 1))),
            "fingerprint": meta.get("fingerprint", _fingerprint(value)),
            "summary": meta.get("summary", _summary(value)),
            "preview_policy": "handle_only" if _is_large_value(value) else "safe",
            "preview": _preview(value),
        }
        if _is_matrix_like(value):
            stats = _matrix_stats(value)
            if "nnz" in stats:
                detail["nnz"] = stats["nnz"]
            if "density" in stats:
                detail["density"] = stats["density"]
        return detail

    def snapshot(self) -> dict:
        variables = [self._variable_detail(name) for name in sorted(self._vars.keys())]
        return {"status": "ok", "count": len(variables), "variables": variables}

    def allocate_ref(self, prefix: str = "A") -> str:
        base = (prefix or "A").strip() or "A"
        if base not in self._vars:
            return base
        index = 2
        while f"{base}{index}" in self._vars:
            index += 1
        return f"{base}{index}"

    def set_var(
        self,
        name: str,
        value: Any,
        source: str = "Agent",
        notes: str = "",
        *,
        role: str = "",
        origin: str = "",
        parent_refs: list[str] | None = None,
        display_name: str = "",
        created_by_tool: str = "",
    ) -> dict:
        var_name = (name or "").strip()
        if not var_name:
            return {"status": "error", "message": "变量名不能为空"}
        self._vars[var_name] = value
        self._slice_budget.pop(var_name, None)
        self._set_meta(
            var_name,
            value,
            source,
            notes,
            role=role,
            origin=origin,
            parent_refs=parent_refs,
            display_name=display_name,
            created_by_tool=created_by_tool,
        )
        self._log("set", var_name, {"kind": _kind(value), "shape": _infer_shape(value)})
        return {"status": "ok", "name": var_name, "variable": self._variable_detail(var_name)}

    def set_user_var(self, name: str, value: Any, source: str = "Agent", notes: str = "") -> dict:
        shape = _infer_shape(value)
        element_count = _element_count(shape)
        if element_count > MAX_WORKSPACE_SET_ELEMENTS:
            return self._protocol_error(
                (name or "").strip(),
                "workspace_set 仅用于标量、小向量和轻量变量；大矩阵/长向量必须由加载器或计算工具注册到 Workspace，再通过 ref 引用。",
                {
                    "shape": shape,
                    "element_count": element_count,
                    "limit": MAX_WORKSPACE_SET_ELEMENTS,
                },
            )
        return self.set_var(name=name, value=value, source=source, notes=notes, origin="workspace_set")

    def alias(self, name: str, ref: str, notes: str = "") -> dict:
        alias_name = (name or "").strip()
        ref_name = (ref or "").strip()
        if not alias_name:
            return {"status": "error", "message": "别名不能为空"}
        if ref_name not in self._vars:
            return {"status": "error", "message": f"变量不存在: {ref_name}"}
        if alias_name == ref_name:
            return {"status": "ok", "name": alias_name, "ref": ref_name, "variable": self._variable_detail(ref_name)}
        self._vars[alias_name] = self._vars[ref_name]
        self._slice_budget.pop(alias_name, None)
        ref_meta = self._meta.get(ref_name, {})
        self._set_meta(
            alias_name,
            self._vars[ref_name],
            "workspace_alias",
            notes or f"Alias of {ref_name}",
            role=ref_meta.get("role", ""),
            origin=ref_meta.get("origin", ref_meta.get("source", "workspace_alias")),
            parent_refs=[ref_name],
            display_name=alias_name,
            created_by_tool="workspace_alias",
            alias_of=ref_name,
        )
        self._log("alias", alias_name, {"ref": ref_name})
        return {
            "status": "ok",
            "name": alias_name,
            "ref": ref_name,
            "variable": self._variable_detail(alias_name),
        }

    def bind_role(self, role: str, ref: str) -> dict:
        role_name = (role or "").strip()
        ref_name = (ref or "").strip()
        if not role_name:
            return {"status": "error", "message": "角色不能为空"}
        if ref_name not in self._vars:
            return {"status": "error", "message": f"变量不存在: {ref_name}"}
        self._meta.setdefault(ref_name, {})
        self._meta[ref_name]["role"] = role_name
        self._meta[ref_name]["updatedAt"] = self._now()
        self._log("bind_role", ref_name, {"role": role_name})
        return {"status": "ok", "name": ref_name, "role": role_name, "variable": self._variable_detail(ref_name)}

    def get_var(self, name: str) -> dict:
        var_name = (name or "").strip()
        if var_name not in self._vars:
            return {"status": "error", "message": f"变量不存在: {var_name}"}
        self._log("get", var_name, {"context_value": _is_large_value(self._vars[var_name])})
        return {
            "status": "ok",
            "name": var_name,
            "value": _value_for_context(var_name, self._vars[var_name], meta=self._meta.get(var_name)),
            "variable": self._variable_detail(var_name),
        }

    def get_raw_var(self, name: str) -> Any:
        var_name = (name or "").strip()
        if var_name not in self._vars:
            raise KeyError(f"变量不存在: {var_name}")
        return self._vars[var_name]

    def get_matrix(self, name: str, *, sparse: bool = False) -> Any:
        value = self.get_raw_var(name)
        if sparse:
            matrix = _to_sparse_matrix(value)
            if len(matrix.shape) != 2:
                raise ValueError(f"变量 {name} 不是二维矩阵")
            self._log("get_matrix", name, {"sparse": True, "shape": list(matrix.shape)})
            return matrix
        shape = _infer_shape(value)
        if (
            _storage_type(value) == "sparse"
            and len(shape) == 2
            and _element_count(shape) > MAX_DENSE_CONVERSION_ELEMENTS
        ):
            raise ValueError(
                f"变量 {name} 是大型稀疏矩阵，禁止隐式稠密化；请改用支持 A_ref 的稀疏工具。"
            )
        arr = _to_dense_array(value)
        if arr.ndim != 2:
            raise ValueError(f"变量 {name} 不是二维矩阵")
        self._log("get_matrix", name, {"sparse": bool(sparse), "shape": list(arr.shape)})
        return arr

    def list_vars(self, detail: bool = False) -> dict:
        names = sorted(self._vars.keys())
        self._log("list", None, {"detail": bool(detail), "count": len(names)})
        if not detail:
            return {"status": "ok", "count": len(names), "variables": names}

        return self.snapshot()

    def clear(self, name: str | None = None) -> dict:
        var_name = (name or "").strip()
        if not var_name:
            cleared = sorted(self._vars.keys())
            self._vars.clear()
            self._meta.clear()
            self._versions.clear()
            self._slice_budget.clear()
            self._log("clear_all", None, {"cleared": cleared})
            return {"status": "ok", "cleared": cleared, "message": "已清空全部变量"}
        if var_name in self._vars:
            self._vars.pop(var_name, None)
            self._meta.pop(var_name, None)
            self._versions.pop(var_name, None)
            self._slice_budget.pop(var_name, None)
            self._log("clear", var_name)
            return {"status": "ok", "cleared": [var_name], "message": f"已清除变量: {var_name}"}
        return {"status": "ok", "cleared": [], "message": f"变量不存在，无需清除: {var_name}"}

    def write_ans(self, value: Any, source: str = "last_tool") -> dict:
        self._vars["ans"] = value
        self._slice_budget.pop("ans", None)
        self._set_meta("ans", value, source, "最近一次工具调用结果", role="last_result", origin=source)
        self._log("write_ans", "ans", {"kind": _kind(value), "shape": _infer_shape(value)})
        return {"status": "ok", "name": "ans", "variable": self._variable_detail("ans")}

    def summary(self, name: str) -> dict:
        var_name = (name or "").strip()
        if var_name not in self._vars:
            return {"status": "error", "message": f"变量不存在: {var_name}"}
        value = self._vars[var_name]
        self._log("summary", var_name)
        return {
            "status": "ok",
            "name": var_name,
            "handle": _object_handle(var_name, value, meta=self._meta.get(var_name)),
            "variable": self._variable_detail(var_name),
        }

    def stats(self, name: str) -> dict:
        var_name = (name or "").strip()
        if var_name not in self._vars:
            return {"status": "error", "message": f"变量不存在: {var_name}"}
        value = self._vars[var_name]
        result: dict[str, Any] = {"status": "ok", "name": var_name, "summary": _summary(value)}
        try:
            arr = _to_dense_array(value)
            numeric = np.asarray(arr, dtype=float)
            result.update(
                {
                    "shape": [int(x) for x in numeric.shape],
                    "norm_2": float(np.linalg.norm(numeric)),
                    "min": float(np.nanmin(numeric)) if numeric.size else None,
                    "max": float(np.nanmax(numeric)) if numeric.size else None,
                    "nan_count": int(np.isnan(numeric).sum()),
                    "inf_count": int(np.isinf(numeric).sum()),
                    "element_count": int(numeric.size),
                }
            )
            if numeric.ndim == 2:
                result["fro_norm"] = float(np.linalg.norm(numeric, ord="fro"))
                result.update({k: v for k, v in _matrix_stats(value).items() if k in ("nnz", "density")})
        except Exception as exc:
            result.update({"status": "warning", "message": f"仅返回基础摘要，统计失败: {exc}"})
        self._log("stats", var_name)
        return result

    def structure(self, name: str, tol: float = 1e-10) -> dict:
        var_name = (name or "").strip()
        if var_name not in self._vars:
            return {"status": "error", "message": f"变量不存在: {var_name}"}
        value = self._vars[var_name]
        result = {
            "status": "ok",
            "name": var_name,
            "kind": _kind(value),
            "storage_type": _storage_type(value),
            "shape": _infer_shape(value),
            "summary": _summary(value),
        }
        try:
            arr = _to_dense_array(value)
            if arr.ndim == 2:
                m, n = arr.shape
                result["is_square"] = bool(m == n)
                result["is_symmetric"] = bool(m == n and np.allclose(arr, arr.T, atol=float(tol)))
                result["is_diagonal"] = bool(m == n and np.allclose(arr, np.diag(np.diag(arr)), atol=float(tol)))
                nonzero = np.argwhere(np.abs(arr) > float(tol))
                if nonzero.size:
                    result["bandwidth_lower"] = int(np.max(nonzero[:, 0] - nonzero[:, 1]))
                    result["bandwidth_upper"] = int(np.max(nonzero[:, 1] - nonzero[:, 0]))
                result.update({k: v for k, v in _matrix_stats(value).items() if k in ("nnz", "density", "format")})
        except Exception as exc:
            result.update({"status": "warning", "message": f"结构分析失败: {exc}"})
        self._log("structure", var_name)
        return result

    def read_slice(self, name: str, rows: list[int] | None = None, cols: list[int] | None = None) -> dict:
        var_name = (name or "").strip()
        if var_name not in self._vars:
            return {"status": "error", "message": f"变量不存在: {var_name}"}
        try:
            arr = _to_dense_array(self._vars[var_name])
            if arr.ndim == 1:
                row_idx = [int(i) for i in (rows or list(range(min(arr.shape[0], MAX_SLICE_ELEMENTS))))]
                if len(row_idx) > MAX_SLICE_ELEMENTS:
                    return self._protocol_error(
                        var_name,
                        f"单次切片最多返回 {MAX_SLICE_ELEMENTS} 个元素",
                        {"requested_elements": len(row_idx), "limit": MAX_SLICE_ELEMENTS},
                    )
                sliced = arr[row_idx]
                request_fingerprint = f"rows:{','.join(str(i) for i in row_idx)}"
            elif arr.ndim == 2:
                row_idx = [int(i) for i in (rows or list(range(min(arr.shape[0], 10))))]
                col_idx = [int(i) for i in (cols or list(range(min(arr.shape[1], 10))))]
                if len(row_idx) * len(col_idx) > MAX_SLICE_ELEMENTS:
                    return self._protocol_error(
                        var_name,
                        f"单次切片最多返回 {MAX_SLICE_ELEMENTS} 个元素",
                        {"requested_elements": len(row_idx) * len(col_idx), "limit": MAX_SLICE_ELEMENTS},
                    )
                sliced = arr[np.ix_(row_idx, col_idx)]
                request_fingerprint = (
                    f"rows:{','.join(str(i) for i in row_idx)}|cols:{','.join(str(i) for i in col_idx)}"
                )
            else:
                return {"status": "error", "message": "仅支持一维或二维对象切片"}
            returned_elements = int(np.asarray(sliced).size)
            total_elements = int(arr.size)
            if total_elements > MAX_CONTEXT_ELEMENTS:
                budget = self._slice_budget.setdefault(
                    var_name,
                    {"calls": 0, "elements": 0, "fingerprints": set()},
                )
                projected_calls = int(budget["calls"]) + 1
                projected_elements = int(budget["elements"]) + returned_elements
                coverage_limit = max(
                    MAX_SLICE_ELEMENTS,
                    min(MAX_CUMULATIVE_SLICE_ELEMENTS, int(total_elements * MAX_SLICE_COVERAGE_RATIO)),
                )
                is_new_region = request_fingerprint not in budget["fingerprints"]
                if projected_calls > MAX_SLICE_CALLS_PER_VAR or (
                    is_new_region and projected_elements > coverage_limit
                ):
                    detail = {
                        "slice_calls": int(budget["calls"]),
                        "slice_elements_returned": int(budget["elements"]),
                        "requested_elements": returned_elements,
                        "limit": coverage_limit,
                        "max_calls": MAX_SLICE_CALLS_PER_VAR,
                    }
                    self._log("slice_budget_exceeded", var_name, detail)
                    return self._protocol_error(
                        var_name,
                        "受控切片累计预算已用尽；请改用摘要、统计、结构分析或后端计算工具，不要通过多次切片重建完整对象。",
                        detail,
                    )
                budget["calls"] = projected_calls
                budget["elements"] = projected_elements
                budget["fingerprints"].add(request_fingerprint)
            self._log(
                "slice",
                var_name,
                {
                    "rows": rows,
                    "cols": cols,
                    "returned_shape": list(np.asarray(sliced).shape),
                    "returned_elements": returned_elements,
                },
            )
            return {
                "status": "ok",
                "name": var_name,
                "shape": [int(x) for x in np.asarray(sliced).shape],
                "data": np.asarray(sliced).tolist(),
                "policy": {
                    "max_elements": MAX_SLICE_ELEMENTS,
                    "max_calls_per_var": MAX_SLICE_CALLS_PER_VAR,
                    "max_cumulative_elements": MAX_CUMULATIVE_SLICE_ELEMENTS,
                    "max_coverage_ratio": MAX_SLICE_COVERAGE_RATIO,
                },
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def audit(self, limit: int = 50) -> dict:
        count = max(0, min(int(limit or 50), MAX_AUDIT_RECORDS))
        return {"status": "ok", "count": min(count, len(self._audit)), "records": self._audit[-count:]}
