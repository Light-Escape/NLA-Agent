"""
LAPACK/BLAS 后端封装工具。
"""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np
import scipy
from scipy.linalg import blas, lapack
from scipy.linalg import lstsq as scipy_lstsq


_DTYPE_ALIASES = {
    "float64": np.float64,
    "double": np.float64,
    "d": np.float64,
    "float32": np.float32,
    "single": np.float32,
    "s": np.float32,
    "complex128": np.complex128,
    "complex": np.complex128,
    "z": np.complex128,
    "complex64": np.complex64,
    "c": np.complex64,
}

_LAPACK_PROBE_GROUPS = {
    "linear_systems": ("gesv", "posv", "sysv"),
    "least_squares": ("gels", "gelsy", "gelsd", "gelss"),
    "svd": ("gesvd", "gesdd", "gejsv"),
    "eigen": ("geev", "gees", "ggev", "syev", "syevd", "syevr", "syevx", "sygv", "sygvd", "sygvx"),
    "factorizations": ("getrf", "getrs", "getri", "potrf", "potrs", "geqrf", "gerqf", "geqp3", "orgqr", "ormqr"),
    "condition_estimates": ("gecon", "pocon"),
}

_REQUIRED_LAPACK_PROBES = {"gesv", "posv", "gelsd", "gesvd"}

_KNOWN_ROUTINE_NAMES = {
    routine
    for routines in _LAPACK_PROBE_GROUPS.values()
    for routine in routines
} | {
    "gemm",
    "gemv",
    "ger",
    "syrk",
    "herk",
    "trsm",
    "trsv",
    "dot",
    "nrm2",
    "axpy",
    "scal",
}


def _to_2d_float64(rows: list[list[float]], *, name: str) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组")
    if arr.size == 0:
        raise ValueError(f"{name} 不能为空")
    return np.asfortranarray(arr)


def _to_rhs(rhs: list[float] | list[list[float]]) -> tuple[np.ndarray, bool]:
    arr = np.asarray(rhs, dtype=np.float64)
    if arr.ndim == 1:
        return np.asfortranarray(arr.reshape(-1, 1)), True
    if arr.ndim == 2:
        return np.asfortranarray(arr), False
    raise ValueError("b 必须是一维或二维数组")


def _serialize_rhs(x: np.ndarray, was_vector: bool) -> list[float] | list[list[float]]:
    if was_vector:
        return x.reshape(-1).tolist()
    return x.tolist()


def _dtype_from_name(dtype: str | None) -> np.dtype:
    key = (dtype or "float64").strip().lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError("dtype 仅支持 float64/float32/complex128/complex64")
    return np.dtype(_DTYPE_ALIASES[key])


def _to_fortran_array(value: Any, *, dtype: np.dtype, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        raise ValueError(f"{name} 必须是一维或二维数组")
    if arr.size == 0:
        raise ValueError(f"{name} 不能为空")
    return np.asfortranarray(arr)


def _normalize_driver_name(func_name: str) -> str:
    name = (func_name or "").strip().lower()
    if not name:
        raise ValueError("func_name 不能为空")
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        raise ValueError("func_name 只能包含字母、数字和下划线")
    # SciPy get_*_funcs 用无精度前缀名选择函数；允许用户输入 dgesv/sgejsv 等习惯写法。
    # 但 syev/syevd/syevr 本身就是无精度前缀名，不能把开头的 s 错误截掉。
    if name in _KNOWN_ROUTINE_NAMES:
        return name
    if len(name) > 2 and name[0] in {"s", "d", "c", "z"} and name[1:] in _KNOWN_ROUTINE_NAMES:
        return name[1:]
    return name


def _serialize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    return value


def _serialize_driver_outputs(raw: Any, output_names: Optional[list[str]] = None) -> dict:
    if isinstance(raw, tuple):
        values = list(raw)
    else:
        values = [raw]

    names = output_names or []
    outputs: dict[str, Any] = {}
    for idx, value in enumerate(values):
        key = names[idx] if idx < len(names) and names[idx] else f"result_{idx}"
        outputs[key] = _serialize_value(value)
    return outputs


def _scipy_func_name(func: Any, fallback: str) -> str:
    raw_name = str(getattr(func, "__name__", "") or fallback)
    return raw_name.removeprefix("function ").strip()


def _build_driver_inputs(
    arrays: dict[str, Any],
    kwargs: Optional[dict[str, Any]],
    dtype: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], tuple[np.ndarray, ...], str]:
    dtype_obj = _dtype_from_name(dtype)
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError("arrays 必须是非空字典，例如 {'a': [[1, 2], [3, 4]]}")

    array_kwargs = {
        str(name): _to_fortran_array(value, dtype=dtype_obj, name=str(name))
        for name, value in arrays.items()
    }
    call_kwargs = dict(kwargs or {})
    return array_kwargs, call_kwargs, tuple(array_kwargs.values()), str(dtype_obj)


def get_linalg_backend_info() -> dict:
    """
    返回当前 NumPy/SciPy 与 LAPACK/BLAS 可用性信息。
    """
    info = {
        "status": "ok",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "blas_opt_info": {},
        "lapack_opt_info": {},
        "available_lapack_funcs": [],
        "lapack_probe_groups": {},
        "unavailable_lapack_funcs": {},
        "available_blas_funcs": [],
    }

    get_info = getattr(np.__config__, "get_info", None)
    if callable(get_info):
        info["blas_opt_info"] = get_info("blas_opt_info") or {}
        info["lapack_opt_info"] = get_info("lapack_opt_info") or {}

    lapack_errors = {}
    required_errors = {}
    for group_name, func_names in _LAPACK_PROBE_GROUPS.items():
        group_available = []
        for func_name in func_names:
            try:
                resolved = lapack.get_lapack_funcs(func_name).__name__
                info["available_lapack_funcs"].append(resolved)
                group_available.append(resolved)
            except Exception as exc:
                lapack_errors[func_name] = str(exc)
                if func_name in _REQUIRED_LAPACK_PROBES:
                    required_errors[func_name] = str(exc)
        info["lapack_probe_groups"][group_name] = group_available
    if required_errors:
        info["status"] = "warning"
        info["lapack_errors"] = required_errors
    if lapack_errors:
        info["unavailable_lapack_funcs"] = lapack_errors

    try:
        info["available_blas_funcs"] = [blas.get_blas_funcs("gemm").__name__]
    except Exception as exc:
        info["status"] = "warning"
        info["blas_error"] = str(exc)

    return info


def call_lapack(
    func_name: str,
    arrays: dict[str, Any],
    kwargs: Optional[dict[str, Any]] = None,
    dtype: str = "float64",
    output_names: Optional[list[str]] = None,
) -> dict:
    """
    MATLAB 风格的通用 LAPACK driver 调用入口。

    该工具不直接加载 Windows DLL/.so，而是通过 SciPy f2py 包装层自动选择
    d/s/c/z 前缀函数。例如 func_name='gejsv' 或 'dgejsv' 都会解析到 dgejsv。
    """
    try:
        driver_name = _normalize_driver_name(func_name)
        array_kwargs, call_kwargs, selector_arrays, used_dtype = _build_driver_inputs(
            arrays,
            kwargs,
            dtype,
        )
        driver = lapack.get_lapack_funcs(driver_name, selector_arrays)
        raw = driver(**array_kwargs, **call_kwargs)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    outputs = _serialize_driver_outputs(raw, output_names)
    info = outputs.get("info")
    status = "ok" if info is None or int(info) == 0 else "warning"
    return {
        "status": status,
        "backend": "scipy.linalg.lapack",
        "requested_func": func_name,
        "resolved_func": _scipy_func_name(driver, driver_name),
        "dtype": used_dtype,
        "outputs": outputs,
    }


def call_blas(
    func_name: str,
    arrays: dict[str, Any],
    kwargs: Optional[dict[str, Any]] = None,
    dtype: str = "float64",
    output_names: Optional[list[str]] = None,
) -> dict:
    """
    MATLAB 风格的通用 BLAS routine 调用入口，通过 SciPy f2py 包装层选择实现。
    """
    try:
        routine_name = _normalize_driver_name(func_name)
        array_kwargs, call_kwargs, selector_arrays, used_dtype = _build_driver_inputs(
            arrays,
            kwargs,
            dtype,
        )
        routine = blas.get_blas_funcs(routine_name, selector_arrays)
        raw = routine(**array_kwargs, **call_kwargs)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    return {
        "status": "ok",
        "backend": "scipy.linalg.blas",
        "requested_func": func_name,
        "resolved_func": _scipy_func_name(routine, routine_name),
        "dtype": used_dtype,
        "outputs": _serialize_driver_outputs(raw, output_names),
    }


def solve_linear_lapack(
    A_rows: list[list[float]],
    b: list[float] | list[list[float]],
    assume: str = "auto",
) -> dict:
    """
    用 LAPACK 求解 Ax=b。

    assume:
    - auto: 先尝试 posv（对称正定）再回退 gesv
    - pos:  仅用 posv
    - gen:  仅用 gesv
    """
    try:
        A = _to_2d_float64(A_rows, name="A_rows")
        B, b_is_vector = _to_rhs(b)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    m, n = A.shape
    if m != n:
        return {"status": "error", "message": "线性方程求解要求 A 为方阵"}
    if B.shape[0] != n:
        return {"status": "error", "message": "b 与 A 维度不匹配"}

    strategy = (assume or "auto").strip().lower()
    if strategy not in {"auto", "pos", "gen"}:
        return {"status": "error", "message": "assume 仅支持 auto/pos/gen"}

    attempt_order = []
    if strategy == "auto":
        attempt_order = ["pos", "gen"]
    else:
        attempt_order = [strategy]

    A_orig = A.copy(order="F")
    B_orig = B.copy(order="F")
    errors: list[str] = []

    for mode in attempt_order:
        if mode == "pos":
            posv = lapack.get_lapack_funcs("posv", (A_orig, B_orig))
            c, x, info = posv(
                A_orig.copy(order="F"),
                B_orig.copy(order="F"),
                lower=False,
                overwrite_a=False,
                overwrite_b=False,
            )
            if info == 0:
                residual = float(np.linalg.norm(A_orig @ x - B_orig))
                return {
                    "status": "ok",
                    "solver": "lapack.posv",
                    "assume": mode,
                    "fallback_used": strategy == "auto" and mode != attempt_order[0],
                    "x": _serialize_rhs(x, b_is_vector),
                    "residual_norm_2": residual,
                    "lapack_info": int(info),
                }
            errors.append(f"posv 失败，info={int(info)}")
            continue

        gesv = lapack.get_lapack_funcs("gesv", (A_orig, B_orig))
        lu, piv, x, info = gesv(
            A_orig.copy(order="F"),
            B_orig.copy(order="F"),
            overwrite_a=False,
            overwrite_b=False,
        )
        if info == 0:
            residual = float(np.linalg.norm(A_orig @ x - B_orig))
            return {
                "status": "ok",
                "solver": "lapack.gesv",
                "assume": mode,
                "fallback_used": strategy == "auto" and mode != attempt_order[0],
                "x": _serialize_rhs(x, b_is_vector),
                "residual_norm_2": residual,
                "lapack_info": int(info),
            }
        errors.append(f"gesv 失败，info={int(info)}")

    return {"status": "error", "message": "；".join(errors) or "LAPACK 求解失败"}


def least_squares_lapack(
    A_rows: list[list[float]],
    b: list[float] | list[list[float]],
    driver: str = "gelsd",
) -> dict:
    """
    用 LAPACK driver 计算最小二乘。
    """
    try:
        A = _to_2d_float64(A_rows, name="A_rows")
        B, b_is_vector = _to_rhs(b)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    if A.shape[0] != B.shape[0]:
        return {"status": "error", "message": "A 与 b 行数不匹配"}

    drv = (driver or "gelsd").strip().lower()
    if drv not in {"gelsd", "gelss", "gelsy"}:
        return {"status": "error", "message": "driver 仅支持 gelsd/gelss/gelsy"}

    x, residuals, rank, singular_values = scipy_lstsq(A, B, lapack_driver=drv)
    residual_norm = float(np.linalg.norm(A @ x - B))

    return {
        "status": "ok",
        "solver": f"lapack.{drv}",
        "x": _serialize_rhs(np.asarray(x), b_is_vector),
        "rank": int(rank),
        "residual_norm_2": residual_norm,
        "residuals": np.asarray(residuals).tolist(),
        "singular_values": np.asarray(singular_values).tolist(),
    }


def gemm_blas(
    A_rows: list[list[float]],
    B_rows: list[list[float]],
    alpha: float = 1.0,
    beta: float = 0.0,
    C_rows: Optional[list[list[float]]] = None,
    trans_a: bool = False,
    trans_b: bool = False,
) -> dict:
    """
    用 BLAS GEMM 计算矩阵乘法：
    C <- alpha * op(A) * op(B) + beta * C
    """
    try:
        A = _to_2d_float64(A_rows, name="A_rows")
        B = _to_2d_float64(B_rows, name="B_rows")
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    op_a = A.T if trans_a else A
    op_b = B.T if trans_b else B
    if op_a.shape[1] != op_b.shape[0]:
        return {"status": "error", "message": "op(A) 与 op(B) 维度不匹配"}

    c = None
    if C_rows is not None:
        try:
            C = _to_2d_float64(C_rows, name="C_rows")
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
        if C.shape != (op_a.shape[0], op_b.shape[1]):
            return {"status": "error", "message": "C_rows 形状与乘积结果不匹配"}
        c = C

    gemm = blas.get_blas_funcs("gemm", (A, B))
    out = gemm(
        alpha=float(alpha),
        a=A,
        b=B,
        beta=float(beta),
        c=c,
        trans_a=bool(trans_a),
        trans_b=bool(trans_b),
        overwrite_c=False,
    )

    return {
        "status": "ok",
        "solver": "blas.gemm",
        "shape": [int(out.shape[0]), int(out.shape[1])],
        "C": np.asarray(out).tolist(),
    }
