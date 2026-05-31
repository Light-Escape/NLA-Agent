"""
SciPy 稀疏线性代数后端封装工具。
"""

from __future__ import annotations

import inspect
from typing import Optional

import numpy as np
import scipy
from scipy.sparse import csc_matrix, issparse
from scipy.sparse import linalg as sparse_linalg

try:
    from .parsers import csc_payload_to_matrix
except ImportError:
    from parsers import csc_payload_to_matrix


def _to_sparse_matrix(
    *,
    A_csc: Optional[dict] = None,
    A_rows: Optional[list[list[float]]] = None,
    name: str = "A",
) -> csc_matrix:
    if A_csc is not None:
        if issparse(A_csc):
            matrix = A_csc
        else:
            matrix = csc_payload_to_matrix(A_csc)
    elif A_rows is not None:
        arr = np.asarray(A_rows, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{name}_rows 必须是二维数组")
        if arr.size == 0:
            raise ValueError(f"{name}_rows 不能为空")
        matrix = csc_matrix(arr)
    else:
        raise ValueError("A_csc 与 A_rows 至少提供一个")

    if not issparse(matrix):
        matrix = csc_matrix(matrix)
    return matrix.astype(float, copy=False).tocsc()


def _to_rhs(rhs: list[float] | list[list[float]]) -> tuple[np.ndarray, bool]:
    arr = np.asarray(rhs, dtype=float)
    if arr.ndim == 1:
        return arr, True
    if arr.ndim == 2:
        return arr, False
    raise ValueError("b 必须是一维或二维数组")


def _serialize_scalar(value) -> float | dict:
    scalar = np.asarray(value).item()
    if isinstance(scalar, complex):
        if abs(scalar.imag) <= 1e-12:
            return float(scalar.real)
        return {"real": float(scalar.real), "imag": float(scalar.imag)}
    return float(scalar)


def _serialize_array(arr: np.ndarray):
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return _serialize_scalar(arr)
    return [_serialize_array(item) for item in arr]


def _serialize_solution(x: np.ndarray, was_vector: bool):
    arr = np.asarray(x)
    if was_vector:
        arr = arr.reshape(-1)
    return _serialize_array(arr)


def _iterative_kwargs(tol: float, maxiter: Optional[int]) -> dict:
    params = inspect.signature(sparse_linalg.cg).parameters
    kwargs = {"maxiter": maxiter}
    if "rtol" in params:
        kwargs["rtol"] = float(tol)
        kwargs["atol"] = 0.0
    else:
        kwargs["tol"] = float(tol)
    return kwargs


def get_sparse_backend_info() -> dict:
    """返回 SciPy sparse.linalg 可用性信息。"""
    funcs = ["spsolve", "cg", "gmres", "eigsh", "eigs"]
    return {
        "status": "ok",
        "scipy_version": scipy.__version__,
        "available_sparse_linalg_funcs": [
            name for name in funcs if callable(getattr(sparse_linalg, name, None))
        ],
    }


def spsolve_sparse(
    A_csc: Optional[dict] = None,
    b: list[float] | list[list[float]] = None,
    A_rows: Optional[list[list[float]]] = None,
) -> dict:
    """用 scipy.sparse.linalg.spsolve 求解稀疏方阵线性系统 Ax=b。"""
    try:
        A = _to_sparse_matrix(A_csc=A_csc, A_rows=A_rows)
        if b is None:
            return {"status": "error", "message": "b 不能为空"}
        B, b_is_vector = _to_rhs(b)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    m, n = A.shape
    if m != n:
        return {"status": "error", "message": "spsolve 要求 A 为方阵"}
    if B.shape[0] != n:
        return {"status": "error", "message": "b 与 A 维度不匹配"}

    try:
        x = sparse_linalg.spsolve(A, B)
        x_arr = np.asarray(x)
        residual = float(np.linalg.norm(A @ x_arr - B))
        return {
            "status": "ok",
            "solver": "scipy.sparse.linalg.spsolve",
            "shape": [int(m), int(n)],
            "nnz": int(A.nnz),
            "x": _serialize_solution(x_arr, b_is_vector),
            "residual_norm_2": residual,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def cg_sparse(
    A_csc: Optional[dict] = None,
    b: list[float] = None,
    A_rows: Optional[list[list[float]]] = None,
    tol: float = 1e-8,
    maxiter: Optional[int] = None,
) -> dict:
    """用共轭梯度法求解稀疏对称正定线性系统 Ax=b。"""
    try:
        A = _to_sparse_matrix(A_csc=A_csc, A_rows=A_rows)
        if b is None:
            return {"status": "error", "message": "b 不能为空"}
        B, b_is_vector = _to_rhs(b)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    m, n = A.shape
    if m != n:
        return {"status": "error", "message": "cg 要求 A 为方阵"}
    if not b_is_vector:
        return {"status": "error", "message": "cg 仅支持一维 b"}
    if B.shape[0] != n:
        return {"status": "error", "message": "b 与 A 维度不匹配"}

    try:
        x, info = sparse_linalg.cg(A, B, **_iterative_kwargs(tol, maxiter))
        residual = float(np.linalg.norm(A @ x - B))
        converged = int(info) == 0
        return {
            "status": "ok" if converged else "warning",
            "solver": "scipy.sparse.linalg.cg",
            "shape": [int(m), int(n)],
            "nnz": int(A.nnz),
            "x": _serialize_solution(np.asarray(x), True),
            "residual_norm_2": residual,
            "converged": converged,
            "info": int(info),
            "message": "已收敛" if converged else "未在 maxiter 内收敛或参数非法",
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def gmres_sparse(
    A_csc: Optional[dict] = None,
    b: list[float] = None,
    A_rows: Optional[list[list[float]]] = None,
    tol: float = 1e-8,
    restart: Optional[int] = None,
    maxiter: Optional[int] = None,
) -> dict:
    """用 GMRES 求解一般稀疏方阵线性系统 Ax=b。"""
    try:
        A = _to_sparse_matrix(A_csc=A_csc, A_rows=A_rows)
        if b is None:
            return {"status": "error", "message": "b 不能为空"}
        B, b_is_vector = _to_rhs(b)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    m, n = A.shape
    if m != n:
        return {"status": "error", "message": "gmres 要求 A 为方阵"}
    if not b_is_vector:
        return {"status": "error", "message": "gmres 仅支持一维 b"}
    if B.shape[0] != n:
        return {"status": "error", "message": "b 与 A 维度不匹配"}

    try:
        kwargs = _iterative_kwargs(tol, maxiter)
        kwargs["restart"] = restart
        x, info = sparse_linalg.gmres(A, B, **kwargs)
        residual = float(np.linalg.norm(A @ x - B))
        converged = int(info) == 0
        return {
            "status": "ok" if converged else "warning",
            "solver": "scipy.sparse.linalg.gmres",
            "shape": [int(m), int(n)],
            "nnz": int(A.nnz),
            "x": _serialize_solution(np.asarray(x), True),
            "residual_norm_2": residual,
            "converged": converged,
            "info": int(info),
            "message": "已收敛" if converged else "未在 maxiter 内收敛或参数非法",
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def eigsh_sparse(
    A_csc: Optional[dict] = None,
    A_rows: Optional[list[list[float]]] = None,
    k: int = 6,
    which: str = "LM",
    tol: float = 0.0,
    maxiter: Optional[int] = None,
    return_eigenvectors: bool = True,
) -> dict:
    """用 scipy.sparse.linalg.eigsh 计算稀疏对称矩阵的部分特征值。"""
    return _sparse_eigs(
        solver_name="eigsh",
        A_csc=A_csc,
        A_rows=A_rows,
        k=k,
        which=which,
        tol=tol,
        maxiter=maxiter,
        return_eigenvectors=return_eigenvectors,
    )


def eigs_sparse(
    A_csc: Optional[dict] = None,
    A_rows: Optional[list[list[float]]] = None,
    k: int = 6,
    which: str = "LM",
    tol: float = 0.0,
    maxiter: Optional[int] = None,
    return_eigenvectors: bool = True,
) -> dict:
    """用 scipy.sparse.linalg.eigs 计算一般稀疏矩阵的部分特征值。"""
    return _sparse_eigs(
        solver_name="eigs",
        A_csc=A_csc,
        A_rows=A_rows,
        k=k,
        which=which,
        tol=tol,
        maxiter=maxiter,
        return_eigenvectors=return_eigenvectors,
    )


def _sparse_eigs(
    *,
    solver_name: str,
    A_csc: Optional[dict],
    A_rows: Optional[list[list[float]]],
    k: int,
    which: str,
    tol: float,
    maxiter: Optional[int],
    return_eigenvectors: bool,
) -> dict:
    try:
        A = _to_sparse_matrix(A_csc=A_csc, A_rows=A_rows)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    m, n = A.shape
    if m != n:
        return {"status": "error", "message": "特征值计算要求 A 为方阵"}
    if n <= 1:
        return {"status": "error", "message": "矩阵阶数过小，无法使用 ARPACK 部分特征值算法"}
    if not isinstance(k, int) or k < 1 or k >= n:
        return {"status": "error", "message": "k 必须满足 1 <= k < n"}

    try:
        solver = getattr(sparse_linalg, solver_name)
        result = solver(
            A,
            k=int(k),
            which=(which or "LM"),
            tol=float(tol),
            maxiter=maxiter,
            return_eigenvectors=bool(return_eigenvectors),
        )
        if return_eigenvectors:
            eigenvalues, eigenvectors = result
            residuals = [
                float(np.linalg.norm(A @ eigenvectors[:, idx] - eigenvalues[idx] * eigenvectors[:, idx]))
                for idx in range(len(eigenvalues))
            ]
            payload = {
                "eigenvectors": _serialize_array(np.asarray(eigenvectors)),
                "residual_norms_2": residuals,
            }
        else:
            eigenvalues = result
            payload = {}

        return {
            "status": "ok",
            "solver": f"scipy.sparse.linalg.{solver_name}",
            "shape": [int(m), int(n)],
            "nnz": int(A.nnz),
            "k": int(k),
            "which": which,
            "eigenvalues": _serialize_array(np.asarray(eigenvalues)),
            **payload,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
