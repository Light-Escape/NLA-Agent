from __future__ import annotations

import unittest

import numpy as np
from scipy.sparse import csc_matrix

from sparse_backend import (
    cg_sparse,
    eigs_sparse,
    eigsh_sparse,
    get_sparse_backend_info,
    gmres_sparse,
    spsolve_sparse,
)


def _to_csc_payload(rows: list[list[float]]) -> dict:
    matrix = csc_matrix(np.asarray(rows, dtype=float))
    return {
        "format": "csc",
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": int(matrix.nnz),
        "indptr": matrix.indptr.tolist(),
        "indices": matrix.indices.tolist(),
        "data": matrix.data.tolist(),
    }


class TestSparseBackend(unittest.TestCase):
    def test_sparse_backend_info(self):
        ret = get_sparse_backend_info()
        self.assertEqual(ret["status"], "ok")
        self.assertIn("spsolve", ret["available_sparse_linalg_funcs"])
        self.assertIn("cg", ret["available_sparse_linalg_funcs"])

    def test_spsolve_sparse(self):
        A = [[4.0, 1.0], [1.0, 3.0]]
        b = [1.0, 2.0]
        ret = spsolve_sparse(A_csc=_to_csc_payload(A), b=b)
        self.assertEqual(ret["status"], "ok")
        x = np.asarray(ret["x"], dtype=float)
        self.assertLess(np.linalg.norm(np.asarray(A) @ x - np.asarray(b)), 1e-10)

    def test_cg_sparse(self):
        A = [[4.0, 1.0], [1.0, 3.0]]
        b = [1.0, 2.0]
        ret = cg_sparse(A_csc=_to_csc_payload(A), b=b, tol=1e-12)
        self.assertEqual(ret["status"], "ok")
        self.assertTrue(ret["converged"])
        x = np.asarray(ret["x"], dtype=float)
        self.assertLess(np.linalg.norm(np.asarray(A) @ x - np.asarray(b)), 1e-10)

    def test_gmres_sparse(self):
        A = [[3.0, 1.0], [0.0, 2.0]]
        b = [1.0, 4.0]
        ret = gmres_sparse(A_csc=_to_csc_payload(A), b=b, tol=1e-12)
        self.assertEqual(ret["status"], "ok")
        self.assertTrue(ret["converged"])
        x = np.asarray(ret["x"], dtype=float)
        self.assertLess(np.linalg.norm(np.asarray(A) @ x - np.asarray(b)), 1e-10)

    def test_eigsh_sparse(self):
        A = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]]
        ret = eigsh_sparse(A_csc=_to_csc_payload(A), k=1, which="LA")
        self.assertEqual(ret["status"], "ok")
        self.assertAlmostEqual(float(ret["eigenvalues"][0]), 5.0, places=10)
        self.assertLess(ret["residual_norms_2"][0], 1e-10)

    def test_eigs_sparse(self):
        A = [[1.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
        ret = eigs_sparse(A_csc=_to_csc_payload(A), k=1, which="LM")
        self.assertEqual(ret["status"], "ok")
        self.assertAlmostEqual(float(ret["eigenvalues"][0]), 4.0, places=10)
        self.assertLess(ret["residual_norms_2"][0], 1e-10)


if __name__ == "__main__":
    unittest.main()
