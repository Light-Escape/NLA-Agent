from __future__ import annotations

import unittest

import numpy as np

from linalg_backend import (
    call_blas,
    call_lapack,
    gemm_blas,
    get_linalg_backend_info,
    least_squares_lapack,
    solve_linear_lapack,
)


class TestLinalgBackend(unittest.TestCase):
    def test_backend_info(self):
        ret = get_linalg_backend_info()
        self.assertIn(ret["status"], ("ok", "warning"))
        self.assertIn("numpy_version", ret)
        self.assertIn("scipy_version", ret)
        self.assertTrue(isinstance(ret.get("available_lapack_funcs", []), list))
        self.assertTrue(isinstance(ret.get("lapack_probe_groups", {}), dict))
        self.assertIn("eigen", ret.get("lapack_probe_groups", {}))

    def test_solve_linear_lapack_auto(self):
        A = [[4.0, 1.0], [1.0, 3.0]]
        b = [1.0, 2.0]
        ret = solve_linear_lapack(A_rows=A, b=b, assume="auto")
        self.assertEqual(ret["status"], "ok")
        x = np.asarray(ret["x"], dtype=float)
        self.assertLess(np.linalg.norm(np.asarray(A) @ x - np.asarray(b)), 1e-10)

    def test_solve_linear_lapack_fallback_to_gesv(self):
        A = [[0.0, 1.0], [1.0, 0.0]]
        b = [1.0, 2.0]
        ret = solve_linear_lapack(A_rows=A, b=b, assume="auto")
        self.assertEqual(ret["status"], "ok")
        self.assertEqual(ret["solver"], "lapack.gesv")
        self.assertTrue(ret["fallback_used"])

    def test_least_squares_lapack(self):
        A = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
        b = [1.0, 2.0, 2.0]
        ret = least_squares_lapack(A_rows=A, b=b, driver="gelsd")
        self.assertEqual(ret["status"], "ok")
        x = np.asarray(ret["x"], dtype=float)
        self.assertLess(np.linalg.norm(np.asarray(A) @ x - np.asarray(b)), 0.8)

    def test_gemm_blas(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        ret = gemm_blas(A_rows=A, B_rows=B)
        self.assertEqual(ret["status"], "ok")
        C = np.asarray(ret["C"], dtype=float)
        np.testing.assert_allclose(C, np.asarray(A) @ np.asarray(B), atol=1e-12)

    def test_call_lapack_gesv_with_matlab_style_driver_name(self):
        A = [[3.0, 1.0], [1.0, 2.0]]
        b = [[9.0], [8.0]]
        ret = call_lapack(
            func_name="dgesv",
            arrays={"a": A, "b": b},
            output_names=["lu", "piv", "x", "info"],
        )
        self.assertEqual(ret["status"], "ok")
        self.assertEqual(ret["resolved_func"], "dgesv")
        x = np.asarray(ret["outputs"]["x"], dtype=float)
        np.testing.assert_allclose(np.asarray(A) @ x, np.asarray(b), atol=1e-12)
        self.assertEqual(ret["outputs"]["info"], 0)

    def test_call_lapack_gejsv_if_available(self):
        A = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        ret = call_lapack(
            func_name="gejsv",
            arrays={"a": A},
            kwargs={"jobu": 1, "jobv": 1},
            output_names=["s", "u", "v", "workout", "iworkout", "info"],
        )
        if ret["status"] == "error" and "gejsv" in ret.get("message", "").lower():
            self.skipTest(ret["message"])
        self.assertEqual(ret["status"], "ok")
        self.assertEqual(ret["outputs"]["info"], 0)
        singular_values = np.asarray(ret["outputs"]["s"], dtype=float)
        np.testing.assert_allclose(singular_values, np.linalg.svd(np.asarray(A), compute_uv=False), atol=1e-10)

    def test_call_lapack_syev_and_dsyev_names(self):
        A = [[2.0, 1.0], [1.0, 2.0]]
        expected = np.linalg.eigvalsh(np.asarray(A))
        for func_name in ("syev", "dsyev"):
            with self.subTest(func_name=func_name):
                ret = call_lapack(
                    func_name=func_name,
                    arrays={"a": A},
                    kwargs={"compute_v": 1},
                    output_names=["w", "v", "info"],
                )
                if ret["status"] == "error" and "syev" in ret.get("message", "").lower():
                    self.skipTest(ret["message"])
                self.assertEqual(ret["status"], "ok")
                self.assertEqual(ret["resolved_func"], "dsyev")
                self.assertEqual(ret["outputs"]["info"], 0)
                np.testing.assert_allclose(np.asarray(ret["outputs"]["w"], dtype=float), expected, atol=1e-12)

    def test_call_lapack_geev_if_available(self):
        A = [[0.0, -1.0], [1.0, 0.0]]
        ret = call_lapack(
            func_name="geev",
            arrays={"a": A},
            kwargs={"compute_vl": 0, "compute_vr": 0},
            output_names=["wr", "wi", "vl", "vr", "info"],
        )
        if ret["status"] == "error" and "geev" in ret.get("message", "").lower():
            self.skipTest(ret["message"])
        self.assertEqual(ret["status"], "ok")
        self.assertEqual(ret["resolved_func"], "dgeev")
        self.assertEqual(ret["outputs"]["info"], 0)
        eigenvalues = np.asarray(ret["outputs"]["wr"], dtype=float) + 1j * np.asarray(ret["outputs"]["wi"], dtype=float)
        np.testing.assert_allclose(np.sort_complex(eigenvalues), np.sort_complex(np.linalg.eigvals(np.asarray(A))), atol=1e-12)

    def test_call_blas_gemm(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        ret = call_blas(
            func_name="dgemm",
            arrays={"a": A, "b": B},
            kwargs={"alpha": 1.0},
            output_names=["c"],
        )
        self.assertEqual(ret["status"], "ok")
        np.testing.assert_allclose(np.asarray(ret["outputs"]["c"], dtype=float), np.asarray(A) @ np.asarray(B), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
