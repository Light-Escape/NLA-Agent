from __future__ import annotations

import unittest
from pathlib import Path
import sys
import json

import numpy as np
from scipy.sparse import issparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import (  # noqa: E402
    _load_matrix_csc_content,
    _run_python_snippet,
    _run_python_workspace_snippet,
    _solve_linear_lapack,
    _spsolve_sparse,
    _workspace_store,
    workspace_alias,
    workspace_audit,
    workspace_bind_role,
    workspace_get,
    workspace_list,
    workspace_set,
    workspace_slice,
)


class TestMatrixHandles(unittest.TestCase):
    def setUp(self):
        _workspace_store.clear()

    def tearDown(self):
        _workspace_store.clear()

    def test_loader_returns_handle_without_matrix_elements(self):
        ret = _load_matrix_csc_content("2 2\n0 2 4\n0 1 0 1\n4.0 1.0 1.0 3.0\n")

        self.assertEqual(ret["status"], "ok")
        self.assertEqual(ret["matrix_ref"], "A")
        self.assertEqual(ret["matrix_refs"], ["A"])
        self.assertEqual(ret["handle"]["matrix_handle"], True)
        self.assertEqual(ret["role"], "system_matrix")
        self.assertNotIn("A_rows", ret)
        self.assertNotIn("A_csc", ret)

        workspace_ret = workspace_get("A")
        self.assertEqual(workspace_ret["status"], "ok")
        self.assertEqual(workspace_ret["value"]["ref"], "A")
        self.assertNotIn("A_rows", workspace_ret["value"])
        self.assertNotIn("A_csc", workspace_ret["value"])

    def test_repeated_loaders_allocate_distinct_matrix_refs(self):
        first = _load_matrix_csc_content("2 2\n0 2 4\n0 1 0 1\n4.0 1.0 1.0 3.0\n")
        second = _load_matrix_csc_content("2 2\n0 1 2\n0 1\n1.0 2.0\n")
        listed = workspace_list(detail=True)

        self.assertEqual(first["matrix_ref"], "A")
        self.assertEqual(second["matrix_ref"], "A2")
        self.assertEqual([item["name"] for item in listed["variables"] if item["name"].startswith("A")], ["A", "A2"])

    def test_workspace_set_rejects_large_matrix_but_alias_and_role_work(self):
        _load_matrix_csc_content("2 2\n0 2 4\n0 1 0 1\n4.0 1.0 1.0 3.0\n")

        blocked = workspace_set("Big", np.eye(9).tolist())
        alias_ret = workspace_alias("K", "A")
        role_ret = workspace_bind_role("preconditioner", "K")
        alias_workspace_ret = workspace_get("K")

        self.assertEqual(blocked["status"], "error")
        self.assertEqual(blocked["error_type"], "workspace_protocol_violation")
        self.assertEqual(alias_ret["status"], "ok")
        self.assertEqual(alias_ret["variable"]["alias_of"], "A")
        self.assertEqual(role_ret["variable"]["role"], "preconditioner")
        self.assertTrue(alias_workspace_ret["value"]["matrix_handle"])

    def test_dense_and_sparse_tools_accept_a_ref(self):
        _load_matrix_csc_content("2 2\n0 2 4\n0 1 0 1\n4.0 1.0 1.0 3.0\n")
        b = [1.0, 2.0]

        dense_ret = _solve_linear_lapack(A_ref="A", b=b)
        sparse_ret = _spsolve_sparse(A_ref="A", b=b)
        sparse_matrix = _workspace_store.get_matrix("A", sparse=True)

        self.assertEqual(dense_ret["status"], "ok")
        self.assertEqual(sparse_ret["status"], "ok")
        self.assertTrue(issparse(sparse_matrix))
        np.testing.assert_allclose(dense_ret["x"], sparse_ret["x"], atol=1e-10)

    def test_plain_python_snippet_cannot_see_workspace_matrix(self):
        _load_matrix_csc_content("2 2\n0 2 4\n0 1 0 1\n4.0 1.0 1.0 3.0\n")

        ret = _run_python_snippet("RESULT = {'shape': A.shape}")

        self.assertEqual(ret["status"], "error")
        self.assertIn("NameError", ret.get("stderr", ""))

    def test_workspace_python_snippet_uses_array_refs(self):
        _load_matrix_csc_content("2 2\n0 2 4\n0 1 0 1\n4.0 1.0 1.0 3.0\n")

        ret = _run_python_workspace_snippet(
            "RESULT = {'shape': list(A.shape), 'nnz': int(A.nnz), 'trace': float(A.diagonal().sum())}",
            array_refs={"A": "A"},
        )

        self.assertEqual(ret["status"], "ok")
        payload = json.loads(ret["stdout"])
        self.assertEqual(payload["shape"], [2, 2])
        self.assertEqual(payload["nnz"], 4)
        self.assertAlmostEqual(payload["trace"], 7.0)
        self.assertEqual(ret["workspace_refs_used"]["array_refs"], {"A": "A"})

    def test_workspace_python_snippet_blocks_file_access(self):
        _load_matrix_csc_content("2 2\n0 2 4\n0 1 0 1\n4.0 1.0 1.0 3.0\n")

        ret = _run_python_workspace_snippet(
            "from scipy.io import mmread\nRESULT = mmread('494_bus.mtx.gz')",
            array_refs={"A": "A"},
        )

        self.assertEqual(ret["status"], "error")
        self.assertEqual(ret["error_type"], "workspace_python_safety_error")
        self.assertIn("next_allowed_actions", ret)

    def test_frontend_context_does_not_emit_server_path(self):
        app_tsx = Path(__file__).resolve().parents[1] / "frontend" / "src" / "App.tsx"
        source = app_tsx.read_text(encoding="utf-8")

        self.assertNotIn("server_path:", source)
        self.assertNotIn("只有存在 server_path", source)

    def test_large_solver_result_is_saved_as_handle(self):
        n = 70
        A = np.eye(n).tolist()
        b = np.arange(n, dtype=float).tolist()

        ret = _solve_linear_lapack(A_rows=A, b=b)

        self.assertEqual(ret["status"], "ok")
        self.assertNotIn("x", ret)
        self.assertEqual(ret["x_ref"], "x")
        self.assertTrue(ret["x_handle"]["object_handle"])

        x_ret = workspace_get("x")
        self.assertEqual(x_ret["status"], "ok")
        self.assertTrue(x_ret["value"]["object_handle"])
        self.assertEqual(x_ret["value"]["shape"], [n])

    def test_existing_workspace_matrix_rejects_large_inline_rows(self):
        n = 9
        A = np.eye(n).tolist()
        b = np.ones(n).tolist()
        _workspace_store.set_var("A", A)

        blocked = _solve_linear_lapack(A_rows=A, b=b)
        ok_ret = _solve_linear_lapack(A_ref="A", b=b)

        self.assertEqual(blocked["status"], "error")
        self.assertEqual(blocked["error_type"], "workspace_protocol_violation")
        self.assertTrue(blocked["fallback_required"])
        self.assertIn("next_allowed_actions", blocked)
        self.assertEqual(ok_ret["status"], "ok")
        np.testing.assert_allclose(ok_ret["x"], b, atol=1e-10)

    def test_tool_error_includes_fallback_fields(self):
        ret = _solve_linear_lapack()

        self.assertEqual(ret["status"], "error")
        self.assertEqual(ret["error_type"], "tool_error")
        self.assertTrue(ret["fallback_required"])
        self.assertIn("next_allowed_actions", ret)

    def test_workspace_slice_tool_is_limited(self):
        _workspace_store.set_var("A", np.arange(400, dtype=float).reshape(20, 20).tolist())

        ok_ret = workspace_slice("A", rows=[0, 1], cols=[0, 1])
        blocked_ret = workspace_slice("A", rows=list(range(11)), cols=list(range(10)))

        self.assertEqual(ok_ret["status"], "ok")
        self.assertEqual(ok_ret["data"], [[0.0, 1.0], [20.0, 21.0]])
        self.assertEqual(blocked_ret["status"], "error")

    def test_workspace_audit_records_accesses(self):
        _workspace_store.set_var("A", [[1.0, 0.0], [0.0, 1.0]])
        workspace_get("A")

        audit = workspace_audit(limit=5)

        self.assertEqual(audit["status"], "ok")
        self.assertGreaterEqual(audit["count"], 2)


if __name__ == "__main__":
    unittest.main()
