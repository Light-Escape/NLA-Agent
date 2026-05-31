from __future__ import annotations

import unittest

from workspace import WorkspaceStore


class TestWorkspaceStore(unittest.TestCase):
    def setUp(self):
        self.ws = WorkspaceStore()

    def test_who_and_whos(self):
        self.ws.set_var("A", [[1, 2], [3, 4]])
        self.ws.set_var("b", [1, 0])

        who_ret = self.ws.list_vars(detail=False)
        self.assertEqual(who_ret["status"], "ok")
        self.assertEqual(who_ret["variables"], ["A", "b"])

        whos_ret = self.ws.list_vars(detail=True)
        self.assertEqual(whos_ret["status"], "ok")
        self.assertEqual(whos_ret["variables"][0]["name"], "A")
        self.assertEqual(whos_ret["variables"][0]["shape"], [2, 2])

    def test_clear_and_clear_var_name(self):
        self.ws.set_var("A", [[1, 2], [3, 4]])
        self.ws.set_var("x", [1, 1])
        self.ws.clear(name="x")

        who_ret = self.ws.list_vars(detail=False)
        self.assertEqual(who_ret["variables"], ["A"])

        self.ws.clear()
        who_ret_after_clear = self.ws.list_vars(detail=False)
        self.assertEqual(who_ret_after_clear["variables"], [])

    def test_ans_auto_write(self):
        self.ws.set_var("A", [[2, 0], [0, 3]])
        self.ws.write_ans({"status": "ok", "x": [0.5, 1.0]})

        ans_ret = self.ws.get_var("ans")
        self.assertEqual(ans_ret["status"], "ok")
        self.assertEqual(ans_ret["value"]["x"], [0.5, 1.0])

    def test_matrix_get_returns_handle_not_elements(self):
        self.ws.set_var("A", [[1.0, 2.0], [3.0, 4.0]])

        ret = self.ws.get_var("A")

        self.assertEqual(ret["status"], "ok")
        self.assertEqual(ret["value"]["matrix_handle"], True)
        self.assertEqual(ret["value"]["ref"], "A")
        self.assertEqual(ret["value"]["shape"], [2, 2])
        self.assertNotIn([[1.0, 2.0], [3.0, 4.0]], ret.values())
        self.assertEqual(self.ws.get_matrix("A").shape, (2, 2))

    def test_workspace_protocol_metadata(self):
        self.ws.set_var(
            "A",
            [[1.0, 0.0], [0.0, 2.0]],
            source="test",
            role="system_matrix",
            origin="unit_test",
            created_by_tool="test_loader",
        )

        detail = self.ws.list_vars(detail=True)["variables"][0]

        self.assertEqual(detail["kind"], "matrix")
        self.assertEqual(detail["storage_type"], "dense")
        self.assertEqual(detail["role"], "system_matrix")
        self.assertEqual(detail["origin"], "unit_test")
        self.assertEqual(detail["created_by_tool"], "test_loader")
        self.assertEqual(detail["version"], 1)
        self.assertIn("fingerprint", detail)
        self.assertIn("summary", detail)
        self.assertEqual(detail["nnz"], 2)
        self.assertEqual(detail["density"], 0.5)

    def test_allocate_ref_and_alias_do_not_copy_elements(self):
        self.ws.set_var("A", [[1.0, 0.0], [0.0, 1.0]], role="system_matrix")

        next_ref = self.ws.allocate_ref("A")
        alias_ret = self.ws.alias("K", "A")
        role_ret = self.ws.bind_role("preconditioner", "K")

        self.assertEqual(next_ref, "A2")
        self.assertEqual(alias_ret["status"], "ok")
        self.assertEqual(alias_ret["variable"]["alias_of"], "A")
        self.assertEqual(role_ret["status"], "ok")
        self.assertEqual(role_ret["variable"]["role"], "preconditioner")
        self.assertEqual(self.ws.get_matrix("K").shape, (2, 2))

    def test_user_set_rejects_large_objects(self):
        ret = self.ws.set_user_var("A", [[float(i * 9 + j) for j in range(9)] for i in range(9)])

        self.assertEqual(ret["status"], "error")
        self.assertEqual(ret["error_type"], "workspace_protocol_violation")
        self.assertTrue(ret["fallback_required"])

    def test_controlled_reads_and_audit(self):
        self.ws.set_var("A", [[1.0, 2.0], [2.0, 5.0]])

        summary = self.ws.summary("A")
        stats = self.ws.stats("A")
        structure = self.ws.structure("A")
        slice_ret = self.ws.read_slice("A", rows=[0], cols=[1])
        audit = self.ws.audit(limit=10)

        self.assertEqual(summary["status"], "ok")
        self.assertTrue(summary["handle"]["object_handle"])
        self.assertEqual(stats["status"], "ok")
        self.assertGreater(stats["fro_norm"], 0)
        self.assertTrue(structure["is_symmetric"])
        self.assertEqual(slice_ret["data"], [[2.0]])
        self.assertGreaterEqual(audit["count"], 4)

    def test_slice_limit(self):
        self.ws.set_var("A", [[float(i * 20 + j) for j in range(20)] for i in range(20)])

        ret = self.ws.read_slice("A", rows=list(range(11)), cols=list(range(10)))

        self.assertEqual(ret["status"], "error")
        self.assertEqual(ret["error_type"], "workspace_protocol_violation")
        self.assertIn("最多返回", ret["message"])

    def test_cumulative_slice_budget_blocks_matrix_reconstruction(self):
        self.ws.set_var("A", [[float(i * 20 + j) for j in range(20)] for i in range(20)])

        first = self.ws.read_slice("A", rows=list(range(10)), cols=list(range(10)))
        second = self.ws.read_slice("A", rows=list(range(10, 20)), cols=list(range(10)))
        audit = self.ws.audit(limit=10)

        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "error")
        self.assertEqual(second["error_type"], "workspace_protocol_violation")
        self.assertTrue(any(record["action"] == "slice_budget_exceeded" for record in audit["records"]))
        self.assertTrue(any(record["action"] == "protocol_violation" for record in audit["records"]))

    def test_large_vector_returns_handle(self):
        self.ws.set_var("x", list(range(100)))

        ret = self.ws.get_var("x")

        self.assertTrue(ret["value"]["object_handle"])
        self.assertEqual(ret["value"]["kind"], "vector")
        self.assertNotIn("0, 1, 2", str(ret["value"]))


if __name__ == "__main__":
    unittest.main()
