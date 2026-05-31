from __future__ import annotations

import tempfile
import unittest

from memory.config import MemoryConfig
from memory.agent_integration import MemoryAgentBridge
from memory.store import NLAMemoryStore


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = MemoryConfig(
            persist_dir=self.tmp.name,
            auto_write_mode=True,
            readonly_mode=False,
            min_quality_score=0.1,
        )
        self.store = NLAMemoryStore(config=cfg)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _sample_item(self):
        return {
            "problem_pattern": "稀疏对称正定系统求解",
            "math_topic": ["Krylov"],
            "matrix_properties": ["稀疏", "SPD"],
            "solution_pattern": "PCG + 预条件",
            "method_reason": "利用 SPD 性质提升效率",
            "failure_modes": ["非SPD失效"],
            "assumptions": ["矩阵可用稀疏存储"],
            "complexity_hint": "O(nnz*iter)",
            "code_hint": "",
            "source_meta": {"human_confirmed": True},
            "quality_score": 0.9,
        }

    def _issue_resolution_item(self):
        return {
            "problem_pattern": "运行时出现 ValueError 报错，矩阵维度不一致",
            "math_topic": ["通用数值线性代数"],
            "matrix_properties": ["稀疏"],
            "solution_pattern": "定位维度来源并修复输入，重新对齐 shape 后问题已解决",
            "method_reason": "先定位根因再修复数据路径，避免盲目替换算法。",
            "failure_modes": ["维度不一致导致计算失败"],
            "assumptions": ["输入数据来源稳定"],
            "complexity_hint": "以排查与校验为主，计算复杂度影响较小。",
            "code_hint": "",
            "source_meta": {
                "human_confirmed": False,
                "experience_type": "issue_resolution",
                "priority": "high",
            },
            "quality_score": 0.2,
        }

    def test_add_and_search(self):
        add_ret = self.store.add_memory(self._sample_item(), require_confirmation=False)
        self.assertIn(add_ret["status"], ("stored", "deduplicated"))
        ret = self.store.search_memory("如何解稀疏SPD线性系统", top_k=3)
        self.assertEqual(ret["status"], "ok")
        self.assertGreaterEqual(len(ret["items"]), 1)
        self.assertIn("score", ret["items"][0])

        mem = self.store.get_memory(add_ret["memory_id"])
        self.assertEqual(mem["status"], "ok")
        self.assertGreaterEqual(mem["item"].get("use_count", 0), 1)

    def test_update_and_delete(self):
        add_ret = self.store.add_memory(self._sample_item(), require_confirmation=False)
        memory_id = add_ret["memory_id"]
        updated = self._sample_item()
        updated["solution_pattern"] = "改用IC(0)+PCG"
        up_ret = self.store.update_memory(memory_id, updated)
        self.assertEqual(up_ret["status"], "updated")
        del_ret = self.store.delete_memory(memory_id)
        self.assertEqual(del_ret["status"], "deleted")

    def test_summarize(self):
        self.store.add_memory(self._sample_item(), require_confirmation=False)
        q = self.store.search_memory("稀疏SPD", top_k=2)
        summary = self.store.summarize_memory(q)
        self.assertIn("可复用历史解法", summary)

    def test_issue_resolution_is_prioritized_on_store(self):
        ret = self.store.add_memory(self._issue_resolution_item(), require_confirmation=True)
        self.assertEqual(ret["status"], "stored")
        mem_id = ret["memory_id"]
        active = self.store.collection.get(ids=[mem_id])
        pending = self.store.pending_collection.get(ids=[mem_id])
        self.assertTrue(active["ids"])
        self.assertFalse(pending["ids"])

    def test_pending_memory_can_be_listed_and_approved(self):
        cfg = MemoryConfig(
            persist_dir=self.tmp.name,
            auto_write_mode=False,
            readonly_mode=False,
            min_quality_score=0.1,
        )
        store = NLAMemoryStore(config=cfg)
        try:
            ret = store.add_memory(self._sample_item(), require_confirmation=True)
            self.assertEqual(ret["status"], "pending")

            pending = store.list_pending_memories()
            self.assertEqual(pending["total"], 1)
            mem_id = pending["items"][0]["id"]
            self.assertEqual(mem_id, ret["memory_id"])

            approved = store.approve_pending_memory(mem_id)
            self.assertEqual(approved["status"], "approved")
            recall = store.search_memory("稀疏SPD系统", top_k=1)
            self.assertGreaterEqual(len(recall["items"]), 1)
        finally:
            store.close()

    def test_issue_resolution_is_prioritized_on_recall(self):
        general = self._sample_item()
        general["problem_pattern"] = "运行时出现错误，求解流程说明"
        general["solution_pattern"] = "给出标准求解步骤"
        general["source_meta"] = {"human_confirmed": True, "experience_type": "general_solution"}
        self.store.add_memory(general, require_confirmation=False)
        self.store.add_memory(self._issue_resolution_item(), require_confirmation=False)

        ret = self.store.search_memory("这个报错怎么修复，运行时 error 了", top_k=1)
        self.assertEqual(ret["status"], "ok")
        self.assertGreaterEqual(len(ret["items"]), 1)
        top = ret["items"][0]["item"]
        self.assertEqual(top.get("source_meta", {}).get("experience_type"), "issue_resolution")

    def test_ingest_execution_issue_and_recall(self):
        ret = self.store.ingest_execution_issue(
            tool_name="run_python_snippet",
            error_text="run_python_snippet status=error stderr=UserWarning: Glyph 32570 missing from current font",
            fix_text="在绘图前设置 plt.rcParams['font.sans-serif']=['SimHei']，并设置 axes.unicode_minus=False。",
            context_text="用户要求 matplotlib 图中支持中文显示",
            require_confirmation=False,
        )
        self.assertIn(ret["status"], ("stored", "deduplicated"))
        recall = self.store.search_memory("代码执行时中文显示报错怎么修复", top_k=1)
        self.assertEqual(recall["status"], "ok")
        self.assertGreaterEqual(len(recall["items"]), 1)
        top = recall["items"][0]["item"]
        self.assertEqual(top.get("source_meta", {}).get("experience_type"), "execution_issue")
        self.assertIn("simhei", top.get("code_hint", "").lower())

    def test_jacobi_svd_api_usage_is_stored_and_recalled(self):
        user_q = "要求使用Jacobi svd方法, Jacobi svd可以通过scipy.linalg.lapack调用"
        assistant_a = """
        正确调用是 SciPy LAPACK 的 gejsv 系列：
        ```python
        from scipy.linalg import lapack
        A_f = np.asfortranarray(A, dtype=np.float64)
        gejsv = lapack.get_lapack_funcs("gejsv", (A_f,))
        s, u, v, workout, iworkout, info = gejsv(A_f, jobu=1, jobv=1)
        ```
        """

        ret = self.store.ingest_dialogue(user_q, assistant_a, require_confirmation=True)
        self.assertEqual(ret["status"], "stored")

        recall = self.store.search_memory(
            "Jacobi SVD 用 scipy.linalg.lapack 应该怎么调用？",
            top_k=1,
        )
        self.assertEqual(recall["status"], "ok")
        self.assertGreaterEqual(len(recall["items"]), 1)
        top = recall["items"][0]["item"]
        self.assertEqual(top.get("source_meta", {}).get("experience_type"), "api_usage")
        self.assertIn("gejsv", top.get("code_hint", "").lower())

        summary = self.store.summarize_memory(recall)
        self.assertIn("调用提示", summary)
        self.assertIn("gejsv", summary.lower())

    def test_jacobi_svd_static_hint_when_memory_is_empty(self):
        bridge = MemoryAgentBridge(self.store)
        context = bridge.recall_context(
            "要求使用Jacobi svd方法, Jacobi svd可以通过scipy.linalg.lapack调用",
            top_k=1,
        )
        self.assertIn("高风险 API 记忆", context)
        self.assertIn("gejsv", context.lower())
        self.assertIn("info == 0", context)

    def test_injection_gate_and_strip_injected_memory(self):
        bridge = MemoryAgentBridge(self.store)
        no_context = bridge.inject_memory_context("今天聊点和矩阵无关的日常问题", top_k=1)
        self.assertNotIn("[历史记忆召回]", no_context)

        self.store.add_memory(self._sample_item(), require_confirmation=False)
        injected = bridge.inject_memory_context("稀疏SPD线性系统怎么解", top_k=1)
        self.assertIn("[历史记忆召回]", injected)
        stripped = bridge.strip_injected_memory(injected)
        self.assertEqual(stripped, "稀疏SPD线性系统怎么解")

    def test_hybrid_retrieval_handles_chinese_short_query(self):
        item = self._sample_item()
        item["problem_pattern"] = "大规模稀疏对称正定方程组"
        item["solution_pattern"] = "使用 PCG 结合不完全 Cholesky 预条件"
        self.store.add_memory(item, require_confirmation=False)

        recall = self.store.search_memory("对称正定大稀疏方程用什么迭代法", top_k=1)
        self.assertEqual(recall["status"], "ok")
        self.assertGreaterEqual(len(recall["items"]), 1)
        self.assertIn("pcg", recall["items"][0]["item"]["solution_pattern"].lower())


if __name__ == "__main__":
    unittest.main()
