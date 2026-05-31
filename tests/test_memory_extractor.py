from __future__ import annotations

import unittest

from memory.extractor import extract_memory_from_dialogue


class TestMemoryExtractor(unittest.TestCase):
    def test_extract_memory_from_dialogue(self):
        user_q = "稀疏SPD线性系统怎么解？最好有预条件，并说明收敛判据。"
        assistant_a = "可使用预条件共轭梯度法（PCG），并结合残差与条件数评估稳定性。"
        item = extract_memory_from_dialogue(user_q, assistant_a)
        self.assertIn("problem_pattern", item)
        self.assertIn("solution_pattern", item)
        self.assertGreater(item["quality_score"], 0.0)
        self.assertTrue(item["embedding_text"])
        self.assertIn("|", item["problem_pattern"])
        self.assertIn("pcg", item["solution_pattern"].lower())
        self.assertTrue(item.get("source_meta", {}).get("answer_ingested"))
        self.assertEqual(item.get("source_meta", {}).get("memory_granularity"), "answer_experience")
        self.assertIn(item.get("memory_type"), {"episodic_solution", "semantic_solution"})
        self.assertGreater(item.get("confidence", 0.0), 0.0)
        self.assertTrue(item.get("evidence"))

    def test_extract_jacobi_svd_lapack_api_usage(self):
        user_q = "要求使用Jacobi svd方法, Jacobi svd可以通过scipy.linalg.lapack调用"
        assistant_a = """
        正确做法是调用 SciPy 暴露的 LAPACK gejsv driver，而不是按 numpy.linalg.svd 的返回值写。
        ```python
        from scipy.linalg import lapack
        A_f = np.asfortranarray(A, dtype=np.float64)
        gejsv = lapack.get_lapack_funcs("gejsv", (A_f,))
        s, u, v, workout, iworkout, info = gejsv(A_f, jobu=1, jobv=1)
        ```
        """
        item = extract_memory_from_dialogue(user_q, assistant_a)

        self.assertEqual(item.get("source_meta", {}).get("experience_type"), "api_usage")
        self.assertEqual(item.get("source_meta", {}).get("lapack_driver"), "gejsv")
        self.assertTrue(item.get("source_meta", {}).get("answer_ingested"))
        self.assertIn("gejsv", item.get("code_hint", "").lower())
        self.assertIn("返回", item.get("method_reason", ""))


if __name__ == "__main__":
    unittest.main()
