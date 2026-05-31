from __future__ import annotations

import unittest

try:
    from NLA_Master.agent_supervisor import supervise_step
except ModuleNotFoundError:
    from agent_supervisor import supervise_step


class TestAgentSupervisorP0(unittest.TestCase):
    def test_phase_sequence_rejects_skip_without_evidence(self):
        ret = supervise_step(
            user_goal="求解线性方程组 Ax=b",
            problem_type="solve_linear_system",
            current_step="直接进入规划",
            proposed_action="准备选择算法",
            phase="plan",
            verified_facts={},
        )
        self.assertEqual(ret["verdict"], "reject")
        self.assertTrue(any("阶段跳步" in i.get("message", "") for i in ret.get("issues", [])))

    def test_phase_sequence_accepts_canonical_verified_facts(self):
        ret = supervise_step(
            user_goal="求解线性方程组 Ax=b",
            problem_type="solve_linear_system",
            current_step="规划下一步",
            proposed_action="整理可选算法，不执行数值代码",
            phase="plan",
            verified_facts={
                "schema_version": "v1",
                "goal_clarified": True,
                "preference_confirmed": True,
                "preconditions_checked": True,
                "properties_checked": True,
                "algorithm_planned": False,
                "completed_phases": ["clarify_goal", "preconditions", "property_check"],
            },
        )
        self.assertNotEqual(ret["verdict"], "reject")
        self.assertFalse(any("阶段跳步" in i.get("message", "") for i in ret.get("issues", [])))

    def test_interactive_gates_prioritize_verified_facts(self):
        ret = supervise_step(
            user_goal="给我数值验证",
            problem_type="solve_linear_system",
            current_step="执行阶段",
            proposed_action="调用执行工具进行验证",
            tool_call={"name": "run_python_snippet", "args": {"code": "print(1)"}},
            phase="solve",
            verified_facts={
                "goal_clarified": True,
                "preference_confirmed": True,
                "preconditions_checked": True,
                "properties_checked": True,
                "algorithm_planned": True,
                "completed_phases": [
                    "clarify_goal",
                    "preconditions",
                    "property_check",
                    "plan",
                ],
            },
        )
        messages = " | ".join(i.get("message", "") for i in ret.get("issues", []))
        self.assertNotIn("偏好确认", messages)
        self.assertNotIn("必要条件确认", messages)
        self.assertNotIn("矩阵关键性质确认", messages)

    def test_interactive_gates_fallback_to_text_when_facts_absent(self):
        ret = supervise_step(
            user_goal="给我数值验证",
            problem_type="solve_linear_system",
            current_step="执行阶段",
            proposed_action="调用执行工具进行验证",
            tool_call={"name": "run_python_snippet", "args": {"code": "print(1)"}},
            phase="solve",
            verified_facts={
                "completed_phases": [
                    "clarify_goal",
                    "preconditions",
                    "property_check",
                    "plan",
                ]
            },
        )
        messages = " | ".join(i.get("message", "") for i in ret.get("issues", []))
        self.assertIn("偏好确认", messages)

    def test_alias_fields_are_normalized(self):
        ret = supervise_step(
            user_goal="对称矩阵特征值分析",
            problem_type="eigen",
            current_step="进入求解",
            proposed_action="执行数值方法",
            tool_call={"name": "run_python_snippet", "args": {"code": "print(1)"}},
            phase="solve",
            verified_facts={
                "solution_goal_confirmed": True,
                "precondition_checklist_ready": True,
                "matrix_info_confirmed": True,
                "algorithm_selected": True,
                "completed_phases": ["clarify_goal"],
            },
        )
        self.assertFalse(any("阶段跳步" in i.get("message", "") for i in ret.get("issues", [])))

    def test_lapack_tool_call_is_blocked_before_solve_phase(self):
        ret = supervise_step(
            user_goal="求解线性方程组 Ax=b",
            problem_type="solve_linear_system",
            current_step="前置检查阶段",
            proposed_action="直接调用 LAPACK 求解",
            tool_call={"name": "solve_linear_lapack", "args": {"A_rows": [[1.0]], "b": [1.0]}},
            phase="preconditions",
            verified_facts={"completed_phases": ["clarify_goal"]},
        )
        self.assertEqual(ret["verdict"], "reject")
        self.assertTrue(any("调用过早" in i.get("message", "") for i in ret.get("issues", [])))

    def test_lapack_tool_call_is_treated_as_algorithm_step(self):
        ret = supervise_step(
            user_goal="给我数值验证",
            problem_type="solve_linear_system",
            current_step="执行阶段",
            proposed_action="调用 LAPACK 求解",
            tool_call={"name": "solve_linear_lapack", "args": {"A_rows": [[2.0]], "b": [4.0]}},
            phase="solve",
            verified_facts={
                "completed_phases": [
                    "clarify_goal",
                    "preconditions",
                    "property_check",
                    "plan",
                ]
            },
        )
        messages = " | ".join(i.get("message", "") for i in ret.get("issues", []))
        self.assertIn("偏好确认", messages)


if __name__ == "__main__":
    unittest.main()
