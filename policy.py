"""
任务路由、矩阵性质分析与算法选择策略模块。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from .parsers import csc_payload_to_matrix, matrix_to_dense
except ImportError:
    from parsers import csc_payload_to_matrix, matrix_to_dense


def route_user_task(user_query: str) -> dict:
    """基于关键词的轻量任务路由，让调用链更显式。"""
    q = (user_query or "").lower()
    if not q.strip():
        return {"status": "error", "message": "user_query 不能为空"}

    if any(k in q for k in ("特征值", "eigen", "特征向量")):
        task = "eigen"
    elif any(k in q for k in ("最小二乘", "least squares", "拟合")):
        task = "least_squares"
    elif any(k in q for k in ("行列式", "determinant", "det(")):
        task = "determinant"
    elif any(k in q for k in ("逆矩阵", "矩阵求逆", "inverse")):
        task = "inverse"
    elif any(k in q for k in ("线性方程", "ax=b", "求解", "solve")):
        task = "solve_linear_system"
    else:
        task = "general_nla"

    if any(k in q for k in ("理解", "一步步", "逐步", "引导", "教练", "思路", "为什么", "原理")):
        learning_intent = "understand"
    elif any(k in q for k in ("直接给答案", "直接答案", "只要结果", "快点", "最终答案")):
        learning_intent = "just_answer"
    else:
        learning_intent = "unspecified"

    if any(k in q for k in ("尽快", "马上", "快速", "快点")):
        urgency = "quick"
    else:
        urgency = "normal"

    needs_interaction = learning_intent != "just_answer"

    return {
        "status": "ok",
        "task_type": task,
        "learning_intent": learning_intent,
        "urgency": urgency,
        "needs_interaction": needs_interaction,
        "message": "已完成任务路由，请结合矩阵性质再选算法。",
    }


def infer_solution_preference(user_query: str) -> dict:
    """
    从用户输入中推断“精确/近似”和“教练/直接答案”偏好。
    """
    q = (user_query or "").lower().strip()
    if not q:
        return {"status": "error", "message": "user_query 不能为空"}

    exact_hits = sum(
        1
        for k in ("精确", "解析", "严格", "闭式", "exact", "symbolic")
        if k in q
    )
    approx_hits = sum(
        1
        for k in ("近似", "数值", "迭代", "估计", "容忍误差", "approx")
        if k in q
    )
    coach_hits = sum(
        1
        for k in ("理解", "一步步", "逐步", "引导", "教练", "为什么", "原理", "思路")
        if k in q
    )
    direct_hits = sum(
        1
        for k in ("直接给答案", "直接答案", "只要结果", "快点", "最终答案")
        if k in q
    )

    if exact_hits > 0 and approx_hits == 0:
        solution_goal = "exact"
    elif approx_hits > 0 and exact_hits == 0:
        solution_goal = "approx"
    else:
        solution_goal = "unspecified"

    if coach_hits > 0 and direct_hits == 0:
        interaction_mode = "coach"
    elif direct_hits > 0 and coach_hits == 0:
        interaction_mode = "direct"
    else:
        interaction_mode = "unspecified"

    confidence = "low"
    if solution_goal != "unspecified" and interaction_mode != "unspecified":
        confidence = "high"
    elif solution_goal != "unspecified" or interaction_mode != "unspecified":
        confidence = "medium"

    followup_questions: list[str] = []
    if solution_goal == "unspecified":
        followup_questions.append("你更希望先求精确解，还是先给可控误差的近似解？")
    if interaction_mode == "unspecified":
        followup_questions.append("你希望我直接给答案，还是进入逐步引导的“数值代数教练”模式？")

    return {
        "status": "ok",
        "solution_goal": solution_goal,
        "interaction_mode": interaction_mode,
        "confidence": confidence,
        "needs_clarification": bool(followup_questions),
        "followup_questions": followup_questions,
    }


def build_precondition_checklist(task_type: str, matrix_info: Optional[dict] = None) -> dict:
    """
    为当前任务生成必要条件清单，便于在求解前确认前提。
    """
    task = (task_type or "").strip().lower()
    info = matrix_info if isinstance(matrix_info, dict) else {}
    shape = info.get("shape")
    is_square = info.get("is_square")
    rank_estimate = info.get("rank_estimate")
    cond2 = info.get("condition_number_2")
    is_symmetric = info.get("is_symmetric")
    is_spd = info.get("is_positive_definite")

    checklist: list[dict] = []

    def _known(value) -> bool:
        return value is not None

    if task == "solve_linear_system":
        checklist = [
            {"item": "A 与 b 维度匹配（A.shape[0] == len(b)）", "status": "unknown"},
            {"item": "若追求精确唯一解，A 应为方阵", "status": "ok" if is_square is True else ("warning" if is_square is False else "unknown")},
            {"item": "A 建议满秩（避免不可解或多解）", "status": "ok" if (_known(rank_estimate) and _known(shape) and rank_estimate == min(shape)) else ("warning" if _known(rank_estimate) and _known(shape) else "unknown")},
            {"item": "条件数不过大（降低数值不稳定风险）", "status": "warning" if (_known(cond2) and isinstance(cond2, (int, float)) and cond2 > 1e12) else ("ok" if _known(cond2) else "unknown")},
        ]
    elif task == "eigen":
        checklist = [
            {"item": "特征值问题需要方阵", "status": "ok" if is_square is True else ("warning" if is_square is False else "unknown")},
            {"item": "若矩阵对称，可优先用更稳定高效算法（eigh/eigsh）", "status": "ok" if is_symmetric is True else ("unknown" if is_symmetric is None else "warning")},
            {"item": "若需实特征结构，需确认输入满足对应条件", "status": "unknown"},
        ]
    elif task == "least_squares":
        checklist = [
            {"item": "A 与 b 维度匹配（A.shape[0] == len(b)）", "status": "unknown"},
            {"item": "检查秩亏风险（rank(A) < n）", "status": "warning" if (_known(rank_estimate) and _known(shape) and rank_estimate < shape[1]) else ("ok" if _known(rank_estimate) and _known(shape) else "unknown")},
            {"item": "若病态，考虑正则化或缩放", "status": "warning" if (_known(cond2) and isinstance(cond2, (int, float)) and cond2 > 1e8) else ("ok" if _known(cond2) else "unknown")},
        ]
    elif task == "inverse":
        checklist = [
            {"item": "矩阵求逆需要方阵", "status": "ok" if is_square is True else ("warning" if is_square is False else "unknown")},
            {"item": "矩阵应可逆（det != 0 / 满秩）", "status": "ok" if (_known(rank_estimate) and _known(shape) and is_square is True and rank_estimate == shape[0]) else ("warning" if _known(rank_estimate) and _known(shape) else "unknown")},
            {"item": "条件数不过大", "status": "warning" if (_known(cond2) and isinstance(cond2, (int, float)) and cond2 > 1e12) else ("ok" if _known(cond2) else "unknown")},
        ]
    elif task == "determinant":
        checklist = [
            {"item": "行列式只对方阵有定义", "status": "ok" if is_square is True else ("warning" if is_square is False else "unknown")},
            {"item": "建议用 slogdet 提升稳定性", "status": "ok"},
        ]
    else:
        checklist = [{"item": "请先明确任务类型，再确认必要条件。", "status": "unknown"}]

    missing_items = [c["item"] for c in checklist if c["status"] == "unknown"]
    risk_items = [c["item"] for c in checklist if c["status"] == "warning"]

    return {
        "status": "ok",
        "task_type": task,
        "checklist": checklist,
        "missing_items": missing_items,
        "risk_items": risk_items,
        "need_user_confirmation": bool(missing_items or risk_items),
        "matrix_snapshot": {
            "shape": shape,
            "is_square": is_square,
            "rank_estimate": rank_estimate,
            "condition_number_2": cond2,
            "is_symmetric": is_symmetric,
            "is_positive_definite": is_spd,
        },
    }


def plan_coach_next_step(
    user_query: str,
    task_type: str,
    coach_state: Optional[dict] = None,
    matrix_info: Optional[dict] = None,
    preference: Optional[dict] = None,
) -> dict:
    """
    教练模式推进器（自适应）：
    - 不再强制线性阶段顺序。
    - 基于“缺失信息”决定下一步，避免机械补全。
    """
    state = coach_state if isinstance(coach_state, dict) else {}
    stage = (state.get("stage") or "clarify_goal").strip().lower()
    pref = preference if isinstance(preference, dict) else infer_solution_preference(user_query=user_query)
    preconditions = build_precondition_checklist(task_type=task_type, matrix_info=matrix_info)
    info = matrix_info if isinstance(matrix_info, dict) else {}

    next_stage = "plan"
    prompt = "已具备核心信息，可先给出方案再按需求解。"
    questions: list[str] = []
    missing_capabilities: list[str] = []

    # 1) 目标/交互偏好是否明确
    preference_ready = not bool(pref.get("needs_clarification", True))
    if not preference_ready:
        missing_capabilities.append("preference")
        questions.extend(pref.get("followup_questions", []))

    # 2) 必要条件是否满足
    preconditions_ready = not bool(preconditions.get("need_user_confirmation", False))
    if not preconditions_ready:
        missing_capabilities.append("preconditions")
        questions.extend(preconditions.get("missing_items", []))
        # 风险项只取前两项，避免过度盘问拖慢流程
        questions.extend(preconditions.get("risk_items", [])[:2])

    # 3) 关键矩阵性质证据是否足够
    known_props = 0
    if isinstance(info.get("is_square"), bool):
        known_props += 1
    if isinstance(info.get("is_symmetric"), bool):
        known_props += 1
    if isinstance(info.get("rank_estimate"), int):
        known_props += 1
    if isinstance(info.get("condition_number_2"), (int, float)):
        known_props += 1

    has_matrix_shape = isinstance(info.get("shape"), (list, tuple)) and len(info.get("shape")) == 2
    properties_ready = known_props >= 2 or not has_matrix_shape
    if not properties_ready:
        missing_capabilities.append("properties")
        if not isinstance(info.get("is_square"), bool):
            questions.append("矩阵是否方阵？")
        if not isinstance(info.get("rank_estimate"), int):
            questions.append("矩阵秩是否已知（是否可能秩亏）？")
        if not isinstance(info.get("condition_number_2"), (int, float)):
            questions.append("条件数是否可估计（是否病态）？")

    # 4) 快速路径：用户明确“直接结果/快速求解”时，允许在前提满足后直达 solve
    q = (user_query or "").lower()
    wants_fast = any(k in q for k in ("直接给答案", "只要结果", "快点", "尽快", "马上", "直接算"))
    interaction_mode = (pref.get("interaction_mode") or "").strip().lower()
    can_fast_track = (
        preference_ready
        and preconditions_ready
        and (properties_ready or not has_matrix_shape)
        and (wants_fast or interaction_mode == "direct")
    )

    if can_fast_track:
        next_stage = "solve"
        prompt = "信息已充分，进入快速求解与结果验证。"
    elif not preference_ready:
        next_stage = "clarify_goal"
        prompt = "先确认求解偏好，避免后续重复沟通。"
    elif not preconditions_ready:
        next_stage = "preconditions"
        prompt = "先补齐必要条件，避免盲目选算法。"
    elif not properties_ready:
        next_stage = "property_check"
        prompt = "先确认关键矩阵性质，再给出更稳健方案。"
    elif stage == "solve":
        next_stage = "solve"
        prompt = "继续求解阶段，聚焦收敛与结果校验。"
    else:
        next_stage = "plan"
        prompt = "先输出简洁方案，必要时再执行求解。"

    return {
        "status": "ok",
        "current_stage": stage,
        "next_stage": next_stage,
        "coach_prompt": prompt,
        "questions_for_user": list(dict.fromkeys(questions)),
        "ready_for_algorithm_selection": next_stage in {"plan", "solve"},
        "fast_track": can_fast_track,
        "missing_capabilities": missing_capabilities,
        "preference": pref,
        "preconditions": preconditions,
    }


def analyze_matrix_properties(
    A_rows: Optional[list[list[float]]] = None,
    A_csc: Optional[dict] = None,
    tol: float = 1e-10,
) -> dict:
    """
    分析矩阵性质（稀疏性、对称性、条件数等），供算法选择使用。
    """
    try:
        if A_rows is None and A_csc is None:
            return {"status": "error", "message": "A_rows 与 A_csc 至少提供一个"}

        sparse_shape = None
        sparse_nnz = None
        if A_csc is not None:
            csc = csc_payload_to_matrix(A_csc)
            sparse_shape = [int(csc.shape[0]), int(csc.shape[1])]
            sparse_nnz = int(csc.nnz)

        A = matrix_to_dense(A_rows=A_rows, A_csc=A_csc)
        m, n = map(int, A.shape)
        square = m == n
        fro = float(np.linalg.norm(A, ord="fro"))
        symmetric = bool(square and np.allclose(A, A.T, atol=tol, rtol=0.0))

        result = {
            "status": "ok",
            "shape": [m, n],
            "is_square": square,
            "is_symmetric": symmetric,
            "frobenius_norm": fro,
            "rank_estimate": int(np.linalg.matrix_rank(A)),
        }

        if sparse_shape is not None and sparse_nnz is not None:
            result["is_sparse_input"] = True
            result["nnz"] = sparse_nnz
            result["density"] = float(sparse_nnz / (sparse_shape[0] * sparse_shape[1]))
        else:
            result["is_sparse_input"] = False
            result["density"] = float(np.count_nonzero(A) / (m * n)) if m * n else 0.0

        if square:
            try:
                eigvals = np.linalg.eigvalsh(A) if symmetric else np.linalg.eigvals(A)
                result["min_abs_eigenvalue"] = float(np.min(np.abs(eigvals)))
                if symmetric:
                    result["is_positive_definite"] = bool(np.min(eigvals) > tol)
            except Exception:
                pass
            try:
                result["condition_number_2"] = float(np.linalg.cond(A))
            except Exception:
                pass

        return result
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def choose_nla_algorithm(
    task_type: str,
    A_rows: Optional[list[list[float]]] = None,
    A_csc: Optional[dict] = None,
) -> dict:
    """
    显式算法策略：根据任务类型 + 矩阵性质返回推荐 API 与理由。
    """
    has_matrix_input = A_rows is not None or A_csc is not None
    analyzed = None
    if has_matrix_input:
        analyzed = analyze_matrix_properties(A_rows=A_rows, A_csc=A_csc)
        if analyzed.get("status") != "ok":
            return analyzed

    task = (task_type or "").strip().lower()
    square = bool(analyzed.get("is_square")) if analyzed else False
    symmetric = bool(analyzed.get("is_symmetric")) if analyzed else False
    sparse = bool(analyzed.get("is_sparse_input")) if analyzed else False
    spd = bool(analyzed.get("is_positive_definite", False)) if analyzed else False

    recommendation: dict = {
        "status": "ok",
        "task_type": task,
        "matrix_properties": analyzed,
        "matrix_properties_available": bool(analyzed),
        "recommended_apis": [],
        "fallback_apis": [],
        "reasoning": "",
        "checks": [],
    }

    if task == "solve_linear_system":
        if not analyzed:
            recommendation["recommended_apis"] = [
                "solve_linear_lapack",
                "numpy.linalg.solve",
                "spsolve_sparse",
            ]
            recommendation["fallback_apis"] = ["numpy.linalg.lstsq", "gmres_sparse"]
            recommendation["reasoning"] = "未提供显式矩阵时，先给出按稠密/稀疏分支的通用解法。"
        elif sparse and square and symmetric and spd:
            recommendation["recommended_apis"] = ["cg_sparse"]
            recommendation["fallback_apis"] = ["spsolve_sparse", "solve_linear_lapack"]
            recommendation["reasoning"] = "稀疏 + 对称正定，优先共轭梯度法。"
        elif sparse and square:
            recommendation["recommended_apis"] = ["spsolve_sparse"]
            recommendation["fallback_apis"] = ["gmres_sparse", "solve_linear_lapack"]
            recommendation["reasoning"] = "稀疏方阵，优先稀疏直接法。"
        elif square and symmetric and spd:
            recommendation["recommended_apis"] = ["solve_linear_lapack", "scipy.linalg.cho_factor", "scipy.linalg.cho_solve"]
            recommendation["fallback_apis"] = ["numpy.linalg.solve"]
            recommendation["reasoning"] = "稠密对称正定，优先 Cholesky 分解。"
        elif square:
            recommendation["recommended_apis"] = ["solve_linear_lapack", "numpy.linalg.solve"]
            recommendation["fallback_apis"] = ["scipy.linalg.lu_factor", "scipy.linalg.lu_solve"]
            recommendation["reasoning"] = "一般方阵，优先直接解法。"
        else:
            recommendation["recommended_apis"] = ["least_squares_lapack", "numpy.linalg.lstsq"]
            recommendation["fallback_apis"] = ["scipy.linalg.lstsq"]
            recommendation["reasoning"] = "非方阵或欠定/超定系统，转最小二乘。"
        recommendation["checks"] = ["检查矩阵秩", "关注条件数", "必要时比较残差"]

    elif task == "eigen":
        if not analyzed:
            recommendation["recommended_apis"] = ["eigs_sparse", "eigsh_sparse"]
            recommendation["fallback_apis"] = ["numpy.linalg.eig", "numpy.linalg.eigh"]
            recommendation["reasoning"] = "未提供显式矩阵时，优先给出适用于大规模稀疏矩阵的部分特征值方案。"
            recommendation["checks"] = ["默认仅计算前 k 个特征值", "避免展开完整矩阵/CSC", "验证 A v ≈ lambda v"]
            return recommendation
        if sparse and square and symmetric:
            recommendation["recommended_apis"] = ["eigsh_sparse"]
            recommendation["fallback_apis"] = ["eigs_sparse"]
            recommendation["reasoning"] = "稀疏对称矩阵，优先 eigsh。"
        elif sparse and square:
            recommendation["recommended_apis"] = ["eigs_sparse"]
            recommendation["fallback_apis"] = ["scipy.linalg.eig"]
            recommendation["reasoning"] = "稀疏一般矩阵，优先部分特征值算法。"
        elif square and symmetric:
            recommendation["recommended_apis"] = ["numpy.linalg.eigh"]
            recommendation["fallback_apis"] = ["scipy.linalg.eigh"]
            recommendation["reasoning"] = "稠密对称矩阵，优先 eigh。"
        elif square:
            recommendation["recommended_apis"] = ["numpy.linalg.eig"]
            recommendation["fallback_apis"] = ["scipy.linalg.eig"]
            recommendation["reasoning"] = "一般方阵，使用 eig。"
        else:
            return {"status": "error", "message": "特征值问题需要方阵"}
        recommendation["checks"] = ["验证 A v ≈ lambda v", "关注是否出现复特征值"]

    elif task == "least_squares":
        recommendation["recommended_apis"] = ["least_squares_lapack", "numpy.linalg.lstsq"]
        recommendation["fallback_apis"] = ["scipy.linalg.lstsq"]
        recommendation["reasoning"] = "最小二乘问题优先 SVD 方案。"
        recommendation["checks"] = ["报告残差范数", "检查秩亏"]

    elif task == "determinant":
        if not analyzed:
            recommendation["recommended_apis"] = ["numpy.linalg.slogdet"]
            recommendation["fallback_apis"] = ["numpy.linalg.det"]
            recommendation["reasoning"] = "未提供显式矩阵时，先给出方阵场景下的稳定计算建议。"
            recommendation["checks"] = ["确认是方阵后再执行", "同时报告 sign 与 logabsdet"]
            return recommendation
        if not square:
            return {"status": "error", "message": "行列式仅适用于方阵"}
        recommendation["recommended_apis"] = ["numpy.linalg.slogdet"]
        recommendation["fallback_apis"] = ["numpy.linalg.det"]
        recommendation["reasoning"] = "优先 slogdet 提升数值稳定性。"
        recommendation["checks"] = ["同时报告 sign 与 logabsdet"]

    elif task == "inverse":
        if not analyzed:
            recommendation["recommended_apis"] = ["numpy.linalg.solve"]
            recommendation["fallback_apis"] = ["numpy.linalg.inv"]
            recommendation["reasoning"] = "未提供显式矩阵时，先给出工程上更稳健的通用建议。"
            recommendation["checks"] = ["确认矩阵可逆", "检查条件数"]
            return recommendation
        if not square:
            return {"status": "error", "message": "矩阵求逆仅适用于方阵"}
        recommendation["recommended_apis"] = ["numpy.linalg.solve"]
        recommendation["fallback_apis"] = ["numpy.linalg.inv"]
        recommendation["reasoning"] = "工程上优先解线性系统，不直接求逆。"
        recommendation["checks"] = ["检查条件数", "验证 A @ x 与 b 一致性"]

    else:
        recommendation["recommended_apis"] = ["search_numpy_scipy_docs"]
        recommendation["reasoning"] = "任务未识别，先检索文档再决策。"
        recommendation["checks"] = ["补充任务类型", "分析矩阵性质"]

    return recommendation
