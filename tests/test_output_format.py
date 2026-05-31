from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "output_format.py"
_SPEC = importlib.util.spec_from_file_location("output_format", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load output_format from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
sanitize_latex_markdown = _MODULE.sanitize_latex_markdown


class TestOutputFormat(unittest.TestCase):
    def test_converts_common_latex_delimiters(self):
        text = r"行内公式 \(Ax=b\)，块级公式：\[A=QR\]"

        normalized = sanitize_latex_markdown(text)

        self.assertIn("$Ax=b$", normalized)
        self.assertIn("$$\nA=QR\n$$", normalized)
        self.assertNotIn(r"\(", normalized)
        self.assertNotIn(r"\[", normalized)

    def test_escapes_unbalanced_inline_dollars(self):
        text = "结论：$x=A^{-1}b\n下一行正常"

        normalized = sanitize_latex_markdown(text)

        self.assertIn(r"结论：\$x=A^{-1}b", normalized)
        self.assertNotIn("结论：$x=A^{-1}b", normalized)

    def test_escapes_unbalanced_display_dollars(self):
        text = "$$\nA=QR"

        normalized = sanitize_latex_markdown(text)

        self.assertEqual(normalized, r"\$\$" + "\nA=QR")

    def test_repairs_missing_display_opener_dollar_at_paragraph_start(self):
        text = (
            "伪逆为：\n"
            "$A^+=\\begin{bmatrix}2/3&-1/3\\\\1/3&1/3\\end{bmatrix}$$\n"
            "继续说明。"
        )

        normalized = sanitize_latex_markdown(text)

        self.assertIn(
            "$$A^+=\\begin{bmatrix}2/3&-1/3\\\\1/3&1/3\\end{bmatrix}$$",
            normalized,
        )
        self.assertNotIn("\n$A^+", normalized)

    def test_keeps_paragraph_start_inline_math_unchanged(self):
        text = "$x$ 是最小范数解。\n\n$$\nA=QR\n$$"

        normalized = sanitize_latex_markdown(text)

        self.assertTrue(normalized.startswith("$x$ 是最小范数解。"))
        self.assertIn("$$\nA=QR\n$$", normalized)

    def test_preserves_valid_display_math_after_unbalanced_inline_dollar(self):
        text = "错误片段：$x\n\n$$\nA=QR\n$$"

        normalized = sanitize_latex_markdown(text)

        self.assertIn(r"错误片段：\$x", normalized)
        self.assertIn("$$\nA=QR\n$$", normalized)

    def test_keeps_inline_code_unchanged(self):
        text = "公式 $Ax=b$，代码 `$not math`，坏片段 $x"

        normalized = sanitize_latex_markdown(text)

        self.assertIn("$Ax=b$", normalized)
        self.assertIn("`$not math`", normalized)
        self.assertIn(r"坏片段 \$x", normalized)

    def test_keeps_code_fences_and_workspace_marker_unchanged(self):
        text = (
            "公式 \\(Ax=b\\)\n"
            "```python\nprint('\\(not math\\)')\n```\n"
            "<nla-workspace>{\"formula\":\"\\(raw\\)\"}</nla-workspace>"
        )

        normalized = sanitize_latex_markdown(text)

        self.assertIn("$Ax=b$", normalized)
        self.assertIn("```python\nprint('\\(not math\\)')\n```", normalized)
        self.assertIn("<nla-workspace>{\"formula\":\"\\(raw\\)\"}</nla-workspace>", normalized)


if __name__ == "__main__":
    unittest.main()
