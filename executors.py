"""
本地 Python 执行模块（带静态安全检查）。
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time

BLOCKED_BUILTINS = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}

BLOCKED_ATTRS = {
    "system",
    "popen",
    "run",
    "Popen",
    "remove",
    "unlink",
    "rmdir",
    "mkdir",
    "makedirs",
    "rename",
    "replace",
    "chmod",
    "chown",
    "rmtree",
    "listdir",
    "walk",
    "scandir",
    "read_text",
    "read_bytes",
    "write_text",
    "write_bytes",
    "load",
    "loadtxt",
    "genfromtxt",
    "fromfile",
}

BLOCKED_NODES = (
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
)


class SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTINS:
            self.errors.append(f"禁止调用内置函数: {node.func.id}")
        if isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_ATTRS:
            self.errors.append(f"禁止调用危险属性: {node.func.attr}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.errors.append("禁止访问 dunder 属性")
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, BLOCKED_NODES):
            self.errors.append(f"禁止语法结构: {type(node).__name__}")
        super().generic_visit(node)


def _safety_check(code: str) -> tuple[bool, list[str]]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, [f"代码语法错误: {exc}"]
    visitor = SafetyVisitor()
    visitor.visit(tree)
    if visitor.errors:
        dedup = list(dict.fromkeys(visitor.errors))
        return False, dedup
    return True, []


def _build_safe_env() -> dict:
    env = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "WINDIR": os.environ.get("WINDIR", ""),
        "TEMP": os.environ.get("TEMP", ""),
        "TMP": os.environ.get("TMP", ""),
    }
    return {k: v for k, v in env.items() if v}


def _extract_image_paths(stdout: str) -> tuple[str, list[str]]:
    """
    从 stdout 中抽取图片路径标记行，返回清理后的 stdout 与图片路径列表。
    """
    if not stdout:
        return "", []
    marker = "__NLA_IMAGE_FILES__="
    image_paths: list[str] = []
    kept_lines: list[str] = []
    for line in stdout.splitlines():
        if line.startswith(marker):
            payload = line[len(marker) :].strip()
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, list):
                    image_paths.extend(str(item) for item in parsed if item)
            except Exception:
                pass
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip(), image_paths


def _run_python_snippet_impl(
    code: str,
    timeout_s: float = 10.0,
    max_output_chars: int = 12_000,
    safety_check: bool = True,
) -> dict:
    src = (code or "").strip()
    if not src:
        return {"status": "error", "message": "code 不能为空"}
    if safety_check:
        ok, errors = _safety_check(src)
        if not ok:
            return {
                "status": "error",
                "error_type": "python_safety_error",
                "message": "代码未通过安全检查",
                "detail": {"errors": errors},
            }

    output_dir = os.path.join(os.getcwd(), "generated_images")
    os.makedirs(output_dir, exist_ok=True)

    prelude = r"""
import os as _os
import time as _time

_NLA_IMAGE_FILES = []
_NLA_IMAGE_OUTPUT_DIR = _os.environ.get("NLA_IMAGE_OUTPUT_DIR") or _os.path.join(_os.getcwd(), "generated_images")
_os.makedirs(_NLA_IMAGE_OUTPUT_DIR, exist_ok=True)
_os.environ.setdefault("MPLBACKEND", "Agg")

def _nla_pick_chinese_font():
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    try:
        from matplotlib import font_manager as _font_manager
        installed = {font.name for font in _font_manager.fontManager.ttflist}
        for name in preferred:
            if name in installed:
                return name
    except Exception:
        pass
    return None

def _nla_patch_matplotlib():
    try:
        import matplotlib as _matplotlib
        _matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as _plt

        chosen_font = _nla_pick_chinese_font()
        if chosen_font:
            _plt.rcParams["font.sans-serif"] = [chosen_font, "DejaVu Sans"]
        else:
            _plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
        _plt.rcParams["axes.unicode_minus"] = False

        _orig_show = _plt.show
        _orig_savefig = _plt.savefig

        def _nla_capture_show(*args, **kwargs):
            del args, kwargs
            fig_nums = list(_plt.get_fignums())
            if not fig_nums:
                return None
            stamp = int(_time.time() * 1000)
            for idx, num in enumerate(fig_nums, start=1):
                fig = _plt.figure(num)
                path = _os.path.join(_NLA_IMAGE_OUTPUT_DIR, f"plot_{stamp}_{idx}.png")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                _NLA_IMAGE_FILES.append(_os.path.abspath(path))
            _plt.close("all")
            return None

        def _nla_capture_savefig(*args, **kwargs):
            result = _orig_savefig(*args, **kwargs)
            if args:
                target = args[0]
                if isinstance(target, str):
                    _NLA_IMAGE_FILES.append(_os.path.abspath(target))
            return result

        _plt.show = _nla_capture_show
        _plt.savefig = _nla_capture_savefig
        return _orig_show
    except Exception:
        return None

_NLA_ORIG_SHOW = _nla_patch_matplotlib()
"""

    trailer = r"""
try:
    import json as _json
    if "RESULT" in globals():
        print(_json.dumps(globals()["RESULT"], ensure_ascii=False, default=str))
    elif "result" in globals():
        print(_json.dumps(globals()["result"], ensure_ascii=False, default=str))
except Exception:
    pass

try:
    import json as _json
    dedup = []
    seen = set()
    for path in globals().get("_NLA_IMAGE_FILES", []):
        if not path:
            continue
        p = str(path)
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    print("__NLA_IMAGE_FILES__=" + _json.dumps(dedup, ensure_ascii=False))
except Exception:
    pass
"""

    start = time.time()
    try:
        with tempfile.TemporaryDirectory() as td:
            snippet_path = os.path.join(td, "snippet.py")
            with open(snippet_path, "w", encoding="utf-8") as f:
                f.write(prelude.strip() + "\n\n")
                f.write(src)
                f.write("\n\n")
                f.write(trailer.strip() + "\n")

            env = dict(os.environ)
            env.update(_build_safe_env())
            env["NLA_IMAGE_OUTPUT_DIR"] = output_dir
            proc = subprocess.run(
                [sys.executable, snippet_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=float(timeout_s),
                cwd=os.getcwd(),
                env=env,
            )

        elapsed_ms = int((time.time() - start) * 1000)
        raw_stdout = (proc.stdout or "").strip()
        stdout, image_files = _extract_image_paths(raw_stdout)
        stderr = (proc.stderr or "").strip()
        if len(stdout) > max_output_chars:
            stdout = stdout[:max_output_chars].rstrip() + "\n...[stdout 已截断]"
        if len(stderr) > max_output_chars:
            stderr = stderr[:max_output_chars].rstrip() + "\n...[stderr 已截断]"
        downloadable_files = [p for p in image_files if os.path.isfile(p)]
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "image_files": downloadable_files,
            "elapsed_ms": elapsed_ms,
            "message": "已在子进程中执行代码片段；若生成图像，可从 image_files 下载。",
        }
    except subprocess.TimeoutExpired:
        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "status": "error",
            "message": f"执行超时（>{timeout_s}s）",
            "elapsed_ms": elapsed_ms,
        }
    except Exception as exc:
        elapsed_ms = int((time.time() - start) * 1000)
        return {"status": "error", "message": str(exc), "elapsed_ms": elapsed_ms}


def run_python_snippet(
    code: str,
    timeout_s: float = 10.0,
    max_output_chars: int = 12_000,
) -> dict:
    return _run_python_snippet_impl(
        code=code,
        timeout_s=timeout_s,
        max_output_chars=max_output_chars,
        safety_check=True,
    )


def run_python_snippet_unchecked(
    code: str,
    timeout_s: float = 10.0,
    max_output_chars: int = 12_000,
) -> dict:
    return _run_python_snippet_impl(
        code=code,
        timeout_s=timeout_s,
        max_output_chars=max_output_chars,
        safety_check=False,
    )
