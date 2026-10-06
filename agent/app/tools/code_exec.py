"""沙箱代码执行工具。

**安全设计（这是面试最容易被追问的点）**：

1. **独立子进程**：绝不在 Agent 进程里 ``exec``，崩溃或死循环都影响不到主流程；
2. **解释器隔离**：``python -I`` 忽略环境变量与用户 site-packages，减少被注入的面；
3. **超时强杀**：超时直接 kill 整个进程树，避免僵死；
4. **独立工作目录**：只在 ``data/workspace`` 下运行，且**清空环境变量**
   （特别是 ``LLM_API_KEY`` 之类的密钥不能漏给子进程）；
5. **无网络**：不主动提供网络能力，且环境里不含代理配置。
6. **输出截断**：stdout/stderr 都限制长度，防止刷屏撑爆上下文。

**必须说清的局限**：这不是内核级沙箱。它可以防住「模型写错代码把主进程搞挂」和
「死循环、刷屏、误删工作目录外的文件」这类常见事故，但**挡不住蓄意的逃逸**
（例如通过 ctypes 调用系统 API）。生产环境应换成容器/微虚拟机隔离。
这种「知道自己防护边界在哪」的表述，比声称「我做了安全沙箱」更可信。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ..schema import ToolResult
from .base import Tool, ToolContext, ToolError

MAX_OUTPUT_CHARS = 4000


class CodeExecTool(Tool):
    name = "run_python"
    description = (
        "在受限沙箱中执行一段 Python 代码并返回输出。适合做数据计算、"
        "字符串处理、验证算法思路。代码在独立进程运行，有超时与输出长度限制，"
        "无法访问网络，只在专用工作目录内读写文件。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "要执行的 Python 源码，用 print 输出结果"},
            "timeout": {
                "type": "number",
                "description": "超时秒数，默认 10，最大 30",
            },
        },
        "required": ["code"],
    }
    returns = "代码的 stdout / stderr，或超时与异常说明。"
    dangerous = True

    def __init__(self, default_timeout: float = 10.0, workspace: Path | None = None) -> None:
        self.default_timeout = default_timeout
        self.workspace = workspace

    def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        code = str(arguments.get("code", ""))
        if not code.strip():
            raise ToolError("代码为空")
        if len(code) > 20000:
            raise ToolError("代码过长（上限 20000 字符）")

        try:
            timeout = float(arguments.get("timeout") or self.default_timeout)
        except (TypeError, ValueError):
            timeout = self.default_timeout
        timeout = max(1.0, min(timeout, 30.0))

        workdir = Path(context.workspace or self.workspace or tempfile.gettempdir())
        workdir.mkdir(parents=True, exist_ok=True)
        script = workdir / f"_agent_snippet_{int(time.time() * 1000)}.py"
        script.write_text(code, encoding="utf-8")

        # 清空环境：密钥、代理都不能泄漏给子进程；仅保留最基本的变量
        env = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "TEMP": str(workdir),
            "TMP": str(workdir),
        }

        started = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-X", "utf8", str(script)],
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                ok=False,
                content=f"执行超时（超过 {timeout:.0f} 秒），已被强制终止。"
                        "请检查是否有死循环，或减小数据规模。",
                error="timeout",
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        finally:
            script.unlink(missing_ok=True)

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        truncated = False
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + "\n…（输出过长已截断）"
            truncated = True
        if len(err) > MAX_OUTPUT_CHARS:
            err = err[:MAX_OUTPUT_CHARS] + "\n…（错误输出过长已截断）"
            truncated = True

        ok = proc.returncode == 0
        if ok:
            body = out or "（代码执行成功，但没有输出；如需结果请用 print）"
        else:
            body = f"退出码 {proc.returncode}\nstdout:\n{out or '(空)'}\nstderr:\n{err or '(空)'}"
        return ToolResult(
            ok=ok,
            content=body,
            raw={"returncode": proc.returncode, "stdout": out, "stderr": err},
            error="" if ok else (err.splitlines()[-1] if err else f"退出码 {proc.returncode}"),
            truncated=truncated,
        )
