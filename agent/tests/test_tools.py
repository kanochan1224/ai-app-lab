"""工具层测试：安全求值、参数校验、失败兜底、沙箱行为。

全部离线，不依赖网络与大模型。
"""

from __future__ import annotations

import pytest

from app.schema import ToolCall
from app.tools import CalculatorTool, CodeExecTool, CurrentTimeTool
from app.tools.base import ToolContext, ToolError, ToolRegistry
from app.tools.calculator import safe_eval


# --------------------------------------------------------------------------- #
# 计算器：正确性 + 安全性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "expr,expected",
    [
        ("1+2*3", 7),
        ("(1+2)*3", 9),
        ("2**10", 1024),
        ("10/4", 2.5),
        ("7//2", 3),
        ("7%3", 1),
        ("-5+3", -2),
        ("sqrt(16)", 4),
        ("log(8,2)", 3),
        ("max(3,7,5)", 7),
    ],
)
def test_safe_eval_arithmetic(expr, expected):
    assert safe_eval(expr) == pytest.approx(expected)


def test_safe_eval_floats_are_cleaned():
    """浮点尾差要被抹掉，否则模型会拿到 0.30000000000000004 这类噪音。"""
    assert safe_eval("0.1+0.2") == 0.3
    assert safe_eval("1/3*3") == 1


def test_safe_eval_constants():
    assert safe_eval("pi") == pytest.approx(3.14159265, rel=1e-6)
    assert safe_eval("e") == pytest.approx(2.71828182, rel=1e-6)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('whoami')",   # 代码注入
        "open('/etc/passwd')",                 # 文件访问
        "eval('1+1')",                         # 嵌套求值
        "().__class__.__bases__",              # 属性穿透
        "lambda: 1",
        "[x for x in range(3)]",               # 推导式
        "a = 1",                               # 赋值
        "1 if True else 2",                    # 条件表达式
        "print(1)",                            # 非白名单函数
    ],
)
def test_safe_eval_rejects_dangerous_input(expr):
    with pytest.raises(ToolError):
        safe_eval(expr)


def test_safe_eval_guards_against_huge_power():
    with pytest.raises(ToolError):
        safe_eval("2**99999")


def test_safe_eval_rejects_empty_and_oversized():
    with pytest.raises(ToolError):
        safe_eval("")
    with pytest.raises(ToolError):
        safe_eval("1+" * 400 + "1")


def test_calculator_tool_returns_value():
    result = CalculatorTool().run({"expression": "12*12"}, ToolContext())
    assert result.ok is True
    assert "144" in result.content
    assert result.raw["value"] == 144


# --------------------------------------------------------------------------- #
# 注册表：校验、兜底、裁剪
# --------------------------------------------------------------------------- #
def test_registry_unknown_tool_returns_failure_not_exception():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    result = registry.invoke(ToolCall(name="not_exist", arguments={}))
    assert result.ok is False
    assert "不存在名为 not_exist 的工具" in result.error
    assert "calculator" in result.content          # 提示可用工具，便于模型纠正


def test_registry_validates_required_arguments():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    result = registry.invoke(ToolCall(name="calculator", arguments={}))
    assert result.ok is False
    assert "缺少必填参数" in result.error


def test_registry_rejects_unknown_arguments():
    """模型偶尔会臆造参数名，要明确指出来，而不是静默忽略。"""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    result = registry.invoke(
        ToolCall(name="calculator", arguments={"expression": "1+1", "precision": 2})
    )
    assert result.ok is False
    assert "未定义参数" in result.error


def test_registry_truncates_long_observation():
    class LongTool(CalculatorTool):
        name = "long"

        def run(self, arguments, context):
            from app.schema import ToolResult

            return ToolResult(ok=True, content="x" * 5000)

    registry = ToolRegistry(max_observation_chars=200)
    registry.register(LongTool())
    result = registry.invoke(ToolCall(name="long", arguments={"expression": "1"}))
    assert result.truncated is True
    assert len(result.content) < 400
    assert "已截断" in result.content


def test_registry_catches_tool_internal_exception():
    class BoomTool(CalculatorTool):
        name = "boom"

        def run(self, arguments, context):
            raise ValueError("内部炸了")

    registry = ToolRegistry()
    registry.register(BoomTool())
    result = registry.invoke(ToolCall(name="boom", arguments={"expression": "1"}))
    assert result.ok is False
    assert "工具内部异常" in result.error
    assert "内部炸了" in result.error


def test_registry_records_elapsed_time():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    result = registry.invoke(ToolCall(name="calculator", arguments={"expression": "2+2"}))
    assert result.elapsed_ms >= 0


# --------------------------------------------------------------------------- #
# 代码沙箱
# --------------------------------------------------------------------------- #
def test_code_exec_runs_and_captures_stdout(tmp_path):
    tool = CodeExecTool(workspace=tmp_path)
    result = tool.run({"code": "print(sum(range(11)))"}, ToolContext(workspace=str(tmp_path)))
    assert result.ok is True
    assert "55" in result.content


def test_code_exec_reports_error_with_stderr(tmp_path):
    tool = CodeExecTool(workspace=tmp_path)
    result = tool.run({"code": "raise ValueError('boom')"}, ToolContext(workspace=str(tmp_path)))
    assert result.ok is False
    assert "ValueError" in result.content


def test_code_exec_times_out(tmp_path):
    tool = CodeExecTool(workspace=tmp_path)
    result = tool.run(
        {"code": "while True: pass", "timeout": 2},
        ToolContext(workspace=str(tmp_path)),
    )
    assert result.ok is False
    assert "超时" in result.content


def test_code_exec_does_not_leak_env_secrets(tmp_path, monkeypatch):
    """子进程环境必须被清空：密钥不能泄漏给模型生成的代码。"""
    monkeypatch.setenv("LLM_API_KEY", "sk-super-secret")
    tool = CodeExecTool(workspace=tmp_path)
    result = tool.run(
        {"code": "import os; print(os.environ.get('LLM_API_KEY', 'NOT_FOUND'))"},
        ToolContext(workspace=str(tmp_path)),
    )
    assert "NOT_FOUND" in result.content
    assert "sk-super-secret" not in result.content


def test_code_exec_cleans_up_script_file(tmp_path):
    tool = CodeExecTool(workspace=tmp_path)
    tool.run({"code": "print(1)"}, ToolContext(workspace=str(tmp_path)))
    leftovers = list(tmp_path.glob("_agent_snippet_*.py"))
    assert leftovers == [], f"临时脚本未清理：{leftovers}"


def test_code_exec_rejects_empty_code(tmp_path):
    tool = CodeExecTool(workspace=tmp_path)
    with pytest.raises(ToolError):
        tool.run({"code": "   "}, ToolContext(workspace=str(tmp_path)))


def test_code_exec_clamps_timeout(tmp_path):
    """超时参数必须被夹到上限，不能由模型要求「等一小时」。"""
    tool = CodeExecTool(workspace=tmp_path)
    result = tool.run(
        {"code": "print('ok')", "timeout": 99999},
        ToolContext(workspace=str(tmp_path)),
    )
    assert result.ok is True


# --------------------------------------------------------------------------- #
# 时间工具
# --------------------------------------------------------------------------- #
def test_current_time_default_timezone():
    result = CurrentTimeTool().run({}, ToolContext())
    assert result.ok is True
    assert "当前时间" in result.content
    assert result.raw["date"].count("-") == 2


def test_current_time_rejects_unknown_timezone():
    with pytest.raises(ToolError):
        CurrentTimeTool().run({"timezone": "Mars/Olympus"}, ToolContext())


def test_tool_specs_convert_to_openai_schema():
    spec = CalculatorTool().spec()
    schema = spec.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculator"
    assert "expression" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["description"]
