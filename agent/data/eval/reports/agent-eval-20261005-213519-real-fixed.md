# AgentLab 任务评估报告

- 生成时间：2026-10-05 21:35:19
- 决策后端：**deepseek-chat**
- 任务数：12

## 一、工程层指标（与决策后端无关，任何模式下都成立）

| 指标 | 数值 | 说明 |
|---|---|---|
| 工具执行成功率 | 100.0% | 工具调用未抛异常、未超时的比例 |
| 轨迹完整率 | 100.0% | 步骤序列完整且可反序列化的比例 |
| 防护触发率 | 0.0% | 步数/重复/连续失败防护被激活的比例 |
| 平均耗时 | 6.0 s | 端到端，含模型与工具时间 |

## 二、能力层指标（真实模型决策）

| 指标 | 数值 | 说明 |
|---|---|---|
| 任务成功率 | **100.0%** | 工具使用 + 答案关键词综合判定 |
| 答案覆盖度 | 100.0% | 预期关键词命中比例 |
| 工具选择准确率 | 100.0% | 是否用了该用的工具 |
| 平均步数 | 3.17 | 越少越高效 |
| 平均工具调用 | 1.58 | 反映绕路程度 |
| 平均 token | 5254 | 真实成本 |
| 失败恢复率 | 0.0% | 工具报错后仍完成的比例 |

### 停止原因分布

| 原因 | 次数 |
|---|---|
| finished | 12 |

## 三、逐题明细

| 任务 | 结果 | 停止原因 | 步数 | 工具序列 | 备注 |
|---|---|---|---|---|---|
| t001 | ✔ | finished | 3 | search_course_kb → search_course_kb |  |
| t002 | ✔ | finished | 3 | search_course_kb |  |
| t003 | ✔ | finished | 3 | search_course_kb |  |
| t004 | ✔ | finished | 3 | search_course_kb |  |
| t005 | ✔ | finished | 3 | search_course_kb → search_course_kb |  |
| t006 | ✔ | finished | 3 | search_course_kb |  |
| t007 | ✔ | finished | 3 | search_course_kb |  |
| t008 | ✔ | finished | 3 | calculator → calculator |  |
| t009 | ✔ | finished | 3 | run_python |  |
| t010 | ✔ | finished | 3 | current_time |  |
| t011 | ✔ | finished | 5 | web_search → current_time → fetch_url → web_search |  |
| t012 | ✔ | finished | 3 | search_course_kb → search_course_kb |  |
