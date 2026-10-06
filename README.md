# AI App Lab · RAG 知识库问答 + 工具调用 Agent

> 两个能量化的 LLM 应用项目。
> 底层是**检索**（让模型有依据地回答），上层是**Agent**（让模型动手完成任务），
> 两者共享同一套语料、索引与模型配置，构成一条完整的技术栈。

**此仓库的每个数字都是实测的**，包括失败与返工的过程——
每个项目都有一节「踩坑记录」，记录那些只有真正跑起来才会暴露的问题。

---

## 一、两个项目

| 项目 | 目录 |     |
| --- | --- | --- |
| **CourseRAG** | [`rag/`](rag/) | 课程知识库问答：混合检索 + 重排 + 章节级引用 + 主动拒答，配可量化的消融评估 |
| **AgentLab** | [`agent/`](agent/) | 多步工具调用 Agent：6 类工具、五种失控防护、执行轨迹全量回放、分层评估 |

### 它们如何协作

```
agent/  ──复用──▶  rag/
（决策与规划）      （检索与溯源）
     │                  ▲
     │                  │
     └── search_course_kb 工具直接调用 rag 的「混合检索 + 重排」
```

Agent 不重新实现一套检索，而是把 CourseRAG 的检索器**当成一个工具**。
检测不到 `rag/` 时，Agent 会自动降级到内置迷你知识库，**单独 clone 也能跑**。

---

## 二、实测结果

### CourseRAG（38 条标注样本，片段级严格口径）

| 检索方案 | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 |
| --- | --- | --- | --- | --- | --- |
| 仅向量检索 | 0.600 | 0.914 | 0.943 | 0.754 | 0.701 |
| 仅 BM25 | 0.543 | 0.943 | 1.000 | 0.748 | 0.728 |
| **混合检索 RRF** | **0.657** | 0.943 | 0.971 | **0.806** | **0.750** |
| 混合检索 + 重排 | 0.600 | **0.971** | **1.000** | 0.779 | 0.740 |

生成质量（LLM-as-Judge）：**忠实度 5.0 / 相关性 5.0 / 引用正确性 5.0**，
**拒答准确率 1.00（零误拒）、零幻觉**。

关键消融：**查询改写让 Hit@1 从 0.429 提升到 0.657（+53%）、MRR 提升 24%**。

### AgentLab（12 条任务评估集，deepseek-chat）

| 指标 | 数值 |
| --- | --- |
| 任务成功率 | **100%**（12/12） |
| 工具选择准确率 | **100%**（含「课程内问题不得联网」的克制性检查） |
| 工具执行成功率 / 轨迹完整率 | **100% / 100%** |
| 平均步数 / 耗时 / token | 3.17 步 / 6.0 秒 / 5254 token |

---

## 三、快速开始

需要 Python 3.10+（开发环境为 3.13）。两个项目共用一个虚拟环境。

### 1. 装依赖

```bash
cd rag
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt   # 核心依赖 + 本地向量/重排模型（torch 为 CPU 版）
```

### 2. 跑 CourseRAG

```bash
cp .env.example .env              # 可留空，不填 Key 也能跑（会降级为抽取式回答）
python -m scripts.build_index     # 建索引（首次会下载向量模型，约 95MB）
python -m scripts.serve           # 打开 http://127.0.0.1:8000
```

### 3. 跑 AgentLab

```bash
cd ../agent
pip install -r requirements.txt   # 只补几个包，复用上面那个虚拟环境

python -m scripts.run --preflight "信息增益和基尼指数有什么区别？"
python -m scripts.serve           # 打开 http://127.0.0.1:8010
```

> **不配 API Key 也能完整跑通**：CourseRAG 会降级为抽取式回答，
> AgentLab 会降级为脚本化策略。两者都会在界面与报告里**明确标注当前是降级模式**，
> 不会把降级结果当成真实能力。配了 Key 之后能力自动解锁。

---

## 四、技术栈与设计取舍

**共同点**：Python · FastAPI · 中文文档 · 全离线单元测试（共 169 个）· 明确的降级与边界说明

| | CourseRAG | AgentLab |
| --- | --- | --- |
| 检索 | BM25 + 向量 + RRF 融合 + Cross-Encoder 精排 | 复用 CourseRAG |
| 生成 | OpenAI 兼容任意端点（DeepSeek / 硅基流动 / 通义…） | 同上，支持 function calling |
| 存储 | Chroma + BM25 缓存 | 执行轨迹 JSONL |
| 评估 | Hit@k / MRR / nDCG + LLM 裁判 | 任务成功率 / 工具准确率 / 恢复率 |

### 三个刻意的设计决定

1. **能不用框架就不用**：混合检索、融合、重排、Agent 主循环都自己实现。
   LangChain 只用在真正合适的地方（Embeddings 协议、文本切分工具）。
   理由是要能解释每一行为什么这样写，而不是「框架里就是这么调的」。

2. **降级必须显式**：没有 Key、没有外部项目、工具失败——每一种降级都在
   界面、日志和报告里写明当前状态。**一个没标注清楚的降级模式比没有降级更危险。**

3. **评估先于优化**：两个项目都先建评估集再改代码。
   并且都遇到过同一个陷阱——**数据不干净时，优化方向也会错**
   （CourseRAG 曾因 15% 的「空壳片段」得出了错误的融合策略结论，
   详见 [`rag/README.md`](rag/README.md) 第 4.4 节）。

---

## 五、仓库结构

```
ai-app-lab/
├── rag/                     CourseRAG · 知识库问答
│   ├── app/                 解析 → 切分 → 向量化 → 混合检索 → 重排 → 带引用生成
│   ├── scripts/             build_index / ask / evaluate / serve / calibrate
│   ├── data/raw/            课程语料（12 篇讲义、实验手册、FAQ）
│   ├── data/eval/           两套标注评估集 + 实测报告
│   └── docs/architecture.md 设计决策与权衡
│
└── agent/                   AgentLab · 工具调用 Agent
    ├── app/
    │   ├── tools/           6 类工具（含内置迷你知识库兜底）
    │   ├── brain/           决策后端（function calling + 离线 mock）
    │   ├── runtime/         主循环 + 五种防护 + 轨迹存储
    │   └── eval/            分层评估
    ├── web/                 轨迹回放前端
    └── docs/architecture.md 设计决策与权衡
```

---

## 六、测试

```bash
cd rag   && python -m pytest -q    # 60 个
cd agent && python -m pytest -q    # 109 个
```

**全部离线运行**：不下载模型、不联网、不需要 API Key。
覆盖重点包括注入攻击拒绝、沙箱环境变量不泄漏、防护机制边界场景、
轨迹往返一致、配置空值回退、以及「礼貌拒答不能算成功」这类判据陷阱。



| 现象 | 原因 | 解法 |
| --- | --- | --- |
| `pip install -r requirements.txt` 报 `UnicodeDecodeError: 'gbk' codec` | requirements 里有中文注释，pip 用系统默认编码读取 | `$env:PYTHONUTF8=1; $env:PYTHONIOENCODING='utf-8'` |
| 装不上 `langchain`（`No matching distribution`） | 清华 TUNA 镜像当前无法解析该包（实测） | 换源 `-i https://mirrors.aliyun.com/pypi/simple/` |
| `git push` 报 `schannel: server closed abruptly` 或连接超时 | 系统走了本地代理，但 git 没配 | `git config --global http.proxy http://127.0.0.1:<端口>` |

> 第三条的判断方法：如果浏览器能打开 GitHub 但 `git` 连不上，
> 查一下系统代理（`HKCU:\...\Internet Settings` 的 `ProxyServer`）并给 git 配上同一个。

---

## 七、已知局限

诚实列出边界，比声称完美更可信。完整清单见各项目的 README，这里是最重要的几条：

1. **沙箱不是内核级隔离**：能防住死循环、刷屏、误删工作目录外文件、
   子进程环境泄漏密钥，但**挡不住蓄意逃逸**。生产环境应换容器或微虚拟机。
2. **无 Key 时拒答能力打折**：抽取式降级只做词汇重合判断，
   对「用课程术语包装但库里没有答案」的问题会返回片段摘要。
3. **联网搜索靠解析结果页 HTML**：不依赖付费 API，因此页面结构变化会导致解析失效
   （已做双引擎回退与明确报错）。
4. **评估集偏小**：38 条问答 + 12 条 Agent 任务，足以对比方案差异，
   但要下结论建议扩到数百条并引入人工评分交叉验证。
5. **LLM 裁判与被测模型同源**：存在自我偏好，严格评估应引入人工基线。

---

## 八、许可

本项目采用 [MIT License](LICENSE)，可自由使用、修改与分发。

课程语料为演示用整理内容（非任何机构的真实内部资料），可替换为自己的资料。

