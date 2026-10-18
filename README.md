# AI App Lab · 课程知识库 RAG 问答 + 多步工具调用 Agent

两个可运行、可评估、可复现的大模型应用项目。**所有指标都有对应的评估报告文件，命令可复跑。**

```
agent/  ──复用──▶  rag/
（决策与规划）      （检索与溯源）
     │                  ▲
     └── search_course_kb 工具直接调用 rag 的「混合检索 + 重排」
```

| 项目 | 目录 | 一句话 | 核心验证 |
| --- | --- | --- | --- |
| **CourseRAG** | [`rag/`](rag/) | 课程知识库问答：混合检索 + 精排 + 章节级引用 + 两道拒答闸门 | 38 条标注集的检索消融实验 |
| **AgentLab** | [`agent/`](agent/) | 多步工具调用 Agent：6 类工具、五种失控防护、轨迹落盘与回放 | 12 条任务的分层评估 |

Agent 不重新实现检索，而是把 CourseRAG 的检索器**当成一个工具**；检测不到 `rag/` 时自动降级到内置迷你知识库，**单独 clone 也能跑**。

---

## 一、实测结果

### CourseRAG · 检索消融实验

评估集 **38 条**，其中 35 条为可判定检索的问答，3 条为「库里没有」的越界拒答样本（不参与检索指标分母）。
判定粒度：**片段级（章节命中）**。切分 `recursive / size=500 / overlap=80`，向量模型 `bge-small-zh-v1.5`，精排 `bce-reranker-base_v1`。

| 检索方案 | Hit@1 | Hit@3 | Hit@5 | MRR | Recall@5 | nDCG@5 | 平均耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 仅向量检索（基线） | 0.600 | 0.914 | 0.943 | 0.754 | 0.779 | 0.701 | 1968 ms |
| 仅 BM25 | 0.543 | 0.943 | **1.000** | 0.748 | 0.860 | 0.728 | 3317 ms |
| **混合检索 RRF** | **0.657** | 0.943 | 0.971 | **0.806** | 0.821 | **0.750** | 1270 ms |
| 混合检索 加权分数融合 | 0.629 | 0.943 | 0.971 | 0.787 | 0.821 | 0.735 | 1352 ms |
| 混合检索 + 精排 | 0.600 | **0.971** | **1.000** | 0.779 | **0.843** | 0.740 | 5415 ms |

> **口径说明**：上表「混合检索 + 精排」的耗时含**候选池 20、送模型文本不截断**的开销（当年评估的真实配置）。
> 线上默认已把 `RERANK_CANDIDATE_POOL` 降到 10、`RERANK_MAX_CHARS` 截到 320，实测约 **1.1 s/query**；
> 这组延迟数字属于优化后的配置，**指标未在该配置下重测**，两者不要混着讲（详见 [rag/README §4.6](rag/README.md)）。

**相对基线（仅向量检索）**：Hit@1 **+9.5%**、MRR **+6.8%**、nDCG@5 +7.0%、Recall@5 +5.5%，且混合检索比纯向量**更快**（1968 → 1270 ms）。

> ⚠️ **这里有一个必须说清的反直觉结论**：加精排后 Hit@5 与 Recall@5 变好了，但 **Hit@1 反而从 0.657 掉到 0.600**。
> 原因是 cross-encoder 以相关性排序为目标，会把正确答案从第 1 位挪到第 2–3 位，换取「top5 里一定有」。
> 所以「加精排一定更好」是错的——**是否上精排取决于你要 Hit@1 还是 Hit@5，以及能否接受 4.3 倍的延迟**。
> 报告里两个方案的指标与延迟是并列呈现的，没有只报好的那一栏。

**查询改写消融**（`ENABLE_QUERY_REWRITE`，默认关闭）：

| 场景 | Hit@1 | MRR |
| --- | --- | --- |
| 无改写（[报告](rag/data/eval/reports/report-20261005-220943-cmp-norewrite.md)） | 0.429 | 0.649 |
| 开启改写（[报告](rag/data/eval/reports/report-20261005-222858-FINAL-rewrite.md)） | **0.657** | **0.806** |

Hit@1 相对提升 **+53%**，MRR 相对提升 **+24%**。

> 定位过程：逐样本排查发现「K-means 里 K 该怎么选」这类口语提问与教材标题「8.3 如何选择簇数 K」字面不重合，
> 稠密检索排到第 17 名、BM25 排第 14 名，**两路都进不了最终上下文**，模型只能合理拒答。
> 这不是检索器坏了，是查询与文档的表述不对齐——用一次轻量 LLM 调用补上教材术语即可解决。

**生成质量**（LLM-as-Judge，1–5 分，38/38 样本参与打分）：忠实度 5.0 / 相关性 5.0 / 引用正确性 5.0，越界拒答 **3/3**（n=3，全部正确拒答、**零误拒**），未检出幻觉片段，平均端到端延迟 6384 ms。

> 说明：这三项满分是在**本项目自建的 38 条评估集**上取得的，样本量小、且裁判与被测模型同源（见[已知局限](#六已知局限)第 5 条）。它说明的是「在这套标注集上稳定」，而不是「普遍无幻觉」。

### AgentLab · 分层评估（决策后端 deepseek-chat，12 条任务）

| 层级 | 指标 | 数值 |
| --- | --- | --- |
| 工程层（与后端无关） | 工具执行成功率 / 轨迹完整率 | 100% / 100% |
| 工程层 | 防护触发率 | 0.0% |
| 能力层（真实模型决策） | 任务成功率 | **100%（12/12）**；重复运行 91.7%–100% |
| 能力层 | 工具选择准确率 | **100%**（含「课程内问题不得联网」的克制性检查） |
| 能力层 | 平均步数 / 耗时 / token | 3.17 步 / 6.0 s / 5254 token |

逐题明细见 [agent-eval-20261005-213519-real-fixed.md](agent/data/eval/reports/agent-eval-20261005-213519-real-fixed.md)（含每题的停止原因、步数、工具序列）。

> 为什么必须分两层：只看任务成功率，模型把 6 类工具全调一遍也可能「成功」。
> 所以判据同时约束**该用哪些工具**与**不该用哪些工具**（`forbid_tools`），让「工具使用的克制」也被量化。

---

## 二、快速开始

需要 Python 3.10+（开发环境 3.13）。两个项目共用一个虚拟环境。

```bash
cd rag
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt   # 核心依赖 + 本地向量/精排模型（torch 为 CPU 版）
cp .env.example .env              # 可留空
python -m scripts.build_index     # 建索引（首次下载向量模型约 95MB）
python -m scripts.serve           # http://127.0.0.1:8000
```

```bash
cd ../agent
pip install -r requirements.txt   # 只补几个包，复用上面的虚拟环境
python -m scripts.run --preflight "信息增益和基尼指数有什么区别？"
python -m scripts.serve           # http://127.0.0.1:8010
```

> **不配 API Key 也能完整跑通**：CourseRAG 降级为抽取式回答，AgentLab 降级为脚本化策略。
> 两者都会在界面、日志与报告里**显式标注当前是降级模式**，不会把降级结果当成真实能力。

### 复现评估（不需要 API Key）

```bash
cd rag
# 检索消融：5 种方案一次跑完，输出 data/eval/reports/report-<时间>.{json,md}
# 无需 API Key（只跑检索指标）
python -m scripts.evaluate --retrieval-only

# 查询改写前后对照：多跑一次带 --rewrite 的（需 API Key，每题多一次模型调用）
python -m scripts.evaluate --retrieval-only
python -m scripts.evaluate --retrieval-only --rewrite

cd ../agent
# 12 条任务评估；未配置 Key 时会用离线 mock，并在报告里明确标注
python -m scripts.evaluate
```

> **关于报告里的路径**：`data/eval/reports/*` 的「运行环境」一节记录了生成时的绝对路径，
> 那时这个仓库还叫 `course-rag`（后改名为 `ai-app-lab`），因此报告里能看到旧目录名。
> 报告本身是原始产物，我选择保留原样而不是改路径字符串——**改一个数字或路径来让报告「看起来一致」，比留着不一致更糟**。
> 要验证一致性，请直接跑上面的命令，新报告会写入你本机的真实路径。

---

## 三、技术栈与设计取舍

**共同点**：Python · FastAPI · 中文语料 · 全离线单元测试（共 196 个）· 显式降级与边界说明

| | CourseRAG | AgentLab |
| --- | --- | --- |
| 检索 | BM25 + 向量 + RRF 融合 + Cross-Encoder 精排 | 复用 CourseRAG |
| 生成 | OpenAI 兼容任意端点（DeepSeek / 硅基流动 / 通义…） | 同上，支持 function calling |
| 存储 | Chroma + BM25 缓存 | 执行轨迹 JSONL + 摘要索引 |
| 评估 | Hit@k / MRR / nDCG + LLM 裁判 | 工程层 / 能力层双层指标 |

### 三个刻意的设计决定

**1. 关键部分先手写一遍，但不排斥框架。** 混合检索、融合、精排、Agent 主循环都先自己实现；LangChain 只用在真正合适的地方（Embeddings 协议、文本切分工具、Chroma 适配）。理由是要能解释每一行为什么这样写，而不是「框架里就是这么调的」。

主循环后来又用 **LangGraph 重写了一版并与手写版并存**（见 [agent/README.md](agent/README.md) 第八节）——两个版本行为等价，但 LangGraph 版多出 checkpointer 持久化与人工审批。**先手写是为了理解；理解之后才有资格判断框架到底替你做了什么、代价是什么。** 只会框架的人讲不出这个取舍。

**2. 降级必须显式。** 没有 Key、没有外部项目、工具失败——每种降级都在界面、日志和报告里写明当前状态。**一个没标注清楚的降级模式比没有降级更危险。**

**3. 评估先于优化。** 两个项目都先建评估集再改代码，并且都栽在同一个坑上：**数据不干净时，优化方向也会错**。CourseRAG 曾因约 15% 的「空壳片段」得出过错误的融合策略结论（详见 [rag/README.md](rag/README.md)）。

---

## 四、仓库结构

```
ai-app-lab/
├── rag/                        CourseRAG · 知识库问答
│   ├── app/
│   │   ├── ingest/             解析 → 切分（fixed / recursive / semantic）
│   │   ├── llm/                向量化、查询改写、生成（带引用）
│   │   ├── retrieval/          BM25 + 向量 + RRF 融合 + cross-encoder 精排
│   │   ├── pipeline/           端到端链路 + 两道拒答闸门
│   │   ├── eval/               指标、消融执行器、LLM 裁判
│   │   └── api/                FastAPI + SSE 流式
│   ├── scripts/                build_index / ask / evaluate / serve / calibrate
│   ├── data/raw/               课程语料（12 篇讲义、实验手册、FAQ）
│   ├── data/eval/              两套标注评估集 + 4 份实测报告
│   └── docs/architecture.md    设计决策与权衡
│
└── agent/                      AgentLab · 工具调用 Agent
    ├── app/
    │   ├── tools/              6 类工具（含内置迷你知识库兜底）
    │   ├── brain/              决策后端（function calling + 离线 mock）
    │   ├── runtime/            主循环 + 五种防护 + 轨迹存储
    │   └── eval/               分层评估（工程层 / 能力层）
    ├── web/                    轨迹回放前端
    └── docs/architecture.md    设计决策与权衡
```

---

## 五、测试

```bash
cd rag   && python -m pytest -q    # 60 个
cd agent && python -m pytest -q    # 136 个（未装可选依赖 LangGraph 时为 109 个）
```

合计 **196 个**；未装可选依赖 LangGraph 时为 **169 个**（60 + 109）。

**全部离线运行**：不下载模型、不联网、不需要 API Key。覆盖重点包括注入攻击拒绝、沙箱环境变量不泄漏、五种防护的边界场景、轨迹往返一致、配置空值回退，以及「礼貌拒答不能算成功」这类判据陷阱。

### 国内网络环境下实际踩到的三个坑

| 现象 | 原因 | 解法 |
| --- | --- | --- |
| `pip install -r requirements.txt` 报 `UnicodeDecodeError: 'gbk' codec` | requirements 里有中文注释，pip 用系统默认编码读取 | `$env:PYTHONUTF8=1; $env:PYTHONIOENCODING='utf-8'` |
| 装不上 `langchain`（`No matching distribution`） | 清华 TUNA 镜像无法解析该包（实测） | 换源 `-i https://mirrors.aliyun.com/pypi/simple/` |
| `git push` 报 `schannel: server closed abruptly` | 系统走了本地代理，但 git 没配 | `git config --global http.proxy http://127.0.0.1:<端口>` |

> 第三条的判断方法：浏览器能打开 GitHub 但 `git` 连不上时，查系统代理（`HKCU:\...\Internet Settings` 的 `ProxyServer`）并给 git 配上同一个。

---

## 六、已知局限

诚实列出边界比声称完美更可信。完整清单见各项目 README，这里是最重要的几条：

1. **沙箱不是内核级隔离**：能防住死循环、刷屏、误删工作目录外文件、子进程环境泄漏密钥，但**挡不住蓄意逃逸**（如 `ctypes` 直接调系统 API）。生产环境应换容器或微虚拟机。
2. **无 Key 时拒答能力打折**：抽取式降级只做词汇重合判断，对「用课程术语包装但库里没有答案」的问题会返回片段摘要。
3. **联网搜索靠解析结果页 HTML**：不依赖付费 API，因此页面结构变化会导致解析失效（已做双引擎回退与明确报错）。
4. **评估集偏小**：38 条问答 + 12 条 Agent 任务，足以对比方案差异，但**不足以对总体能力下定论**，扩到数百条并引入人工评分交叉验证才稳妥。
5. **LLM 裁判与被测模型同源**：存在自我偏好，严格评估应引入人工基线。这也是上面三个 5.0 分不能外推的原因。
6. **精排会牺牲 Hit@1 且贵 4 倍**：见第一节的反直觉结论——加精排后 Hit@1 从 0.657 掉到 0.600，延迟从 1.27 s 升到 5.4 s。代码里 `RERANK_ENABLED` 默认为 `true`，如果你更看重 Hit@1 或部署在 CPU 上，应显式关掉它并改用 RRF 结果。

---

## 七、许可

本项目采用 [MIT License](LICENSE)，可自由使用、修改与分发。
课程语料为演示用整理内容（非任何机构的真实内部资料），可替换为自己的资料。
