"""内置迷你知识库：让 agent-lab 单独 clone 下来也能回答课程问题。

**为什么需要它**：`search_course_kb` 正常情况下复用 course-rag 项目的完整
「混合检索 + 重排」能力（那是本项目的设计亮点）。但如果别人只 clone 了
agent-lab 这一个仓库，那个工具就会直接报「未找到 course-rag 项目」——
面试官不会为了看一个 Agent 再去 clone 第二个仓库，第一印象就此毁掉。

所以这里内置一份**最小可用的课程知识**（每个主题一小段），
用轻量的 BM25 做检索，保证：

- **单独 clone → 开箱能跑**，Agent 的完整链路（工具调用、轨迹、防护）都能演示；
- **同级存在 course-rag → 自动升级**，用完整的混合检索 + 重排 + 章节级引用；
- 两者接口一致，上层代码完全不感知差异。

内置语料刻意写得短而准，覆盖课程最常被问到的主题；它不是 course-rag 的替代品，
只是「没有外部依赖时也能跑」的兜底。
"""

from __future__ import annotations

from typing import Any

# (标题, 正文)。刻意覆盖最常被问到的主题，每个 100~200 字。
BUILTIN_KB: list[tuple[str, str]] = [
    (
        "信息熵与信息增益",
        "信息熵度量随机变量的不确定性：Ent(D) = -Σ p_k log2 p_k，越纯越小。"
        "信息增益是划分前后熵的减少量：Gain(D,a) = Ent(D) - Ent(D|a)，ID3 用它选划分属性。"
        "缺陷是偏向取值多的属性（如用「学号」划分，每个子集只有一个样本，增益最大但毫无泛化能力）。"
        "C4.5 用信息增益率 Gain_ratio = Gain / IV(a) 修正，IV 是属性 a 的固有值。",
    ),
    (
        "基尼指数",
        "CART 使用基尼指数：Gini(D) = 1 - Σ p_k²，表示随机抽两个样本类别不一致的概率。"
        "属性 a 的基尼指数为各子集基尼指数的加权平均，CART 选择使其最小的属性。"
        "相比信息熵，基尼指数不需要对数运算，计算更快，且只生成二叉树。",
    ),
    (
        "决策树剪枝",
        "预剪枝在划分前判断能否带来泛化提升，开销小但基于贪心，有欠拟合风险；"
        "后剪枝先建完整树再自底向上回缩，泛化通常更好但训练开销大。"
        "实践中常用代价复杂度剪枝（CART 的 CCP），用交叉验证选 alpha。",
    ),
    (
        "支持向量机与惩罚系数 C",
        "SVM 的核心是最大间隔：在能分开两类的超平面中选间隔最大的，间隔为 2/||w||。"
        "软间隔引入松弛变量与惩罚系数 C：C 越大越不容忍违反约束的样本，间隔越窄，"
        "偏差低方差高，容易过拟合；C 越小容忍度高、间隔宽，偏差高方差低，可能欠拟合。"
        "SVM 基于距离与内积，必须先做特征标准化。",
    ),
    (
        "核技巧",
        "当原始空间线性不可分时，把样本映射到高维空间往往就线性可分了。"
        "核技巧注意到对偶问题与判别函数中样本只以内积形式出现，"
        "于是只要定义 κ(xi,xj) = φ(xi)·φ(xj) 就无需显式计算高维映射。"
        "常用核：线性核、多项式核、高斯核（RBF，对应无穷维空间）。"
        "核矩阵必须半正定（Mercer 条件）。代价是训练复杂度 O(m²~m³)，样本量大时不实用。",
    ),
    (
        "感知机与 XOR",
        "单层感知机只能解决线性可分问题，决策边界是超平面，因此连 XOR 都学不会"
        "（XOR 的正负样本在平面上对角分布，不存在一条直线能分开）。"
        "加入隐层后，隐层把原空间映射到新的特征空间，XOR 在其中变得线性可分。"
    ),
    (
        "反向传播与梯度消失",
        "反向传播基于梯度下降：先前向计算输出与误差，再用链式法则从输出层往前逐层"
        "计算每个参数的偏导，沿负梯度更新。误差项递推：δ^l = (Σ w δ^{l+1}) f'(z^l)。"
        "梯度消失源于 Sigmoid 导数最大只有 0.25，逐层相乘后梯度指数衰减；"
        "ReLU 在正区间导数恒为 1，梯度可近乎无损回传，因此成为深层网络默认选择。",
    ),
    (
        "Bagging 与 Boosting",
        "Bagging 并行训练多个基学习器，通过 Bootstrap 采样制造差异，投票或平均，主要降低方差；"
        "Boosting 串行训练，每轮聚焦前面做错的样本，主要降低偏差。"
        "随机森林 = Bagging + 特征随机（节点分裂时先在随机特征子集中挑最优），"
        "进一步降低树之间的相关性。AdaBoost 中错误率越低的学习器权重越大："
        "α = 0.5 ln((1-ε)/ε)。",
    ),
    (
        "XGBoost 与 LightGBM",
        "XGBoost 相对 GBDT 的改进：二阶泰勒展开（同时用一阶与二阶导数）；"
        "目标函数显式加入正则项 γT + λ/2 Σw² 惩罚叶子数与叶子权重；支持特征并行；"
        "自动处理缺失值；内置交叉验证与早停。"
        "LightGBM 进一步用直方图算法离散化连续特征、GOSS 对小梯度样本采样、EFB 捆绑互斥稀疏特征，"
        "并采用叶子优先生长（同叶子数下损失更低，但更易过拟合，需限制 num_leaves）。",
    ),
    (
        "K-means 与如何选择 K",
        "K-means 最小化平方误差 E = Σ_k Σ_{x∈C_k} ||x - μ_k||²，用 Lloyd 迭代求解；"
        "目标函数非凸，只能保证收敛到局部最优，与初始质心强相关，故常用 K-means++ 初始化。"
        "它隐含各簇球形且大小相近的假设，必须先做标准化，对非凸形状应改用 DBSCAN。"
        "选择 K 的常用方法：**肘部法**（看 E 下降速度突然变缓的拐点）、"
        "**轮廓系数法**（取平均轮廓系数最大的 K）、**Gap Statistic**（比较实际数据与"
        "均匀参考数据的 E 差距）。三者结合并回到业务语义判断。",
    ),
    (
        "DBSCAN",
        "DBSCAN 是基于密度的聚类，参数为邻域半径 eps 与最少样本数 MinPts。"
        "核心点、边界点、噪声点三类：邻域内样本数≥MinPts 为核心点。"
        "优点是能发现任意形状的簇、自动识别噪声、不需要预设簇数；"
        "缺点是参数难调，各簇密度差异大时一组全局参数无法同时适配。",
    ),
    (
        "PCA 主成分分析",
        "PCA 要找到一组正交基使投影后方差最大（等价于重构误差最小）。"
        "在单位向量约束下最大化投影方差，用拉格朗日乘子法可得 X^T X w = λw，"
        "即 w 是协方差矩阵的特征向量、λ 是对应特征值。按特征值从大到小取前 d' 个即主成分。"
        "实践步骤：标准化 → 协方差矩阵（实现用 SVD 更稳）→ 按累计方差贡献率选个数（常取 85%~95%）。"
        "PCA 是无监督的，最大化方差未必有利于分类，此时 LDA 通常更合适。",
    ),
    (
        "过拟合与欠拟合",
        "过拟合表现为训练误差远低于验证误差：解决办法是增加数据、正则化（L1/L2）、"
        "降低模型复杂度、早停、Dropout。"
        "欠拟合表现为训练误差本身就高：应提高模型复杂度、增加特征、减小正则化强度。"
        "用学习曲线可以直观区分两者。",
    ),
    (
        "为什么不能在测试集上调参",
        "反复用测试集评估并据此选模型，测试集就在事实上变成了验证集，"
        "模型选择过程会向测试集过拟合，报出的泛化误差偏乐观，上线后达不到。"
        "正确做法是用验证集或交叉验证调参，测试集只在最后用一次。"
        "同理，数据泄漏（例如在划分训练测试集之前就做归一化或填充缺失值）也会让评估失真。",
    ),
    (
        "评估指标与类别不平衡",
        "准确率在类别不平衡时会骗人：99% 负样本的数据集上全猜负类也有 99% 准确率。"
        "因此要看查准率（precision）、查全率（recall）与 F1。"
        "类别不平衡时应优先看 P-R 曲线而非 ROC：ROC 的横轴假正例率以负类总数为分母，"
        "负类占绝大多数时分母很大，假正例率增长被稀释，曲线看起来依然很好。",
    ),
    (
        "偏差-方差分解",
        "泛化误差可分解为偏差、方差与噪声之和。模型越复杂偏差越低但方差越高，"
        "越简单则相反，因此存在使总误差最小、泛化最好的复杂度，这也是欠拟合与过拟合的分界。"
        "Bagging 主要降方差，Boosting 主要降偏差。",
    ),
    (
        "交叉验证与分层",
        "k 折交叉验证把数据分成 k 份轮流做验证集，比单次留出法更稳定。"
        "分层 k 折（StratifiedKFold）保证每折的类别比例与整体一致，"
        "避免某折里某类样本极少甚至缺失，在类别不平衡时尤其关键。"
        "留一法（LOO）每次只留一个样本做验证，偏差小但方差大且计算昂贵。",
    ),
    (
        "线性回归与正则化",
        "线性回归用最小二乘求解，闭式解 w = (X^T X)^{-1} X^T y。"
        "当特征近似线性相关或特征数多于样本数时 X^T X 接近奇异，闭式解不稳定甚至不存在。"
        "岭回归（L2）让权重整体收缩但不为零，能缓解共线性；"
        "Lasso（L1）在零点不可导，容易把部分权重压到恰好为零，产生稀疏解，可用于特征选择。",
    ),
    (
        "逻辑回归为什么用交叉熵",
        "交叉熵与 Sigmoid 搭配时梯度形如「预测值减真实值」，误差越大梯度越大，收敛快。"
        "若用均方误差，梯度中会多出 Sigmoid 的导数因子，在输出饱和时趋近于零，"
        "出现梯度消失、收敛缓慢。逻辑回归输出的是对数几率，可解释为概率。",
    ),
    (
        "训练集与测试集为什么要划分",
        "直接用训练误差估计泛化误差会严重偏乐观，因为模型可以记住训练数据。"
        "因此必须留出模型没见过的数据评估。常见做法：留出法（简单，但评估结果依赖划分）、"
        "k 折交叉验证（更稳定）、自助法（有放回采样，适合小数据集，约 36.8% 样本未被抽到，"
        "可用于包外估计）。划分时要注意保持类别比例（分层）。",
    ),
]


def _tokenize(text: str) -> list[str]:
    """分词：优先用 jieba（与 course-rag 一致），没有则退化为字符二元组。

    字符二元组对中文短文本检索已经够用（BM25 只看词项是否共现），
    这样内置知识库就**不依赖任何第三方分词库**也能工作。
    """
    text = (text or "").lower()
    try:
        import jieba

        jieba.setLogLevel(60)
        tokens = [t.strip() for t in jieba.cut_for_search(text) if t.strip()]
        if tokens:
            return tokens
    except Exception:
        pass
    cleaned = "".join(ch for ch in text if not ch.isspace() and ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    if len(cleaned) <= 2:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


class BuiltinKnowledgeBase:
    """内置迷你知识库，接口与 course-rag 的检索器保持一致。

    只依赖标准库（jieba / rank_bm25 缺失时自动降级），不引入任何外部服务，
    保证「单独 clone 也能跑」。
    """

    def __init__(self) -> None:
        from ..schema import KBChunk

        self._chunks = [
            KBChunk(
                chunk_id=f"builtin-{i}",
                doc_id="builtin",
                source_path="内置迷你知识库",
                text=body,
                section_path=title,
                heading=title,
                chunk_index=i,
            )
            for i, (title, body) in enumerate(BUILTIN_KB)
        ]
        self._bm25 = None
        self._build()

    def _build(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self._bm25 = None
            return
        corpus = [_tokenize(c.text) or ["<empty>"] for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(self, query: str, top_k: int = 5, **kwargs: Any):
        """返回 (hits, trace)，与 course-rag 的 HybridRetriever 接口兼容。"""
        from ..schema import KBHit, KBTrace

        trace = KBTrace(query=query)
        if self._bm25 is None:
            # 没有 rank_bm25：退化为字符共现计数，保证仍能返回结果
            scored = [
                (sum(1 for t in _tokenize(query) if t in c.text), i)
                for i, c in enumerate(self._chunks)
            ]
            order = [i for score, i in sorted(scored, reverse=True) if score > 0][:top_k]
            hits = [
                KBHit(chunk=self._chunks[i], score=1.0, channel="sparse", rank=r)
                for r, i in enumerate(order, start=1)
            ]
        else:
            tokens = _tokenize(query)
            scores = self._bm25.get_scores(tokens) if tokens else [0.0] * len(self._chunks)
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            hits = [
                KBHit(
                    chunk=self._chunks[i],
                    score=round(float(scores[i]), 4),
                    channel="sparse",
                    rank=rank,
                )
                for rank, i in enumerate(order, start=1)
                if scores[i] > 0
            ]

        trace.sparse_hits = len(hits)
        trace.fused_hits = len(hits)
        trace.final_hits = len(hits)
        return hits, trace
