# 课程 FAQ 与常见问题

本文件汇总《机器学习导论》课程中同学最常提出的问题，按「环境配置 / scikit-learn 用法 /
概念辨析 / 作业与考试 / 求助渠道」五类整理。所有答案都给出可直接操作的命令或代码。

## 一、环境配置

Q: pip 安装 scikit-learn 时长时间卡住，最后报 `Read timed out`，怎么办？
A: 这是默认源访问慢导致的。换成国内镜像即可：
`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn`。
想永久生效可以执行 `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`，
之后直接 `pip install scikit-learn`。conda 用户可在 `~/.condarc`（Windows 为
`C:\Users\<用户名>\.condarc`）中把 `channels` 换成清华 TUNA 或中科大 USTC 镜像；
只是偶发超时也可以先试 `pip install --timeout 60 --retries 5 scikit-learn`。

Q: 安装时报 `Microsoft Visual C++ 14.0 or greater is required`，或者要现场编译源码？
A: 说明当前 Python 版本没有对应的预编译 wheel，pip 只能尝试本地编译。三个办法：
优先用 conda 安装（`conda install scikit-learn`）；把 Python 换成 3.10 或 3.11 后重装；
确认 pip 已升级（`python -m pip install -U pip`）。不建议为了「新」而使用过新的 Python 版本。

Q: 明明装过 sklearn，却报 `ModuleNotFoundError: No module named 'sklearn'`？
A: 几乎总是「装到了另一个解释器/环境」。在报错的那个环境里执行
`import sys; print(sys.executable)`，再用同一个解释器安装：`python -m pip install scikit-learn`。
Jupyter 里可以用 `!{sys.executable} -m pip install scikit-learn`，保证装到当前内核所在环境。
另外可用 `pip -V` 查看 pip 属于哪个目录，用
`python -c "import sklearn; print(sklearn.__file__)"` 查看实际导入路径，
两者目录不一致就说明装的不是同一个环境。

Q: Jupyter 里选了内核，但代码跑的还是 base 环境，怎么给虚拟环境注册内核？
A: 先激活目标环境，安装 ipykernel，再注册：
`pip install ipykernel`，然后
`python -m ipykernel install --user --name ml --display-name "Python (ml)"`。
重启 Jupyter 后，在 Kernel → Change kernel 中选择「Python (ml)」即可。
用 `jupyter kernelspec list` 可以查看已注册的内核，确认注册是否成功；
若列表里出现多个同名项，可用 `jupyter kernelspec remove <内核名>` 清理。

Q: 画图时中文标题变成方框，还提示 `Glyph ... missing from current font`，怎么解决？
A: matplotlib 默认字体不含中文，需要显式指定：

```python
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]        # Windows；macOS 可用 "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False          # 正常显示负号
```

Linux 服务器上可换成 `["WenQuanYi Zen Hei"]` 或 `["Noto Sans CJK SC"]`。

Q: conda 和 pip 混着用，环境报依赖冲突，甚至 import 就崩，怎么办？
A: 同一个环境里尽量不要混用两个包管理器：二进制科学计算包优先用 conda 安装，
conda 里没有的纯 Python 包再用 pip。若环境已经混乱，最省时间的做法是重建：
`conda env remove -n ml` 后按课程要求重新 `conda create -n ml python=3.11 ...`。

## 二、scikit-learn 用法

Q: `fit`、`predict`、`transform` 到底分别在做什么？
A: `fit` 从数据中**估计参数**（例如标准化的均值与方差、线性模型的权重）；
`transform` 用已估计的参数**转换数据**（例如把特征标准化）；`predict` 输出**预测结果**。
一个估计器通常只有其中一部分方法：`StandardScaler` 有 `fit/transform`，
`LinearRegression` 有 `fit/predict`，`Pipeline` 把它们串起来统一调用。

Q: `fit_transform` 和 `transform` 有什么区别？为什么测试集只能用 `transform`？
A: `fit_transform` = 先在数据上 `fit` 再 `transform`；`transform` 只做转换、不重新估计参数。
若对测试集调用 `fit_transform`，测试集的均值方差会「泄漏」进预处理，
此时评估结果偏乐观，而且线上单条样本无法复现。正确顺序是：训练集 `fit_transform`，
测试集 `transform`；用 `Pipeline` 可以自动保证这一点。

Q: `random_state` 有什么作用？设成同一个数字结果就一定一样吗？
A: 它固定随机数种子，让划分数据、初始化、抽样等随机过程可复现。要得到完全一致的结果，
还需要数据、代码、库版本和参数都相同；极少数求解器仍可能有末位数值差异。
不设置时每次运行结果不同，这会让实验报告的数字无法核对，因此课程作业一律要求写明 `random_state`。

Q: `Pipeline` 有什么用？怎么给管道里的模型设置参数？
A: 它把预处理与模型串成一个整体，从而避免数据泄漏、方便交叉验证与调参。
参数名写成「步骤名 + 双下划线 + 参数名」，例如
`Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])` 对应
`param_grid = {"clf__C": [0.1, 1, 10]}`。若要查合法参数名，用 `pipe.get_params().keys()`。
注意 `make_pipeline` 会自动命名（类名小写，如 `standardscaler`、`logisticregression`），
想用简短的 `clf__C` 就显式写 `Pipeline([...])`。

Q: `predict` 和 `predict_proba` 有什么区别？默认阈值是多少？
A: `predict` 直接返回类别标签；`predict_proba` 返回各类别的概率，列的顺序与 `model.classes_` 一致，
二分类时通常取第 1 列作为正类概率。`predict` 内部相当于用 0.5 做阈值，
要改成 0.3 或 0.7 必须自己写 `(proba >= t).astype(int)`。
多分类时 `predict_proba` 返回形状为 `(样本数, 类别数)` 的矩阵，
`predict` 等价于对每行取 `argmax`；想实现「最高概率不足 0.6 就判为不确定」也必须自己写。

Q: 训练好的模型怎么保存，下次直接用？
A: 用 joblib 保存整个管道，而不是只保存模型对象：

```python
from joblib import dump, load
dump(pipe, "model.joblib")          # pipe 是已 fit 的 Pipeline
pipe2 = load("model.joblib")
print(pipe2.predict(X_new))
```

只保存 `pipe.named_steps["clf"]` 会导致下次预测时缺少标准化步骤，结果完全错误。
另外建议记录 scikit-learn 版本：不同大版本的持久化格式不保证兼容，
换环境加载失败时用相同版本重新训练即可。

Q: 传给模型的 `X` 用 DataFrame 还是 numpy 数组？列的顺序有影响吗？
A: 两者都能接受，DataFrame 便于对照列名排查问题。但**列的顺序与含义必须与训练时一致**，
否则预测结果会静默出错。乳腺癌数据集的列名含空格（如 `mean radius`），
用 DataFrame 时要写 `data["mean radius"]`，不要写成 `data.mean radius`。

Q: `cross_val_score` 返回的一串数字怎么用？
A: 它返回每一折的分数数组，报告时应写成「均值 ± 标准差」，例如 `0.972 ± 0.008`。
只看均值会忽略波动：均值高但标准差很大，说明结果不稳定。需要同时拿到多个指标与耗时，
可以用 `cross_validate`。

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
print(f"{scores.mean():.4f} ± {scores.std():.4f}")
```

注意 `cross_val_score` 不指定 `scoring` 时用的是估计器自带的 `score`，
其含义随模型而变（分类器是准确率，回归器是 $R^2$），因此评分标准务必显式指定。

## 三、概念辨析

Q: 过拟合和欠拟合怎么区分？
A: 看训练分数与验证分数的关系。两者都低且接近 → 欠拟合（高偏差），
应增加模型复杂度或特征；训练分数很高、验证分数明显低、间隙大 → 过拟合（高方差），
应增加数据、加强正则化或简化模型。学习曲线是最直观的诊断工具。

Q: 参数和超参数有什么区别？
A: 参数是模型**从数据里学出来**的，例如线性回归的权重、神经网络的连接权重；
超参数是**训练前由人指定**的，例如岭回归的 `alpha`、决策树深度、学习率、`k` 折的 `k`。
所谓「调参」调的是超参数，评价标准必须是验证集或交叉验证的结果。

Q: 分类和回归有什么区别？
A: 看输出空间：回归预测连续值（房价、温度），分类预测离散类别（是否患病）。
逻辑回归虽是分类器，但输出概率，可用阈值转成类别；
把回归结果硬套阈值会丢掉概率信息，评价指标也不同（回归用 MSE、$R^2$，分类用准确率、查准率/查全率（precision/recall））。

Q: 监督学习和无监督学习有什么区别？
A: 监督学习的数据带标签，任务是预测标签（分类、回归）；
无监督学习没有标签，任务是从数据结构中找规律（聚类、降维、密度估计）。
K-means 与 PCA 属于无监督，决策树与对数几率回归属于监督。

Q: 归一化和标准化有什么区别？该用哪个？
A: 归一化（min-max）把特征线性映射到 $[0,1]$：$x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}$；
标准化（z-score）把特征变成均值 0、标准差 1：$x' = \frac{x-\mu}{\sigma}$。
两者都消除量纲影响：归一化结果有界但对异常值敏感，标准化更适合近似高斯分布、含少量异常值的数据。
需要缩放的是 KNN、SVM、K-means、带正则化的线性模型与神经网络；
决策树、随机森林、GBDT 这类基于分裂点比较的模型不需要缩放。

Q: 为什么不能在测试集上调参？
A: 因为调参本身就是「用数据做选择」。反复用测试集比较模型，测试集就在事实上变成了验证集，
模型选择过程会向测试集过拟合，报出的分数偏乐观，上线后达不到。
正确流程是：训练集训练 → 验证集（或交叉验证）选超参数 → 测试集只评估一次。

Q: 训练误差很低、测试误差很高，一定是过拟合吗？
A: 不一定。还要排除几种可能：测试集与训练集分布不同（例如按时间切分的数据）；
存在数据泄漏（测试样本信息混入了训练集）；测试集太小导致指标抖动；
标签本身有噪声。排查顺序建议先查数据划分与泄漏，再看模型复杂度。

## 四、作业与考试

Q: 作业和实验报告的提交格式是什么？
A: 平时作业提交单个 PDF，命名 `学号_姓名_作业N.pdf`；
实验提交 zip 压缩包，命名 `学号_姓名_实验N.zip`，包内包含报告 PDF 与源码
（`.ipynb` 或 `.py`）。一律通过课程平台提交，不接受邮箱提交；
报告中必须写明运行环境版本与 `random_state`。

Q: 实验报告的评分点是什么？
A: 结果正确性 40%、分析与思考题 30%、图表与报告规范 15%、代码质量 15%。
最容易丢分的是「只贴代码不解释」与「图没有编号、坐标轴没有标签」；
数字必须能在自己代码的输出中找到。

Q: 期末考试是什么题型？范围有哪些？
A: 选择题约 20 分、简答题约 30 分、计算与推导题约 30 分、综合设计题约 20 分。
闭卷考试，可携带一张 A4 手写公式纸（正反面均可）。
重点是线性模型、决策树、模型评估与选择、支持向量机的核心公式与概念，
包括最小二乘闭式解与岭回归解、对数几率回归的交叉熵梯度、信息增益与基尼指数、
查准率/查全率与 ROC/AUC 的计算、偏差-方差分解、正则化与核技巧的作用。
复习建议：先把每章的「关键公式速查」表默写一遍，再动手复现对应实验的最小代码，
最后用往年的简答题自测能否不看笔记讲清楚「为什么」。

Q: 作业迟交了怎么办？
A: 每迟交 1 天扣本次作业或实验成绩的 10%，超过 7 天不再受理。
因病假、出差等特殊情况无法按时提交，请提前联系助教说明并附证明，经同意后可延期。

Q: 可以用网上找的真实数据集做实验吗？可以用 AI 工具写代码吗？
A: 可以使用真实数据集，但必须满足：来源与获取方式写清楚（数据文件过大时提供下载链接或读取脚本）、
预处理过程完整说明、原实验要求的全部指标都要给出，且结果可复现。
AI 工具可用于理解概念与调试报错，但代码必须自己读懂并改写，
报告中要能解释每一行；直接提交生成的内容按抄袭处理。

## 五、求助渠道

Q: 课程的答疑时间与地点是什么？
A: 每周二、四 15:00–17:00，实验楼 A305（如有调整以课程平台公告为准）。
建议带着具体报错与已尝试的方法来，不要只问「跑不出来怎么办」。

Q: 课程群怎么加入？
A: QQ 群 887766554（群名「机器学习导论」），申请时备注「学号 + 姓名」。
群内提问请贴文字版报错与最小可复现代码片段，不要只发截图，也不要刷屏。

Q: 助教邮箱是什么？多久回复？
A: 助教邮箱 `mlcourse.ta@example.edu.cn`，一般在 24 小时内回复（周末顺延）。
邮件主题写成「实验N-学号-姓名-问题摘要」，正文中附上环境版本、数据形状、完整报错与相关代码。

Q: 代码报错时，应该先做什么再求助？
A: 第一步看 traceback 的**最后一行**，那是真正的错误类型；
第二步把报错信息原文复制到搜索引擎或官方文档中检索；
第三步构造最小可复现例子（把数据缩小到几行、删掉无关代码），确认问题仍然出现；
第四步再向助教或课程群求助，并附上环境版本、`X.shape`、完整报错与已尝试的修改。
这条流程本身就是机器学习工程师的日常工作方式。
