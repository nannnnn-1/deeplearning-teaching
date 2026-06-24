/* 深度学习路线 · 数据文件 */
var data = [
{
  id: "l1", t: "第一层：地基（必须掌握）", c: "#c25e00", i: "🏗️", inCourse: false,
  s: [
    {t: "数学基础", desc: "深度学习的语法——不是背公式，而是知道每个公式在代码里对应什么",
      items: [
        {n: "线性代数", d: "向量/矩阵/张量是 PyTorch Tensor 的数学本质。特征分解和 SVD 是 PCA 的底层。", course: null,
         note: "【直觉】把数据想象成一块豆腐——标量是一粒豆子，向量是一串豆子，矩阵是一块豆腐，张量是一摞豆腐。深度学习中所有的输入、参数、中间结果都是张量之间的变换。<br><br>【技术】核心操作：矩阵乘法（全连接层的本质）、转置、求逆、特征值分解。SVD 将任意矩阵分解为 UΣVᵀ，用在 PCA 降维、推荐系统的矩阵分解中。PyTorch 中 torch.matmul、torch.linalg.eig 直接对应这些操作。<br><br>【应用】图像是 (B,C,H,W) 四维张量，文本经过 Embedding 变成 (B,L,D) 三维张量。理解张量的 shape 变换（reshape/permute）是 debug 模型的基本功。"},
        {n: "微积分", d: "链式法则 = 反向传播。梯度 = 参数更新的方向。Hessian 矩阵影响收敛速度。", course: null,
         note: "【直觉】梯度就是「往哪个方向走 loss 降得最快」。想象你在一座雾山上找最低点——你看不清全貌，但能感知脚下的坡度，梯度告诉你下一步该往哪迈。<br><br>【技术】链式法则：∂L/∂x = ∂L/∂y · ∂y/∂x，这是反向传播的数学根基。偏导数组成梯度向量 ∇L。Hessian 矩阵（二阶导）描述曲率——曲率大意味着梯度变化剧烈，学习率要小。<br><br>【应用】PyTorch 的 autograd 在每次 forward 时构建计算图，backward 时自动沿图反向传播求梯度。理解这个机制才能看懂 loss.backward() 到底干了什么。"},
        {n: "概率与统计", d: "概率分布是生成模型（VAE/GAN）的语言。KL 散度衡量两个分布的距离。", course: "第11节",
         note: "【直觉】模型输出的不是「这是猫」，而是「90% 概率是猫，8% 是狗，2% 是其他」。深度学习本质是概率建模——给定输入 x，输出 y 的条件概率分布 P(y|x)。<br><br>【技术】交叉熵 = 真实分布与预测分布的差异（分类问题的默认损失函数）。KL 散度衡量两个分布的距离（VAE 的核心损失项）。贝叶斯定理 P(θ|D) ∝ P(D|θ)P(θ) 是整个统计学习的理论框架。<br><br>【应用】VAE 用 KL 散度让编码分布逼近标准正态分布；GAN 的判别器输出一个概率；Diffusion 模型每一步去噪都在拟合一个高斯分布。交叉熵是 90% 分类任务的首选损失函数。"},
        {n: "优化理论", d: "凸优化是理解 SGD/Adam 收敛性的直觉。拉格朗日乘子用于约束优化。", course: null,
         note: "【直觉】训练模型就是「在一张凹凸不平的 loss 曲面上找最低点」。凸函数只有一个全局最低点（像碗），非凸函数有很多局部最低点（像山脉）。深度学习是非凸优化，但我们发现 SGD 的噪声反而有助于跳出不好的局部最优点。<br><br>【技术】梯度下降：θ_{t+1} = θ_t - η∇L。学习率 η 太大震荡不收敛，太小收敛太慢。学习率调度（Cosine/Warmup/Step）就是在不同阶段用不同的 η。动量法引入惯性项，Adam 引入自适应学习率。<br><br>【应用】理解优化器选择直接影响训练效果：小模型/简单任务用 SGD+Momentum，大模型/复杂任务用 AdamW。学习率 warmup 几乎是 Transformer 训练的标配。"}
      ]},
    {t: "编程与工具", desc: "工欲善其事，先装好工具链",
      items: [
        {n: "Python + NumPy", d: "NumPy 的广播和矩阵运算是 PyTorch 的前身。", course: null,
         note: "【直觉】NumPy 就是没有 GPU 和自动求导的 PyTorch。你用 NumPy 写的矩阵运算，改几行就能跑在 PyTorch 上。<br><br>【技术】广播机制（broadcasting）：不同 shape 的数组自动扩展对齐——这是 PyTorch 行为的重要基础。花式索引（fancy indexing）、切片、reshape 是数据预处理的核心操作。<br><br>【应用】数据加载和预处理大量依赖 NumPy。理解 np.ndarray 的 stride 和内存布局有助于理解 PyTorch Tensor 的 contiguous() 问题。"},
        {n: "PyTorch", d: "研究首选框架。计算图、自动微分、GPU 加速——后续每节课都用。", course: "第2节起",
         note: "【直觉】PyTorch 是深度学习界的 Python——语法直觉、调试友好、生态丰富。相比 TensorFlow 的静态图，PyTorch 的动态图就像写普通 Python 代码，每一步都可以 print 和断点调试。<br><br>【技术】核心组件：torch.Tensor（GPU 加速的多维数组）、autograd（自动求导引擎）、nn.Module（模型基类）、DataLoader（数据加载）。torch.nn 提供了 Linear/Conv2d/Transformer 等开箱即用的层。<br><br>【应用】学术界 80%+ 论文用 PyTorch 实现。HuggingFace、TorchVision、TorchAudio 等生态让复现 SOTA 只需几十行代码。从研究到部署（TorchScript → ONNX → TensorRT）链路完整。"},
        {n: "GPU / CUDA", d: "GPU 的并行计算是深度学习速度的 100x 来源。理解 CUDA 核心概念即可。", course: "第1节",
         note: "【直觉】CPU 是几个数学教授（串行、低频、单任务强），GPU 是几千个小学生（并行、高频、单任务弱但量大）。矩阵乘法正好能拆成几千个独立的小运算，所以 GPU 做深度学习比 CPU 快 10-100 倍。<br><br>【技术】CUDA 是 NVIDIA 的并行计算平台。核心概念：线程层次（thread → block → grid）、共享内存、流处理器。不需要手写 CUDA，但要理解：数据在 CPU 和 GPU 之间搬运是瓶颈（PCIe 带宽 << 显存带宽），所以要尽量减少 .cpu() 和 .cuda() 的调用。<br><br>【应用】训练时 batch size 受限于显存大小。混合精度训练（AMP）用 FP16 省一半显存。推理时用 TensorRT 对 CUDA kernel 做算子融合加速。消费级 RTX 4090（24GB）可以训练 7B 参数的 LoRA 微调。"},
        {n: "Git / GitHub", d: "版本控制是团队协作和实验复现的基础。", course: null,
         note: "【直觉】用 Word 写论文你会保留「论文_v1.doc」「论文_v2_导师改.doc」「论文_final_真的final.doc」。Git 就是给代码做这件事，而且做得自动化、可追溯、能多人协作。<br><br>【技术】核心操作：commit（保存快照）、branch（分岔开发）、merge（合并回来）、push/pull（同步远程）。.gitignore 排除模型权重和大数据集。Git LFS 管理大文件。<br><br>【应用】clone 论文代码 → 创建新分支 → 修改实验 → 如果效果不好就 checkout 回去。GitHub Issues/PR 是开源社区协作的标准方式。"}
      ]}
  ]
},
{
  id: "l2", t: "第二层：机器学习基础", c: "#d49500", i: "🌱", inCourse: false,
  s: [
    {t: "经典机器学习", desc: "深度学习的前传——很多概念（损失函数、正则化、交叉验证）直接沿用",
      items: [
        {n: "监督学习", d: "线性回归/逻辑回归是神经网络的最简形式（1层）。SVM 是理解 Margin 和核方法的直觉。", course: null,
         note: "【直觉】监督学习就是「给标准答案，让模型学映射」。分类 = 判断图片是猫还是狗。回归 = 预测房价具体多少钱。关键区别：分类输出离散标签，回归输出连续数值。<br><br>【技术】线性回归 y=Wx+b 就是没有激活函数的单层神经网络。逻辑回归在它上面套一个 Sigmoid，输出概率。SVM 的核心思想是最大化分类间隔（Margin），引入核函数（Kernel）把低维不可分数据映射到高维使其可分——这是理解神经网络「深度 = 逐层提取特征」的钥匙。<br><br>【应用】逻辑回归到今天仍是工业界二分类问题的 baseline。SVM 在小样本高维数据（如基因表达）上仍有优势。理解这些经典方法有助于对比深度学习的收益来源——当数据量不够时，XGBoost 可能比深度学习更好。"},
        {n: "无监督学习", d: "K-Means 是聚类基准。PCA 是降维基础（Autoencoder 的线性版）。", course: null,
         note: "【直觉】无监督学习就是「不给标准答案，让模型自己找数据里的结构」。聚类 = 把相似的放一起。降维 = 把高维数据压扁但尽量保留信息。<br><br>【技术】K-Means：随机选 K 个中心 → 分配点到最近中心 → 重算中心 → 迭代。PCA：找到数据方差最大的方向（第一主成分），然后找与它正交的第二大方向，以此类推。数学上等于协方差矩阵的特征值分解。<br><br>【应用】PCA 用于数据可视化和噪声去除。Autoencoder 可以看作 PCA 的非线性升级版——如果 Autoencoder 只用线性激活，它就退化成 PCA。聚类用于用户分群、异常检测、数据的初步探索。"},
        {n: "概率图模型", d: "HMM 和 CRF 是序列标注的经典方法，理解它们有助于理解 CRF 在 Transformer 中的应用。", course: null,
         note: "【直觉】概率图模型用「图」来表示变量之间的依赖关系。有向图（贝叶斯网络）= 因果关系，无向图（马尔可夫随机场）= 相关关系。HMM 假设当前状态只依赖上一个状态，是一种「短视」但高效的序列建模。<br><br>【技术】HMM 的三要素：初始概率、转移概率、发射概率。三大问题用前向-后向算法和维特比算法求解。CRF 去掉了 HMM 的「观测独立」限制，能对整个序列同时建模。<br><br>【应用】语音识别最早用 HMM+GMM 实现，NLP 的序列标注（NER/POS）用 CRF。理解 HMM-CRF 的「序列依赖建模」思路，才能理解为什么 Transformer 的 Self-Attention（全局依赖）是质的飞跃。"}
      ]},
    {t: "核心概念", desc: "这些概念从经典 ML 到 DL 完全通用，是理解后续所有实验的根基",
      items: [
        {n: "损失函数 + 梯度下降", d: "损失 = 模型错了多少。梯度下降 = 沿着损失最陡的方向走下山。", course: "第2节",
         note: "【直觉】损失函数就是「考试扣分规则」——做错了扣多少分。梯度下降就是「根据扣分来调整学习方法」——哪个知识点错得多，就多花时间补哪个。<br><br>【技术】MSE 均方误差 = (1/n)Σ(y_pred - y_true)²（回归任务常用，注意除 n 取均值）。Cross-Entropy = -Σ y_true·log(y_pred)（分类任务常用）。梯度下降每一步更新：θ_new = θ_old - learning_rate × gradient。梯度的方向是 loss 上升最快的方向，所以取反方向。<br><br>【应用】选择损失函数就是告诉模型「什么算错」。MSE 对大误差惩罚重（平方项），适合需要严格控制偏差的场景。Cross-Entropy 配合 Softmax 在分类问题上梯度性质更好，不会出现「饱和」问题。"},
        {n: "学习率 + 正则化", d: "学习率 = 步长。正则化 = 不要让模型记答案，要让它学规律。", course: "第4节",
         note: "【直觉】学习率就是步伐大小——太大跨过最低点，太小走不到终点。正则化就是给模型一个「简单偏好」——能用简单规律解释的，就别用复杂的去死记硬背。<br><br>【技术】L1 正则化（Lasso）= |W| 的惩罚，导致稀疏解（很多参数变成 0），天然做特征选择。L2 正则化（Ridge/权重衰减）= W² 的惩罚，让参数变小但不为 0，抑制过拟合。Dropout 是神经网络的「随机裁员」——训练时每条数据随机丢弃一部分神经元。<br><br>【应用】学习率是深度学习最重要的超参数，没有之一。合理范围一般是 1e-5 到 1e-3。正则化强度的选择：如果训练 loss 低但验证 loss 高（过拟合）→ 加强正则化；如果训练 loss 和验证 loss 都高（欠拟合）→ 减弱正则化。"},
        {n: "过拟合 vs 欠拟合", d: "过拟合 = 背答案（训练好测试差）。欠拟合 = 没学会（训练差测试也差）。", course: "第4节",
         note: "【直觉】过拟合 = 学生背下了所有习题的答案编号，换一本题就不会了。欠拟合 = 学生连课本都没翻完，考试全靠蒙。<br><br>【技术】判断标准：过拟合 = train loss 很低但 val loss 在上升（两条曲线分叉）。欠拟合 = train loss 和 val loss 都很高且接近。解决过拟合：加数据、加正则化、降低模型复杂度、Early Stopping。解决欠拟合：加模型复杂度、减正则化、训练更久。<br><br>【应用】这是调参的核心诊断框架。每次看到训练曲线，第一反应应该是「现在是过拟合还是欠拟合？」而不是盲目调参数。Early Stopping = 当 val loss 不再下降就停止训练，简单有效的防过拟合手段。"},
        {n: "交叉验证 + 评价指标", d: "Accuracy = 整体准确率。Precision/Recall = 不冤枉 vs 不漏抓。F1 = 两者调和。", course: "第2-4节",
         note: "【直觉】准确率 99% 的癌症检测模型，如果它永远说「没病」，在癌症发病率 1% 的人群里准确率也是 99%——但这个模型毫无用处。评价指标决定了你「在乎什么」。<br><br>【技术】Accuracy = (TP+TN)/(TP+TN+FP+FN)，最直观但容易被类别不平衡欺骗。Precision = TP/(TP+FP)：「说是正类中有多少真对了」（不冤枉）。Recall = TP/(TP+FN)：「真实正类中被找出了多少」（不漏抓）。F1 = 2PR/(P+R)，精确率和召回率的调和平均。K 折交叉验证：把数据分 K 份，轮流用 K-1 份训练、1 份验证，取平均指标。<br><br>【应用】医疗诊断要高 Recall（宁可误诊也不要漏诊）。垃圾邮件过滤要高 Precision（宁可不拦截也不要误拦重要邮件）。竞赛/论文中常用 K-Fold 给出更可靠的指标估计。"}
      ]}
  ]
},
{
  id: "l3", t: "第三层：深度学习核心（必学主干）", c: "#0d7377", i: "🧠", inCourse: true,
  s: [
    {t: "神经网络基础", desc: "神经网络不是魔法，是矩阵运算 + 非线性激活的堆叠",
      items: [
        {n: "感知机与 MLP", d: "感知机 = 一个神经元。MLP = 很多神经元分层连接。激活函数 = 引入非线性，让模型拟合曲线。", course: "第2节",
         note: "【直觉】一个神经元 = 加权求和 + 过一次激活函数。一层神经元 = 同时做很多次加权求和。多层堆叠 = 逐层提取更抽象的特征。没有激活函数，100 层网络等价于 1 层（因为线性变换的复合还是线性变换），所以激活函数是「深度」有意义的前提。<br><br>【技术】MLP 的公式：h = σ(Wx + b)。W 是权重矩阵，b 是偏置，σ 是激活函数。常见激活函数：ReLU(x)=max(0,x)（简单高效，缓解梯度消失）；Sigmoid(x)=1/(1+e^{-x})（输出 0-1，适合二分类输出层）；GELU（Transformer 标配，平滑版 ReLU）。<br><br>【应用】MLP 是最基础的神经网络，现在主要用作 Transformer 中的 FFN（前馈网络）和分类任务的最后一层。理解 MLP 的矩阵乘法和激活函数的配合，是所有复杂架构的起点。"},
        {n: "反向传播", d: "从输出层倒着算梯度，逐层更新参数。PyTorch 的 autograd 自动完成这一切。", course: "第2节",
         note: "【直觉】你做了一道菜，尝了一口觉得太咸（计算 loss）。你倒推：盐放多了 → 因为盐勺大了 → 下次换小勺。逐层倒推「哪个环节出了多少问题」就是反向传播。<br><br>【技术】前向传播：输入 x，逐层计算直到输出 ŷ，算出 loss。反向传播：从 loss 开始，用链式法则逐层计算 ∂loss/∂W 和 ∂loss/∂b。PyTorch 的 autograd 在 forward 时构建动态计算图（每个操作的 grad_fn 记录了求导规则），调用 loss.backward() 时自动遍历计算图求梯度。<br><br>【应用】理解反向传播的 3 个实际意义：(1) 知道哪些操作可导（才能用深度学习框架），(2) 理解梯度消失/爆炸的根源（链式求导中梯度的连乘），(3) debug 时可以在任意层 hook 梯度，看梯度是否正常。"},
        {n: "参数初始化", d: "Xavier/He 初始化决定训练起点。初始化不好，梯度会消失或爆炸。", course: "第3节",
         note: "【直觉】训练神经网络就像从山顶往山谷下——起点（初始参数）决定了你从哪条路下山。好的初始化让你站在半山腰的开阔地，差的初始化把你扔到悬崖边（梯度爆炸）或沼泽里（梯度消失）。<br><br>【技术】全零初始化 → 所有神经元学到的特征一样（对称性问题）。高斯随机初始化 → 层数深时输出方差指数增长或衰减。Xavier 初始化（Glorot）：W ~ N(0, 2/(n_in + n_out))，适合 Sigmoid/Tanh。He 初始化：W ~ N(0, 2/n_in)，专为 ReLU 设计，现代 CNN/Transformer 普遍使用。<br><br>【应用】PyTorch 中 nn.Linear 默认用 He 初始化（kaiming_uniform_）。如果自己写 custom layer，注意 init 方式。训练不稳定/不收敛时，检查初始化是最基本的排查步骤之一。"},
        {n: "标准化", d: "BatchNorm = 让每一层输入分布稳定，训练更快更稳。LayerNorm = Transformer 的标配。", course: "第4节",
         note: "【直觉】每一层网络处理完数据后，数据的分布会偏移（内部协变量偏移）。标准化就是把偏移的分布拉回标准位置，让下一层看到的输入始终是「零均值、单位方差」——训练平稳多了。<br><br>【技术】BatchNorm（BN）：在一个 batch 内沿 batch 维度标准化（对每个 channel 算均值和方差），训练时用 batch 统计量，推理时用训练时累积的全局统计量。LayerNorm（LN）：沿特征维度标准化（对每个样本独立算），不受 batch size 影响，所以是 Transformer 的标配。InstanceNorm 用于风格迁移，GroupNorm 是 BN 和 LN 的折中。<br><br>【应用】CNN + BN + ReLU 是 CV 任务的经典组合。Transformer 必须用 LN（因为序列长度不固定，BN 不适用）。BN 对 batch size 敏感——batch 太小统计不准确，此时考虑换 GN。"}
      ]},
    {t: "优化算法", desc: "下山的路径选择——不同算法对应不同下山策略",
      items: [
        {n: "SGD + Momentum", d: "SGD = 每次只看一个样本的梯度。Momentum = 下山时有惯性，不来回震荡。", course: "第4节",
         note: "【直觉】SGD 每次只看一个小 batch 就迈步，所以走起来摇摇晃晃。Momentum 给更新加了惯性——如果前面几步都在往右走，就算当前这一步有点偏左，惯性也会让你继续往右。结果是路径更平滑，收敛更快。<br><br>【技术】SGD：θ = θ - η·g_t。SGD+Momentum：v_t = β·v_{t-1} + η·g_t，θ = θ - v_t。β ≈ 0.9 意味着当前更新中 90% 来自历史方向，10% 来自当前梯度。Nesterov 动量 = 先在历史方向上走一步，再算梯度——相当于「提前踩点」。<br><br>【应用】SGD+Momentum 在 CV 分类任务上仍然很强，尤其搭配 Cosine 学习率调度。现在大多数情况 AdamW 是首选，但了解 SGD 是理解更复杂优化器的基石。"},
        {n: "Adam / AdamW", d: "Adam = 自适应学习率（每个参数有自己的步长）。AdamW = 把权重衰减从梯度中分离，效果更好。", course: "第4节",
         note: "【直觉】Adam 就像给每个参数配了专属的学习率调节器——变化大的参数自动降速，变化小的参数自动提速。AdamW 改进了 Adam 的正则化实现——原来 Adam 把 L2 正则化混在梯度里一起做了自适应缩放，AdamW 把它们分开，解耦后泛化性更好。<br><br>【技术】Adam 维护两个变量：一阶矩 m_t（梯度的指数移动平均，方向），二阶矩 v_t（梯度平方的指数移动平均，尺度）。更新：θ = θ - η·m_t/(√v_t + ε)。AdamW 把权重衰减单独做：θ = θ - η·(m_t/(√v_t + ε) + λ·θ)。<br><br>【应用】AdamW 是当前大模型训练的事实标准。几乎所有 LLM（GPT/LLaMA）和扩散模型都用 AdamW。推荐超参：lr=1e-4 到 3e-4，β1=0.9，β2=0.95-0.999，weight_decay=0.1。"},
        {n: "学习率调度", d: "Step = 固定步长衰减。Cosine = 余弦曲线平滑衰减。Warmup = 先慢后快，防止初期震荡。", course: "第4节",
         note: "【直觉】训练初期模型啥都不会，步子大了容易摔（需要在初始参数附近仔细搜索）。训练后期模型接近最优，步子要小（精细调参）。Warmup = 先小步慢走热身，再大步前进，最后小步精调。<br><br>【技术】StepLR：每 N 个 epoch，lr 乘以 γ（如每 30 epoch 降 10 倍）。CosineAnnealing：lr 沿余弦曲线从初始值平滑衰减到接近 0，周期性的升降有助于跳出局部最优。Warmup：前 K 步 lr 从 0 线性增加到目标值。Cosine + Warmup 是 Transformer 训练的标配。<br><br>【应用】不同任务调 lr schedule 的经验：分类任务用 Cosine；GAN 训练一般固定 lr 或用 Step；LLM 预训练必用 Warmup（否则训练初期 loss 会 spike）。画出 lr 曲线确认调度符合预期是个好习惯。"},
        {n: "二阶优化（了解）", d: "L-BFGS 用 Hessian 矩阵信息，收敛更快但内存占用大。了解即可。", course: null,
         note: "【直觉】一阶优化只看坡度（梯度），二阶优化还看坡度的变化率（曲率）。知道曲率就知道踩多大力不会冲过头。但曲率矩阵（Hessian）是 n×n 的（n 为参数量），对深层网络完全不可行。<br><br>【技术】牛顿法：θ = θ - H^{-1}·g，需要求逆 Hessian 矩阵 O(n³) 不可行。L-BFGS 用历史梯度和参数变化近似 H^{-1}，只需存最近 m 步的信息 O(mn)。收敛速度：二阶 > 一阶动量 > 纯 SGD，但每步成本更高。<br><br>【应用】L-BFGS 在风格迁移（Neural Style Transfer）和少数小规模问题上有效。大规模深度学习训练中基本不用，了解它是为了理解「为什么 Adam 的近似二阶行为效果这么好」。"}
      ]},
    {t: "正则化与泛化", desc: "让模型在新数据上也能表现好，而不是只在训练集上表演",
      items: [
        {n: "Dropout", d: "训练时随机关掉一部分神经元，强迫每个神经元独立有用。测试时全部启用。", course: "第4节",
         note: "【直觉】一群学生一起做小组作业，总有几个人偷懒靠别人。Dropout 就是每次随机让一部分人「请假」，剩下的必须自己会做——久而久之每个人都真正学会了，不再依赖特定队友。<br><br>【技术】训练时：每个神经元以概率 p（如 p=0.5）被「丢弃」（输出置 0），保留的神经元输出除以 (1-p) 做补偿。测试时：所有神经元启用，但不需要补偿（因为期望已经对齐）。Dropout 等价于训练大量子网络并在测试时做 ensemble。<br><br>【应用】全连接层常用 p=0.5，卷积层常用 p=0.1-0.2（因为卷积本身有参数共享，不需要太强的 Dropout）。Transformer 中通常只在 FFN 和 Attention 后加少量 Dropout（p=0.1）。数据量很大时 Dropout 的收益会减小。"},
        {n: "数据增强", d: "翻转/旋转/裁剪图片，让模型见过更多变体，提升泛化。", course: "第3节",
         note: "【直觉】与其让模型死记「这张猫照片」是猫，不如把同一张照片翻转、调色、裁剪后都告诉它「这还是猫」。模型被迫关注猫的本质特征（眼睛、耳朵、胡须）而不是像素位置。<br><br>【技术】CV 增强：随机翻转（HorizontalFlip）、旋转（Rotation）、裁剪（RandomCrop/ResizedCrop）、色彩抖动（ColorJitter）、高斯噪声。NLP 增强：同义词替换、回译（translate-then-translate-back）、随机删除。AutoAugment/RandAugment 自动搜索最佳增强策略。<br><br>【应用】数据增强是提升泛化性最直接有效的手段。训练 ImageNet 必用 RandomResizedCrop + HorizontalFlip。小数据集上 RandAugment 能带来 2-5% 的提升。Mixup 和 CutMix 是进阶增强，把两张图及其标签按比例混合。"},
        {n: "Mixup / CutMix", d: "把两张图混合/拼接，让模型学到更平滑的决策边界。", course: null,
         note: "【直觉】传统增强只改一张图，Mixup 把两张图「搅拌」在一起——图 A 的 70% + 图 B 的 30%，标签也按同样比例混合。这强迫模型学习「这张图 70% 像猫、30% 像狗」——决策边界变得更平滑、更鲁棒。<br><br>【技术】Mixup：x̃ = λx_A + (1-λ)x_B，ỹ = λy_A + (1-λ)y_B，λ ~ Beta(α,α)。CutMix：从图 A 切一块，用图 B 的对应区域补上，标签按面积比例混合。两者都让模型看到训练分布之外的「中间态」样本。<br><br>【应用】Mixup/CutMix 在 ImageNet 上是默认操作，可稳定提 1-3% 准确率。在对抗鲁棒性（防御对抗样本）上也有显著效果。NLP 领域的 Mixup 变体将混合操作做在 Embedding 空间。"},
        {n: "权重衰减", d: "L2 正则化 = 让参数变小，模型更简单，不易过拟合。", course: "第4节",
         note: "【直觉】权重衰减就是对「参数太大」这件事收税——模型如果用一个很小的参数就能拟合好数据，就不会去用大的。小的参数意味着输出对输入的微小变化不敏感，泛化性更好。<br><br>【技术】L2 正则化：loss_total = loss_original + λΣW²。求导得梯度多了一项 2λW，所以每步更新会额外减去 2λW——参数不断被拉向 0。AdamW 与 Adam 的关键区别：权重衰减不是加在 loss 里再送入 Adam 的自适应机制，而是直接从参数中减去。<br><br>【应用】权重衰减系数常用 1e-4 到 1e-2。现代 LLM 训练用 AdamW 的 weight_decay 参数（如 LLaMA 用 0.1）。如果模型参数明显比数据集大（容易过拟合），增大权重衰减。"}
      ]},
    {t: "经典网络架构演进", desc: "理解深度学习史的捷径：看网络怎么从浅变深、从卷积到注意力",
      items: [
        {n: "CNN 开山：LeNet -> AlexNet -> VGG -> ResNet", d: "LeNet 证明了 CNN 可行。AlexNet 证明了 GPU 加速可行。VGG 证明深有用。ResNet 证明残差连接让深网络可训练。", course: "第2-3节",
         note: "【直觉】CNN 的发展史就是人类「怎么让网络更深」的探索史——每多一层，网络能看到更高层、更抽象的特征。边缘（浅层）→ 纹理 → 部件 → 物体（深层）。但直接堆层会遇到梯度消失问题，直到 ResNet 的「跳跃连接」解决了它。<br><br>【技术】LeNet-5（1998）：Conv→Pool→Conv→Pool→FC，手写数字识别。AlexNet（2012）：5 层 Conv + 3 层 FC + ReLU + Dropout + 双 GPU，ImageNet 冠军，深度学习爆发的导火索。VGG（2014）：全用 3×3 小卷积核，证明了简单的深度（19 层）就有效。ResNet（2015）：引入残差连接 F(x)+x（输入直接加到输出），让 152 层网络可训练，至今仍是最广泛使用的骨干网络。<br><br>【应用】ResNet-50/101 是 CV 任务的通用 baseline。残差连接的思想影响了后续几乎所有架构：Transformer 的残差连接 + LayerNorm，DenseNet 的密集连接，U-Net 的跳跃连接。理解 ResNet 才能理解为什么深层网络能训练。"},
        {n: "RNN 家族", d: "RNN 有记忆但梯度易消失。LSTM 用门控解决。GRU 简化 LSTM。Seq2Seq 是机器翻译的基础。", course: "第8节",
         note: "【直觉】读一本小说，读到第 100 页时需要记得第 1 页的人物设定——这就是 RNN 想做的事。但普通 RNN 读到后面就忘了前面（梯度消失），LSTM 用「遗忘门」「输入门」「输出门」三个开关，自主决定记住什么、忘掉什么。<br><br>【技术】RNN 公式：h_t = tanh(W_h·h_{t-1} + W_x·x_t)，核心问题：反向传播时梯度要穿过时间步，连乘导致指数衰减（梯度消失）。LSTM：f_t = σ(W_f·[h_{t-1}, x_t])（遗忘门），i_t = σ(W_i·[h_{t-1}, x_t])（输入门），o_t = σ(W_o·[h_{t-1}, x_t])（输出门），c_t = f_t⊙c_{t-1} + i_t⊙tanh(...)（细胞状态）。GRU 合并了隐藏状态和细胞状态，只用 2 个门，参数更少。<br><br>【应用】RNN 曾是 NLP 的主力（机器翻译、语音识别），但 Transformer 出现后被快速替代。如今 RNN 仍有价值：(1) 新架构 Mamba（状态空间模型）是 RNN 的进化，(2) 实时处理（流式 ASR）无法用双向 Attention，(3) 理解 RNN→LSTM→Attention 的演进有助于理解「高效序列建模」的设计思想。"},
        {n: "Self-Attention / Transformer", d: "Attention = 看句子中每个词和其他词的关系。Transformer = 纯 Attention 架构，现代 DL 分水岭。", course: "第8-9节",
         note: "【直觉】读到「银行」一词，你得根据上下文判断它是「商业银行」还是「河岸」。Attention 机制让每个词去看句子中所有其他词，收集相关信息来更新自己的表示。Transformer 把 RNN 的顺序处理改成了「所有词同时互相看」——就像从逐字阅读变成了一眼扫全页。<br><br>【技术】Self-Attention 公式：Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V。Q（查询）=「我在找什么」，K（键）=「我有什么信息」，V（值）=「我的实际内容」。除以 √d_k 防止点积过大导致 softmax 梯度饱和。Multi-Head Attention = 用多组 QKV 从不同角度关注。位置编码（Positional Encoding）弥补 Attention 缺失的「顺序信息」。<br><br>【应用】Transformer 是现代深度学习的「通用骨干」——NLP（BERT/GPT）、CV（ViT/Swin）、语音（Whisper）、多模态（CLIP）、代码生成（Codex）、蛋白质预测（AlphaFold）。它就是 2017 年之后整个领域的「操作系统」。"},
        {n: "Vision Transformer", d: "ViT = 把图片切成 Patch，当句子处理。Swin = 引入窗口机制，降低计算量。", course: "第10节",
         note: "【直觉】CNN 看图片是逐层往上看（从边缘到物体），ViT 看图片是「一眼看全局」——把图片切成 16×16 的小块，每块当做一个「词」，让这些 patch 互相做 Attention。Swin Transformer 在此基础上加了「窗口」限制——只在局部窗口内做 Attention，大幅降低计算量，同时通过窗口偏移让信息跨窗口流通。<br><br>【技术】ViT：image → split into patches → linear projection → + position embedding → Transformer Encoder → classification head。关键局限：需要大量数据预训练（JFT-300M）才能超过 CNN。Swin：分层结构（像 CNN 的逐层降采样）+ 窗口内 Attention + Shifted Window 跨窗口信息交换，计算量从 O(N²) 降到 O(N)。<br><br>【应用】ViT 在 ImageNet 上用 MAE 自监督预训练后能超越 CNN。Swin Transformer 是 COCO 检测和 ADE20K 分割的 backbone。ConvNeXt 反其道而行——「把 CNN 做得像 Transformer」，说明两种范式的边界在融合。"}
      ]}
	  ]
	},
	{
	  id: "l4", t: "第四层：计算机视觉（CV）", c: "#2a9d8f", i: "👁️", inCourse: true,
	  s: [
	    {t: "图像分类", desc: "给图片贴标签——最基础的 CV 任务",
	      items: [
	        {n: "ResNet / EfficientNet", d: "ResNet = 残差连接让网络变深。EfficientNet = 同时缩放宽度/深度/分辨率，效率更高。", course: "第3节",
	         note: "【直觉】ResNet 的核心创新是「高速公路」——输入信号可以跳过中间层直达后面。如果中间层学不到有用的东西，至少输入还能原样传过去（恒等映射）。EfficientNet 发现在三个维度（深度、宽度、分辨率）上按固定比例缩放才能最优利用算力。<br><br>【技术】残差块：output = F(x,{Wi}) + x。如果 F 学到 0，output ≈ x，这就是恒等映射的退路——网络深了至少不会比浅的差。EfficientNet-B0 用 NAS 搜索 baseline，然后用复合系数 φ 同时放大深度（α^φ）、宽度（β^φ）、分辨率（γ^φ），B0→B7 逐级增大。<br><br>【应用】ResNet-50/101 是 CV 检测/分割的通用 backbone。EfficientNet 在移动端和边缘设备上很有优势。TorchVision 直接提供预训练权重。"},
	        {n: "Vision Transformer", d: "ViT 证明 Transformer 也能做视觉，但需要大量数据预训练。Swin 降低计算复杂度。", course: "第10节",
	         note: "【直觉】ViT 把图片切成 16×16 patch，每个 patch 像「词」送入 Transformer 做全局 Attention。这把 CNN「逐层扩大感受野」的设计彻底推翻——ViT 从第一层就看到整张图。代价是需要更大数据量，因为没有 CNN 的归纳偏置（局部性、平移不变性）。<br><br>【技术】输入处理：image(H×W×3) → N 个 Patch(16×16×3) → Linear Projection → N×D 序列 → Position Embedding + [CLS] token → Transformer Encoder。Swin：分层结构（4 stage，像 ResNet）+ W-MSA（窗口内 Attention）+ SW-MSA（偏移窗口跨窗口交互）。<br><br>【应用】ViT 是大规模预训练场景的首选。Swin 在检测和分割上表现优异。ConvNeXt 反向借鉴 Transformer 设计——证明 CNN 用 Transformer 训练策略也能逼近 ViT。"},
	        {n: "自监督预训练", d: "MAE = 遮掉图片大部分，让模型填回去。SimCLR/MoCo = 让同一张图的不同视角表示相似。", course: null,
	         note: "【直觉】MAE（Masked Autoencoder）思路和 BERT 一样——把图片 75% 区域遮掉，让模型把缺失部分画回来。做多了模型学会图片的「结构常识」。SimCLR 思路是「同一张照片翻转变色后，模型应该知道它还是同一张」。<br><br>【技术】MAE：随机 mask 75% patch → 只有未 mask patch 进 Encoder → 所有 patch 进轻量 Decoder → 重建被 mask 像素。非对称设计极快。SimCLR：同一张图两种随机增强 → 过 Encoder + Projection Head → 对比损失 InfoNCE 让表示相似。MoCo 用动量队列维护大量负样本。<br><br>【应用】MAE 预训练 ViT 微调后超越有监督预训练。自监督学习是小标注数据场景的关键技术——大量无标签数据预训练 + 少量有标签数据微调。"}
	      ]},
	    {t: "目标检测", desc: "框出图中的目标——是什么 + 在哪里",
	      items: [
	        {n: "两阶段检测（R-CNN 系列）", d: "先提候选框，再分类。精度高但慢。", course: null,
	         note: "【直觉】两阶段就是「先海选后复赛」——第一阶段生成约 2000 个候选区，第二阶段每个区域做完整分类 + 边框回归。优点是准，缺点是第一阶段太慢（2000 次 CNN 推理）。<br><br>【技术】Faster R-CNN：Backbone 提取特征 → RPN（Region Proposal Network）在特征图上滑动 anchor → RoI Pooling 统一尺寸 → 分类+回归。RPN 用神经网络替代传统 Selective Search，速度快了一大截。<br><br>【应用】需要高精度场景（医疗影像、遥感图像）两阶段仍是首选。Faster R-CNN + FPN（特征金字塔）是经典组合。"},
	        {n: "YOLO 系列（v1-v10）", d: "单阶段 = 一次前向传播同时出框和类别。速度极快，工业界首选。", course: "第5节",
	         note: "【直觉】YOLO（You Only Look Once）核心思想在名字里——只「看」一次。整张图一次 forward 输出所有检测框位置和类别，不像两阶段要「找框→分类」两步。<br><br>【技术】YOLOv1：图片分 S×S 网格，每格预测 B 个框+类别概率。YOLOv3：引入多尺度检测（FPN）+ Darknet-53。YOLOv5（Anchor-based，CSP+Mosaic 增强）和 YOLOv8（Anchor-free，Decoupled Head+C2f 模块）代表了两个技术世代。核心损失 = CIoU 定位损失 + 分类损失 + 置信度损失。<br><br>【应用】YOLO 是工业界目标检测的事实标准——安防、自动驾驶、工业缺陷检测。YOLOv8 精度速度平衡最佳。TFLite/TensorRT 量化后可在边缘设备实时运行（30fps+）。"},
	        {n: "DETR / Deformable DETR", d: "Transformer 做检测 = 不用 NMS，直接预测框。端到端，但训练更慢。", course: null,
	         note: "【直觉】传统检测器最后用 NMS 去掉冗余框——此步不可导。DETR 用 Transformer 全局 Attention 天然做到一对一匹配——每个 object query 自动收敛到一个特定物体，彻底不需要 NMS。<br><br>【技术】DETR：CNN backbone 提取特征 → + Position Encoding → Transformer Encoder 全局建模 → 固定数量 Object Queries → Decoder → 每个 query 输出一个预测。匈牙利算法保证一对一匹配。Deformable DETR 用可变形注意力只关注参考点附近，大幅加速收敛。<br><br>【应用】DETR 的「端到端、无 NMS」影响了检测范式革新。RT-DETR 解决了慢收敛问题，精度速度都很有竞争力。"}
	      ]},
	    {t: "图像分割", desc: "像素级标注——每个像素属于什么类别",
	      items: [
	        {n: "FCN / U-Net", d: "FCN = 用全卷积网络输出与输入等大的 mask。U-Net = 编码器下采样 + 解码器上采样，带跳跃连接。", course: "第6节",
	         note: "【直觉】分类告诉你「图里有猫」，检测告诉你「猫在哪（框）」，分割告诉你「猫每根毛属于哪个像素」。U-Net 像一个 U 形——左边逐层压缩（提取特征），右边逐层放大（恢复分辨率），跳跃连接把左边细节直接传给右边对应层。<br><br>【技术】FCN 把全连接层换成 1×1 卷积，输出与输入同尺寸 mask。U-Net 创新：Encoder（下采样 4 次）+ Decoder（上采样 4 次）+ Skip Connection（特征拼接保留细节）。公式：上采样特征 concat 对应 Encoder 特征 → 卷积 → 继续上采样。<br><br>【应用】U-Net 是医学图像分割的王者。FCN 是所有语义分割方法的鼻祖。理解 Encoder-Decoder 是理解 SegFormer、SAM 等现代分割方法的基础。"},
	        {n: "DeepLab / SegFormer", d: "DeepLab 用空洞卷积扩大感受野。SegFormer 用 Transformer 做编码器。", course: null,
	         note: "【直觉】普通分割网络下采样后细节丢失。DeepLab 用空洞卷积——在卷积核元素间插 0，不增参数、不降分辨率就能看更大范围。SegFormer 用 Transformer 编码器 + 轻量 MLP 解码器，极简但效果极好。<br><br>【技术】空洞卷积 dilation rate=r：插入 r-1 个 0。ASPP（空洞空间金字塔池化）= 并联多个不同 dilation rate 的卷积同时看不同尺度。SegFormer = 层次化 Transformer Encoder + All-MLP Decoder。<br><br>【应用】DeepLabv3+ 是语义分割强 baseline。SegFormer 在城市景观和自动驾驶分割上表现出色。SAM 也用了类似架构。"},
	        {n: "SAM（Segment Anything）", d: "Meta 开源的零样本分割模型。给个点或框就能分割——开箱即用。", course: "第7节",
	         note: "【直觉】SAM 是分割界的 GPT Moment——一个模型训练在 11M 图片、1.1B mask 上，学会「什么是物体」的通用概念。在任何图片点一下或画个框，精细分割出那个物体，完全零样本。<br><br>【技术】SAM：Image Encoder（ViT-H，MAE 预训练）+ Prompt Encoder（编码点/框/mask）+ Mask Decoder（轻量 Transformer 融合特征和提示输出 mask）。SA-1B 数据集含 1100 万张图、11 亿 mask。可以下载权重直接推理或作为预训练模型微调。<br><br>【应用】SAM 是通用分割的里程碑——标注工具、医学影像辅助诊断、视频目标追踪、AR/VR 交互。标志着 CV 从「为每个任务训练一个模型」走向「通用大模型」。"}
	      ]},
	    {t: "图像生成与编辑", desc: "从无到有地创造图像，或改造已有图像",
	      items: [
	        {n: "GAN 家族", d: "GAN = 生成器造假 + 判别器打假，两者博弈。StyleGAN 能精细控制生成人脸的风格。", course: "第11节",
	         note: "【直觉】GAN 就是「造假者 vs 鉴定师」博弈——生成器 G 造更逼真假图，判别器 D 学辨别真伪。两者互相进步直到以假乱真。但这过程极其敏感——双方实力不对等就崩溃。<br><br>【技术】交替训练：固定 G 训练 D 最大化辨别真伪；固定 D 训练 G 最小化被识破概率。难点：梯度消失、Mode Collapse、训练不稳定。WGAN 用 Wasserstein 距离改善。StyleGAN 用 AdaIN 实现层级风格控制（粗糙→中等→精细）。<br><br>【应用】StyleGAN 人脸已达超写实。CycleGAN 做图像翻译（马→斑马）。GAN 速度优势（单步 forward）在实时应用中仍比 Diffusion 有竞争力。"},
	        {n: "Diffusion / Stable Diffusion", d: "Diffusion = 从噪声一步步去噪生成图像。Stable Diffusion 在 Latent 空间做扩散，速度快。", course: "第12-13节",
	         note: "【直觉】扩散模型思路像热力学——给清晰图慢慢加噪声到纯随机噪声；训练网络学「怎么从噪声中去噪」。生成时从纯噪声开始，N 步去噪后露出清晰图像。<br><br>【技术】前向加噪：x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε（有闭合解）。反向去噪：训练网络 ε_θ(x_t,t) 预测噪声 ε，损失为极简 MSE。Stable Diffusion 关键创新：(1) VAE latent 空间扩散（8×压缩），(2) Cross-Attention 用 CLIP text embedding 控制生成。<br><br>【应用】SD 开源引爆 AIGC 浪潮，Midjourney/DALL-E 3 都基于 Diffusion。CFG 控制文字符合度（7-9 最佳），DDIM/DPM-Solver 减采样步数。Sora 用 Diffusion Transformer 生成视频。"},
	        {n: "ControlNet", d: "在 SD 基础上加控制条件（边缘/姿态/深度），让生成服从结构约束。", course: "第13节",
	         note: "【直觉】裸 SD 只能「给文字，出图像」——但你无法控制人物姿势、构图。ControlNet 加了「约束条件」：给姿态骨架图，SD 按骨架生成人物；给边缘图，按边缘填色。生成变得「可控」。<br><br>【技术】ControlNet 在 SD U-Net 每个 Encoder block 旁加可训练「副本」，副本接收额外条件输入（Canny 边缘/深度图/OpenPose 骨架）。通过「零卷积」（1×1 conv 初始化权重为 0）连接原 U-Net——初始不影响原模型，逐渐学习注入条件信息。多 ControlNet 可叠加。<br><br>【应用】ControlNet 是 AIGC 工作流核心——建筑渲染（线稿→效果图）、角色设计（骨架→立绘）、Logo 设计。与 LoRA/DreamBooth 组合可实现「风格+结构」双重控制。"}
	      ]},
	    {t: "视频理解（了解）", desc: "把空间理解扩展到时间维度",
	      items: [
	        {n: "3D CNN / SlowFast", d: "3D CNN = 把卷积扩展到时间轴。SlowFast = 慢分支看空间，快分支看运动。", course: null,
	         note: "【直觉】视频 = 图片 + 时间。3D CNN 把卷积核扩展出第 3 维（时间），在连续帧上滑动。SlowFast 双分支模仿人眼——慢路径高分辨率看「是什么」，快路径低分辨率看「怎么动」。<br><br>【技术】3D CNN 卷积核从 k×k 变 k×k×t，一次看多帧。SlowFast：Slow pathway（低帧率大通道数）+ Fast pathway（高帧率小通道数）+ 侧向连接。VideoMAE 自监督方法让视频理解也可「先预训练再微调」。<br><br>【应用】视频动作识别（Kinetics 数据集）、体育分析、异常检测。视频 Transformer（TimeSformer/ViViT）是更新趋势。"},
	        {n: "目标跟踪", d: "DeepSORT = 检测 + 跟踪。ByteTrack = 用低置信度检测提升跟踪连续性。", course: null,
	         note: "【直觉】跟踪就是「给每帧同一物体标同一个 ID」。做法：每帧检测 → 卡尔曼滤波预测下一帧位置 → 匈牙利算法匹配检测和预测。<br><br>【技术】DeepSORT：检测（YOLO）+ 卡尔曼滤波 + 外观特征（ReID）+ 匈牙利算法（位置+外观联合）。ByteTrack 核心发现：低置信度检测不该扔掉——二次匹配大幅减少 ID 切换。<br><br>【应用】多目标跟踪是安防、自动驾驶、体育分析基础。ByteTrack 是当前最好实时跟踪器之一。"}
	      ]},
	    {t: "三维视觉（了解）", desc: "从 2D 图像理解 3D 世界",
	      items: [
	        {n: "NeRF / 3D Gaussian Splatting", d: "NeRF = 用神经网络表示 3D 场景。3DGS = 用高斯球表示，渲染更快。", course: null,
	         note: "【直觉】NeRF 回答「从 3D 空间某位置、往某方向看，该是什么颜色和透明度」——一个 MLP 将 (x,y,z,θ,φ) 映射到 (RGB,σ)。给几十张不同角度照片，重建整个 3D 场景。<br><br>【技术】NeRF：对每条光线采样 N 个点 → 查询 MLP 得颜色和密度 → 体渲染积分 → 损失 = 渲染 vs 真实。训练慢（几小时）、渲染慢（几秒/帧）。3D Gaussian Splatting：用 3D 高斯椭球显式表示场景 → 可微光栅化渲染 → 训练快（几十分钟）、渲染快（30fps+）。<br><br>【应用】NeRF 用于电影 VFX、VR/AR。3DGS 适合实时应用（游戏、XR）。Luma AI/KIRI Engine 等 App 已在做手机端 3D 扫描。"},
	        {n: "点云处理", d: "PointNet = 直接处理无序点云。3D 目标检测、自动驾驶核心。", course: null,
	         note: "【直觉】LiDAR 输出几万个 (x,y,z) 坐标点。点云无序（打乱顺序还是同一个物体）且无规则网格。PointNet 用对称函数（Max Pooling）解决无序问题。<br><br>【技术】PointNet：每个点 MLP 升维 → 对称函数 Max Pooling 聚合全局特征 + T-Net 学旋转不变性。PointNet++ 增加层次化分组建模局部结构。点云 Transformer（Point-MAE/Point-BERT）是热点。<br><br>【应用】自动驾驶 3D 目标检测（PointPillars/VoxelNet）、激光 SLAM、机器人抓取。"}
	      ]}
	  ]
	},
	{
	  id: "l5", t: "第五层：自然语言处理（NLP）", c: "#4a7c59", i: "💬", inCourse: true,
	  s: [
	    {t: "基础 NLP", desc: "文本预处理和理解的基础技术",
	      items: [
	        {n: "分词（BPE / WordPiece）", d: "把句子切成 token。BPE = 合并最常见的子词对。", course: "第14节",
	         note: "【直觉】计算机不认识「深度学习」，只认识数字。分词把自然语言切成「最小单元」。子词分词（subword）是「按字」和「按词」的折中——既保留常见词完整性，又能用子词组合生僻词。<br><br>【技术】BPE：从字符开始，统计相邻 token 对频率，合并最高频为新 token，重复至词表够大。WordPiece 用概率而非频率决定合并。SentencePiece 不依赖空格分隔（对中文日文友好）。LLaMA 用 BPE，BERT 用 WordPiece。<br><br>【应用】分词器是 NLP 第一步，选错导致 OOV 过多、token 爆炸。中文分词更难（无天然空格），ChatGPT 中文 token 效率只有英文约 1/4。tokenization 是调 LLM API 的隐藏关键。"},
	        {n: "词向量（Word2Vec / GloVe）", d: "把词变成向量，语义相近的词在空间中距离近。", course: null,
	         note: "【直觉】「国王 - 男人 + 女人 ≈ 女王」——词向量最著名例子。神经网络从海量文本自动学到「词的含义 = 它周围常出现什么词」（分布式语义假说）。<br><br>【技术】Word2Vec：CBOW（用周围词预测中心词）和 Skip-gram（用中心词预测周围词，低频词效果好）。GloVe 结合全局共现统计和局部上下文。FastText 把词拆成字符 n-gram，对形态丰富的语言更友好。<br><br>【应用】今天 BERT/GPT 的 Embedding 层是升级版「词向量」，随上下文变化（动态）而非固定（静态）。理解词向量是理解 Embedding 的最佳入口。"},
	        {n: "句法分析 / NER", d: "NER = 命名实体识别。句法分析 = 找词与词的语法关系。", course: null,
	         note: "【直觉】NER 从句子圈出「有名字的东西」——张三（人名）、北京（地名）、谷歌（组织名）。句法分析画语法树——「小明吃了苹果」中「小明」是主语，「苹果」是宾语。<br><br>【技术】NER 是序列标注任务（BIO 标签），常用 BiLSTM+CRF 或 BERT 微调。大模型时代这些被统一到「语言理解」能力中，但理解原理有助于处理 bad case。<br><br>【应用】NER 是信息抽取/知识图谱基础——从新闻提取人物关系、从合同提取甲方乙方。句法分析在搜索引擎、语法纠错中作辅助特征。"}
	      ]},
	    {t: "预训练语言模型（PLM）", desc: "预训练 + 微调范式：先在大语料上学会语言常识，再在小任务上微调",
	      items: [
	        {n: "BERT / RoBERTa", d: "BERT = 编码器，双向理解。Masked LM = 遮住一个词让模型猜。", course: null,
	         note: "【直觉】BERT 像「完形填空」学霸——随机遮掉 15% 的词让它猜，猜多了学会语法、常识、推理。因为是双向（上下文都看），理解力强。但只会填空，不会写作文。<br><br>【技术】BERT：Transformer Encoder only，base 版 12 层 768 维 110M 参数。训练目标：(1) Masked LM，(2) Next Sentence Prediction。RoBERTa 改进：去掉 NSP、更大 batch、更多数据、动态 mask。<br><br>【应用】BERT 曾是 NLP 万能 backbone——任何 NLP 任务加 task-specific head 微调即可。今天虽被 GPT 抢风头，但在文本理解任务上 BERT 类模型仍有竞争力，推理速度远快于 LLM。"},
	        {n: "GPT-1/2/3", d: "GPT = 解码器，单向生成。GPT-3 证明大模型有涌现能力。", course: null,
	         note: "【直觉】GPT 是「只说下一句，不许回头看」的作家——每次只能根据已生成内容预测下一个词。这种「单向」恰好适合「生成」。GPT-3 最大贡献：证明模型大到一定程度（175B），不需要微调，给几个示例就能解决新任务（In-context Learning）。<br><br>【技术】GPT：Transformer Decoder only，Masked Self-Attention 保证自回归。GPT-1（117M）→ GPT-2（1.5B）→ GPT-3（175B），每次扩容带来新能力。涌现能力：规模超阈值后突然出现——CoT 推理、代码生成。<br><br>【应用】GPT 系列是 ChatGPT 基石。Decode-only 架构 = 所有现代 LLM 前提。Scaling Law（性能随参数和数据幂律增长）是核心洞见。"},
	        {n: "T5 / BART", d: "编码器-解码器结构，适合翻译和摘要。T5 = 把所有任务变成文本到文本。", course: null,
	         note: "【直觉】T5 核心思想「万物皆可 seq2seq」——翻译 = 英文→中文，分类 = 文本→'positive'。统一「文本到文本」框架，所有 NLP 任务共用同一模型、同一损失函数。<br><br>【技术】T5 = Text-to-Text Transfer Transformer，Encoder-Decoder 结构。BART 类似，Encoder 双向建模，Decoder 自回归。Flan-T5 通过指令微调获得接近 LLM 能力（小模型大能力）。<br><br>【应用】T5/BART 在文本生成（摘要、翻译）上优于 BERT。Seq2Seq 结构对理解多模态模型（BLIP-2/LLaVA）有帮助。"}
	      ]},
	    {t: "大语言模型（LLM）", desc: "当前最热门方向：大、快、能推理、能对话",
	      items: [
	        {n: "Decoder-only 架构（GPT/LLaMA风格）", d: "只保留解码器，自回归生成。结构简单，适合大规模预训练。", course: "第14节",
	         note: "【直觉】Decoder-only 用「只看前半句猜后一个词」训练。看似简单，但「猜下一个词」本质上包含所有 NLP 能力——要猜对需理解语法、常识、推理甚至数学。<br><br>【技术】GPT/LLaMA 架构要点：Pre-LayerNorm（LN 在 Attention 和 FFN 之前，训练更稳）、RoPE（旋转位置编码，相对位置感知）、SwiGLU 激活（替代 ReLU）、RMSNorm（只缩放不去中心化）。<br><br>【应用】LLaMA 系列是当前开源 LLM 基石。几乎所有中文模型（Qwen/ChatGLM/Yi/DeepSeek）基于类似架构。理解此架构 = 理解 2023-2025 所有大模型核心结构。"},
	        {n: "预训练 → SFT → RLHF", d: "预训练 = 学会说话。SFT = 学会礼貌说话。RLHF = 学会听话。", course: "第14-16节",
	         note: "【直觉】大模型训练像教育——预训练 = 博览群书（学语言知识），SFT = 老师手把手教标准回答（学格式礼貌），RLHF = 好回答奖励差回答惩罚（学对齐人类偏好）。<br><br>【技术】预训练：T 级语料上 next-token prediction，计算量占 95%+。SFT：几千到几万条高质量「指令-回答」对微调。RLHF：训练 Reward Model → LLM 生成 → RM 打分 → PPO 优化。KL 惩罚防止过度优化。<br><br>【应用】ChatGPT/Claude/Gemini 都经历此三阶段。数据质量 >> 数量（1 万高质量 > 10 万噪声）。RLHF 成本推动了 DPO 等简化方法。"},
	        {n: "对齐技术（DPO / PPO）", d: "DPO = 直接偏好优化，不需奖励模型。PPO = 强化学习优化。", course: "第16节",
	         note: "【直觉】RLHF 绕大弯——先训练 Reward Model 再 PPO 优化 LLM。DPO 直接一步到位，把「人类偏好 A>B」变成 LLM 损失函数。数学上等价。<br><br>【技术】PPO（RLHF 中）：π_θ 生成 → RM 打分 → PPO 更新 + KL 惩罚。DPO 损失：L = -log(σ(β·[log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))]))。只需 reference model + 偏好对数据，实现几十行代码。<br><br>【应用】DPO 实现简单、训练稳定，已成对齐主流。Zephyr-7B（DPO）超 70B RLHF 模型。ORPO 合并 SFT+DPO 一波完成。"},
	        {n: "推理优化（KV Cache / 量化）", d: "KV Cache = 缓存已计算的注意力，加速生成。量化 = 用更少 bit 表示权重省显存。", course: "第19节",
	         note: "【直觉】自回归逐 token 生成，每步重算所有 K/V 太浪费。KV Cache 把算好的 K/V 存起来每次只算最新 token——避免 O(n²) 重复计算。量化用更粗糙刻度表示权重——类似四舍五入，损一点精度省一半显存。<br><br>【技术】KV Cache：每层存 (K,V) 矩阵，新 token append。量化：GPTQ/AWQ（训练后量化）、GGUF（CPU 友好，llama.cpp）、bitsandbytes（QLoRA 训练用）。INT8 几乎无损，INT4 轻微损失。FlashAttention 利用 GPU SRAM 层次加速 Attention。<br><br>【应用】KV Cache 是所有 LLM 推理框架核心。量化让 7B 模型在 6GB 显存运行，70B 在 48GB 运行。vLLM 用 PagedAttention 管理 KV Cache。"},
	        {n: "Agent 与工具（ReAct/Function Calling）", d: "Agent = 让 LLM 调用外部工具。ReAct = 思考→行动→观察循环。", course: "第18节",
	         note: "【直觉】LLM 像博学书生——满腹经纶但手无缚鸡之力。Agent 给它配「工具」（搜索、计算器、代码执行、API），让它能动用外部能力解决问题。ReAct 循环：「先想想→调用工具→看到结果→再想想→…直到解决」。<br><br>【技术】ReAct：Thought→Action→Observation→Thought→…，推理和工具调用交替。Function Calling：定义工具函数 JSON Schema → LLM 输出函数调用请求 → 外部执行 → 结果返回 LLM。多 Agent（AutoGen/CrewAI）分工协作。<br><br>【应用】RAG 是最简单常用 Agent 模式。LangChain/AutoGPT 工具链。Agent 是 LLM 从「聊天玩具」走向「生产力工具」的关键桥梁。"},
	        {n: "RAG（检索增强生成）", d: "RAG = 先检索相关文档，再让 LLM 基于文档回答。解决幻觉和知识过时。", course: "第17节",
	         note: "【直觉】LLM 两大硬伤：知识有时效性，会「一本正经胡说八道」。RAG 就是「写论文前先查资料」——系统先去文档库搜相关段落，把搜到内容连同问题给 LLM，让它「看着资料回答」。<br><br>【技术】RAG 流程：文档分块→Embedding 模型编码→存入向量数据库（FAISS/Chroma/Milvus）→用户查询编码→向量检索+关键词检索（BM25）混合→检索结果组成增强 prompt→LLM 生成。进阶：Re-ranking、小→大检索、Self-RAG。<br><br>【应用】企业知识库问答、客服机器人、法律/医疗文档检索。RAG 是当前 LLM 落地最成熟、ROI 最高的方案之一。挑战：文档切分策略、检索召回率、回答忠实度验证。"}
	      ]}
	  ]
	},
	{
	  id: "l6", t: "第六层：多模态学习（Multimodal）", c: "#5c5470", i: "🔄", inCourse: true,
	  s: [
	    {t: "视觉-语言模型", desc: "让模型同时理解图像和文字——AI 的通感",
	      items: [
	        {n: "CLIP", d: "对比学习：拉近匹配的图文对，推开不匹配的。零样本分类 = 不给训练数据直接分类。", course: "第20节",
	         note: "【直觉】CLIP 做件简单事——从网上收集 4 亿对「图片+文字」，训练模型让匹配对表示接近、不匹配对远离。神奇结果：训练完后给图 + N 个文字标签，它能选中最佳——完全没在目标数据集训练过（零样本）。<br><br>【技术】Image Encoder（ViT/ResNet）+ Text Encoder（Transformer）。InfoNCE 对比损失：batch 内让匹配 N 对互相关注（对角线），其他 N²-N 对推开。推理：将类别名填充为模板（「a photo of {class}」）→ 算图文相似度 → softmax。<br><br>【应用】CLIP 是文生图和多模态基础设施——SD 用 CLIP Text Encoder 编码 prompt。零样本分类让「不用训练数据就分类」成为可能。"},
	        {n: "BLIP / BLIP-2", d: "BLIP = 统一理解+生成。BLIP-2 = 用轻量查询 Transformer 桥接冻结的视觉编码器和 LLM。", course: null,
	         note: "【直觉】BLIP 发现网图描述质量差。它用生成器生成更好的描述，过滤器去掉差的，形成高质量图文对反哺预训练。BLIP-2 更进一步——既然视觉编码器和 LLM 都是现成的，能否用轻量适配器连接它们？<br><br>【技术】BLIP：多任务训练（对比+匹配+生成）+ Bootstrap 过滤。BLIP-2：冻结 Image Encoder + 冻结 LLM + Q-Former（~180M 参数）做桥梁。Q-Former 可学习 Queries 从图像特征提取文本相关信息，作为 LLM 的 soft prompt。<br><br>【应用】BLIP-2 是多模态 LLM 里程碑——证明了「冻结+桥接」策略有效。LLaVA/MiniGPT-4 都受了 BLIP-2 影响。「冻结预训练模块+轻量适配器」是高效多模态训练标准做法。"},
	        {n: "DALL-E / Stable Diffusion", d: "文生图 = 从文本描述生成图像。CLIP 的反向应用：用文字引导图像生成。", course: "第13节",
	         note: "【直觉】CLIP 是「看图说话」（image→text），SD 是「看字画画」（text→image）。用 CLIP 把文字变向量，扩散模型反向去噪「听从向量引导」——每步去噪朝更符合文字描述的方向走。<br><br>【技术】SD 组件：VAE（512→64 latent，8×压缩）、U-Net（latent 空间去噪 + Cross-Attention 接收 CLIP text embedding）、CLIP Text Encoder（编码 prompt）、Scheduler。关键参数：CFG Scale（7-9）、Steps（20-30）。<br><br>【应用】SD 开源引爆 AIGC 浪潮。SDXL/SD3 持续提升质量。SD 生态含数千 LoRA 权重。"},
	        {n: "LLaVA / Qwen-VL / GPT-4V", d: "多模态大模型 = 把图像编码成 token，和文本一起送入 LLM。看图说话、图文问答。", course: "第21节",
	         note: "【直觉】把图片「翻译」成 LLM 能读的「语言」——图像经视觉编码器变 token 序列，拼在文字 token 前送 LLM。LLM 不关心 token 来自图片还是文字，同一 Attention 处理一切。<br><br>【技术】LLaVA：CLIP ViT 编码图像 → 线性投影层（映射到 LLM embedding 空间）→ 拼接 visual+text tokens → LLM 自回归生成。两阶段训练：(1) 只训练投影层做图文对齐，(2) 微调投影层+LLM 做指令跟随。<br><br>【应用】多模态 LLM 让 AI「看到世界」——看图问答、图表分析、UI 理解、医学影像。Qwen-VL 和 GPT-4V 在中文场景和复杂推理上表现优秀。当前最活跃 AI 方向之一。"}
	      ]},
	    {t: "音频-语言（了解）", desc: "语音和音乐的理解与生成",
	      items: [
	        {n: "Whisper（ASR）", d: "OpenAI 开源的语音识别模型，多语言支持，效果接近人类。", course: null,
	         note: "【直觉】Whisper 思路极其简单——语音切 30 秒段，Encoder-Decoder Transformer 直接「翻译」音频（Mel 频谱图）成文字。和 NLP 翻译完全相同架构。核心壁垒不是算法多新颖，而是 68 万小时多语言弱监督数据。<br><br>【技术】音频→80 通道 Mel 频谱→2 层 Conv→Positional Encoding→Transformer Encoder→Decoder 自回归生成。弱监督+大规模=强泛化。WhisperX 增加说话人分离和时间对齐。<br><br>【应用】中文识别接近商业 API。本地部署替代付费 ASR。结合 LLM 做会议转录+摘要。"},
	        {n: "TTS / 音乐生成", d: "VITS = 端到端语音合成。Suno = 当前最热门音乐生成工具。", course: null,
	         note: "【直觉】TTS 是 「seq2seq + Vocoder」——文字→声学特征→波形。音乐生成同理——描述→音乐频谱→可播放音频。<br><br>【技术】VITS = VAE+Flow+GAN+HiFi-GAN，端到端合成。ChatTTS/GPT-SoVITS 是优秀中文 TTS 开源方案。Suno/Udio 普遍推测基于 Diffusion 或 Audio Language Model。<br><br>【应用】TTS 用于有声书、导航、虚拟主播。Suno v3 可生成 2 分钟+完整歌曲（含歌词旋律配器），冲击内容创作行业。"}
	      ]},
	    {t: "统一多模态架构（了解）", desc: "终极目标：一个模型理解所有模态",
	      items: [
	        {n: "模态对齐与融合", d: "不同模态的表示要映射到同一空间，才能互相理解。", course: null,
	         note: "【直觉】人类看到苹果时——「红红的圆圆的」（视觉）、「苹果」（文字）、咬一口脆响（听觉），三种信号在大脑中整合为统一概念。多模态 AI 核心挑战就是让不同「感官」信息在同一个「思维空间」对齐。<br><br>【技术】对齐方法：Contrastive（CLIP 式）、Projection（映射到共享空间）、Tokenization（图像变 token 拼入文本序列）。ImageBind 对齐六种模态。融合时机：Early/Middle/Late Fusion。<br><br>【应用】统一多模态是通向 AGI 必经之路。GPT-4o/Gemini 都在朝此方向快速演进。"},
	        {n: "世界模型（World Model）", d: "让 AI 像人一样有物理直觉，理解因果关系和动态变化。", course: null,
	         note: "【直觉】你看到杯子在桌边，不用公式就知道它可能掉下去。世界模型让 AI 建立这种「物理直觉」——看到场景预测下一步会发生什么。这是从「感知」走向「理解」的关键一步。<br><br>【技术】Sora 是最强世界模型代表——3D 时空 patch + Diffusion Transformer，似乎学到隐式物理规则。JEPA（LeCun 提出的联合嵌入预测架构）、Dreamer（RL 世界模型）是其他方向。<br><br>【应用】机器人仿真（「想象」中练习）、自动驾驶、游戏 AI、视频生成。这是通往 Physical AI 的核心技术。"}
	      ]}
	  ]
	},
	{
	  id: "l7", t: "第七层：生成式AI（Generative AI）", c: "#b85c5c", i: "✨", inCourse: true,
	  s: [
	    {t: "生成模型范式", desc: "四种生成图像/数据的核心思路",
	      items: [
	        {n: "VAE（变分自编码器）", d: "编码器把数据压缩到隐空间，解码器从隐空间重建。", course: "第11节",
	         note: "【直觉】VAE 像「会画画的人」——你给猫图，它压缩成关键特征（耳朵尖度、毛色、体型），再从特征画回猫。关键创新：压缩到分布（μ 和 σ²）而非点，赋予了 VAE 生成能力。<br><br>【技术】Encoder 输出 μ 和 σ²，重参数化 z = μ+σ·ε 让采样可导。Decoder 从 z 重建 x。损失 = 重建损失 + β·KL(q_φ||p(z))。KL 项让编码分布靠近标准正态，正则化隐空间。<br><br>【应用】VAE 隐空间连续结构化（相近 z 生成相似图，可插值）。虽生成质量不如 GAN/Diffusion，但思路深刻影响后续所有生成模型——SD 里的 VAE 就用于 latent 压缩。"},
	        {n: "GAN（生成对抗网络）", d: "生成器 vs 判别器，零和博弈。训练不稳定但生成质量高。", course: "第11节",
	         note: "【直觉】GAN 是「造假画」（G）vs「鉴画师」（D）博弈。G 造逼真假画骗 D，D 学辨别真伪。博弈到纳什均衡时假画乱真。但训练极其敏感——实力不对等就崩溃。<br><br>【技术】交替训练：固定 G 训练 D；固定 D 训练 G。难点：梯度消失、Mode Collapse、不稳定。WGAN 用 Wasserstein 距离改善。StyleGAN 用 AdaIN 层级化风格控制（粗糙→中等→精细）。<br><br>【应用】StyleGAN 人脸超写实。CycleGAN 做图像翻译。GAN 速度优势（单步 forward）在实时应用中仍比 Diffusion 有竞争力。"},
	        {n: "Flow-based", d: "用可逆变换把简单分布映射到复杂分布。数学优雅但计算量大。", course: null,
	         note: "【直觉】如果能把标准正态分布一步一步「折叠/拉伸/扭曲」成复杂数据分布，而且可逆（能生成也能反算概率），就太完美了。Flow-based 做到了这点。<br><br>【技术】核心：p_x(x) = p_z(f^{-1}(x))·|det(∂f^{-1}/∂x)|。代表：NICE/RealNVP（仿射耦合层）、Glow（1×1 可逆卷积）。Flow Matching 是 2023-2024 新热点。<br><br>【应用】精确概率密度估计在异常检测中有独特优势。Flow Matching 在生成质量和速度上追赶 Diffusion。"},
	        {n: "Diffusion（扩散模型）", d: "当前主流。前向加噪 + 反向去噪。数学直觉清晰。", course: "第12节",
	         note: "【直觉】训练一个会去噪的模型——先给数据加噪声弄脏，训练网络学擦掉噪声。生成时从纯噪声开始一步步擦出清晰图像。整个过程有热力学物理直觉和优美数学公式。<br><br>【技术】前向：x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε（闭合解）。反向：训练网络 ε_θ(x_t,t) 预测噪声 ε，损失 L = ||ε - ε_θ||²。DDPM 变分界化简后就是极简 MSE。DDIM 做到快采，SD 在 latent 空间扩散。<br><br>【应用】Diffusion 是当前生成式 AI 核心技术。DDPM→DDIM→SD→DiT→Sora 是追踪前沿的必经之路。"}
	      ]},
	    {t: "文本生成", desc: "Prompt 是新时代的编程语言",
	      items: [
	        {n: "Prompt Engineering", d: "零样本 = 直接问。少样本 = 给例子。CoT = 让模型一步步想。", course: null,
	         note: "【直觉】Prompt 像给 AI 下指令——说得好完美执行，说得含糊瞎猜。CoT 就是「别直接给答案，先让我想想」——输出中间推理步骤，准确率显著提升。ToT 让模型探索多条思路选最优。<br><br>【技术】Zero-shot：只指令无示例。Few-shot：给 2-8 示例。CoT：prompt 中加「Let's think step by step」引导。ToT：维护思路树，LLM 自己评估每条思路可行性选最优。<br><br>【应用】CoT 把 GPT-3 数学题 ~17% 提到 ~60%。提示工程是「AI 时代编程语言」——写好 prompt 就是写 AI 程序。"},
	        {n: "可控文本生成", d: "PPLM/CTRL = 用属性向量控制生成风格。前缀微调 = 给模型加前缀引导输出。", course: null,
	         note: "【直觉】同写「今天天气很好」，可用开心语气（「美好的一天！」）也可用悲伤语气（「炎热难熬的一天……」）。可控文本生成给生成加「风格」「情感」等约束。<br><br>【技术】PPLM：不改原模型，加属性模型梯度引导输出。Prefix Tuning：每层 Attention K/V 前拼接可学习「前缀向量」。CTRL：训练时用控制代码作条件 token。<br><br>【应用】写作助手（控制风格/语气/长度）、客服（控制礼貌度）、AI 对话（控制人设）。LoRA 等参数高效微调是可控生成思路延伸。"}
	      ]},
	    {t: "图像/视频/3D 生成", desc: "生成式 AI 最出圈的领域",
	      items: [
	        {n: "Stable Diffusion 生态", d: "LoRA = 小参数微调风格。DreamBooth = 用几张图训练专属概念。", course: "第13节",
	         note: "【直觉】SD 是「通才」，都会画但不精。LoRA 给通才「速成培训班」——10-20 张风格图训练小插件，就能画特定风格。DreamBooth 用 3-5 张宠物照片，让 SD「认识」你的宠物。<br><br>【技术】LoRA（Low-Rank Adaptation）：大模型旁加小矩阵 AB（0.1%-1% 参数），只训练 AB。DreamBooth：稀有词绑定目标概念 + Prior Preservation Loss 防遗忘。ControlNet、IP-Adapter、InstantID 各解决不同维度控制。<br><br>【应用】CivitAI 上万 LoRA 模型。DreamBooth 用于虚拟试衣、个性化头像。SD 生态是最活跃 AI 社区之一。"},
	        {n: "视频生成（Sora 原理）", d: "Sora = 把视频当 3D 时空块处理，用 Diffusion Transformer 生成。", course: null,
	         note: "【直觉】Sora 把视频看作「3D 积木块」——切成 3D 小块（宽×高×时间），每个小块像 token。Diffusion Transformer 在时空块上做扩散生成。这是 ViT 把图片切 patch 的 3D 版。<br><br>【技术】视频压缩网络→latent 时空空间→提取时空 patch→DiT 做扩散。用 recaptioning（GPT 详细描述视频）改善 text-video 对齐。Scalability 是核心——模型越大、数据越多，物理准确度越高。<br><br>【应用】Sora 是全球视频生成风向标。Runway Gen-2/3、Pika、Kling（可灵）、Vidu 是竞品。处于类似 2022 年图像生成爆发前夜。"},
	        {n: "3D 生成", d: "DreamFusion = 用文字生成 3D 模型。Shap-E = 直接输出点云。", course: null,
	         note: "【直觉】3D 生成核心挑战——无海量 3D 标注数据。DreamFusion 聪明解法：利用 2D 扩散模型知识指导 3D 生成——从不同角度渲染，让 SD 打分（「像不像文字描述」），用分数优化 3D 参数。<br><br>【技术】DreamFusion（SDS 损失）：随机初始化 NeRF→随机角度渲染→SD 打分→梯度反传到 3D 参数→迭代。Shap-E：训练文本→点云的扩散模型，推理快但质量不如 SDS。<br><br>【应用】Meshy/Luma AI Genie 提供商业化文生 3D 服务。质量还在「勉强能用」阶段，但 3DGS 给实时高质渲染带来希望。"}
	      ]}
	  ]
	},
	{
	  id: "l8", t: "第八层：强化学习（RL）", c: "#8a7f72", i: "🎮", inCourse: false,
	  s: [
	    {t: "基础 RL", desc: "让 AI 通过试错学习最优策略——像训练宠物一样",
	      items: [
	        {n: "MDP", d: "马尔可夫决策过程 = 状态→动作→奖励→新状态。假设未来只取决于当前状态。", course: null,
	         note: "【直觉】MDP 是 RL 数学语言——你在「状态」里（棋盘局面），选「动作」（走哪步），得「奖励」，转移新「状态」。马尔可夫性假设「未来只取决于现在」——知道当前棋局就够。<br><br>【技术】MDP 五元组：(S,A,P,R,γ)。V(s)=期望回报，Q(s,a)=在 s 做 a 的期望回报。G_t = R_t+γR_{t+1}+γ²R_{t+2}+…。γ∈[0,1) 是折扣因子。理解 MDP 是理解 RL 的前提。<br><br>【应用】MDP 框架通用——游戏、机器人控制、自动驾驶、推荐系统、LLM 对齐。"},
	        {n: "Q-Learning / DQN", d: "Q = 在某个状态做某个动作的预期总奖励。DQN = 用神经网络近似 Q 函数。", course: null,
	         note: "【直觉】Q-Learning 画一张「每个状态每步动作值多少钱」的表格。但真实问题状态太多（围棋 10¹⁷⁰ 盘面），存不下表格，用神经网络近似就是 DQN。<br><br>【技术】Q 更新：Q(s,a) ← Q(s,a) + α[r+γ·max_a'Q(s',a')-Q(s,a)]。DQN 两大创新：(1) Experience Replay（打破数据相关性），(2) Target Network（定期同步让训练稳定）。<br><br>【应用】DQN 在 Atari 游戏上从像素直接学出超人类水平，引发 DL+RL 浪潮。Replay Buffer 和 Target Network 影响了后续几乎所有 RL 算法。"},
	        {n: "Actor-Critic", d: "Actor = 策略网络（决定做什么）。Critic = 价值网络（评价做得好不好）。", course: null,
	         note: "【直觉】Actor-Critic 像演员+评论员搭档。演员选动作，评论员打分。演员按分进步，评论员按实际反馈校准。两者互相促进。<br><br>【技术】Actor π_θ(a|s) 用 Policy Gradient 更新，Critic V_φ(s) 估计价值，Advantage A = Q-V。A3C/A2C 是经典实现，SAC 加最大熵目标探索更强。<br><br>【应用】几乎所有现代 RL 算法都是 Actor-Critic 变体。PPO 本质是 Actor-Critic+Trust Region。理解 Actor-Critic 是理解所有 Policy-based RL 的钥匙。"},
	        {n: "PPO", d: "Proximal Policy Optimization = 策略更新步长不能太大。当前最常用 RL 算法。", course: "第16节（DPO）",
	         note: "【直觉】RL 有「跷跷板」——更新太大旧经验废了，太小训练慢。PPO 给策略更新加「安全距离」——新旧策略不能差太远，安全范围内尽量优化。<br><br>【技术】PPO-Clip：L = min(r·A, clip(r,1-ε,1+ε)·A)。r = π_new/π_old，A 是 Advantage。clip 防止 r 偏离 1 太远。简单、稳定、好用——PPO 成为 RL「默认算法」。<br><br>【应用】ChatGPT RLHF 用 PPO 对齐。OpenAI Five/Dota 2 用 PPO 训练。ANYmal 四足机器狗用 PPO 学步态。"}
	      ]},
	    {t: "RLHF", desc: "强化学习的最大应用：让 LLM 对齐人类偏好",
	      items: [
	        {n: "RLHF 流程", d: "1) 训练奖励模型。2) 用 PPO 优化 LLM 让打分更高。3) 模型变听话。", course: "第16节",
	         note: "【直觉】LLM 预训练后会「说话」但不会「说人喜欢听的话」。RLHF 像驯兽——人类对回答排序（A>B），奖励模型学习模仿打分，LLM 用 PPO 根据打分优化。<br><br>【技术】三阶段：(1) 收集偏好数据，(2) 训练 Reward Model（Bradley-Terry 模型），(3) PPO 优化 LLM：max r_φ(生成回答) - β·KL(π_θ||π_ref)，KL 惩罚防过度优化。<br><br>【应用】ChatGPT/Claude/Gemini 都用了 RLHF。提升不只是「更礼貌」更是「更准、更有用、更拒危害请求」。挑战：RM 可能被 hack、标注成本高。"},
	        {n: "DPO（直接偏好优化）", d: "跳过奖励模型，直接用偏好数据优化。更简单，效果相当。", course: "第16节",
	         note: "【直觉】RLHF 绕大弯——先训 RM 再 PPO。DPO 一步到位，把偏好 A>B 直接变 LLM 损失函数。数学可证等价。<br><br>【技术】DPO 损失：L = -log(σ(β·[log(π_θ(y_w)/π_ref(y_w))-log(π_θ(y_l)/π_ref(y_l))]))。只需 reference model + 偏好对数据，几十行代码。<br><br>【应用】Zephyr-7B（DPO）超 70B RLHF 模型。ORPO 合并 SFT+DPO。DPO 推动对齐从「大厂秘方」走向「社区开源标配」。"}
	      ]}
	  ]
	},
	{
	  id: "l9", t: "第九层：图神经网络（GNN）", c: "#8a7f72", i: "🕸️", inCourse: false,
	  s: [
	    {t: "基础与经典模型", desc: "处理非网格数据（社交网络、分子结构、知识图谱）",
	      items: [
	        {n: "GCN", d: "图卷积 = 把邻居节点的特征聚合到自己。拉普拉斯矩阵是图上的导数。", course: null,
	         note: "【直觉】普通卷积在 grid 上（邻居固定 3×3），图卷积在图上——每个节点邻居由边决定。核心：收集邻居特征，聚合后更新自己。多层堆叠后能感知更远邻居。<br><br>【技术】GCN：H^{l+1} = σ(D^{-½}ÂD^{-½}H^lW^l)。Â=A+I（加自连接），对称归一化防度数大的节点主导。拉普拉斯矩阵 L=D-A 的特征向量是图上「傅里叶基」，图卷积=谱域滤波。<br><br>【应用】社交网络分析、推荐系统、分子性质预测、交通预测。「邻居聚合」思想也启发了 Transformer。"},
	        {n: "GAT", d: "图注意力 = 给不同邻居不同权重。和 Transformer Self-Attention 思路相通。", course: null,
	         note: "【直觉】GCN 对邻居「一视同仁」，但现实中不同邻居重要性不同。GAT 让节点自己学「该听哪个邻居」，给不同权重。<br><br>【技术】对节点 i 邻居 j：e_ij = LeakyReLU(a^T[Wh_i||Wh_j])，α_ij = softmax(e_ij)，h'_i = σ(Σα_ij Wh_j)。多头注意力：K 组独立注意力拼接/平均。与 Transformer 区别：GAT 只在相邻节点间算注意力（局部），Transformer 在所有 token 间（全局）。<br><br>【应用】引文网络分类、知识图谱推理、蛋白质相互作用预测。"},
	        {n: "GraphSAGE", d: "归纳式学习 = 能处理训练时没见过的节点。工业推荐系统常用。", course: null,
	         note: "【直觉】GCN 致命缺陷——训练时所有节点在图中，新节点来要重训。GraphSAGE 改学「聚合函数」——不管邻居是谁，有特征就能算新节点表示。<br><br>【技术】聚合方式：Mean/LSTM/Pooling Aggregator。采样固定数量邻居（降计算量）。核心：归纳式——训练学的是「怎么聚合」而非「特定节点表示」。<br><br>【应用】工业推荐 GNN 基础（Pinterest PinSage = GraphSAGE+RandomWalk）。新用户/商品立即可生成 embedding。这个「归纳式」能力是 GNN 从学术走向工业的关键。"}
	      ]},
	    {t: "应用", desc: "GNN 的工业落地场景",
	      items: [
	        {n: "推荐系统", d: "PinSage = Pinterest 的 GNN 推荐。用户-物品是天然图结构。", course: null,
	         note: "【直觉】推荐天然是图——用户和商品是节点，点击/购买是边。传统协同过滤只看「相似用户喜欢相似东西」，GNN 做深层传播——A 喜欢 B，B 被 C 喜欢，C 还喜欢 D，所以 A 可能也喜欢 D。<br><br>【技术】PinSage：商品-商品图 Random Walk+GraphSAGE，重要性权重采样。MapReduce 化处理 30 亿节点+180 亿边。LightGCN 简化 GCN——去掉特征变换和非线性，只留邻居聚合，效果反而更好。<br><br>【应用】Pinterest 首页推荐、淘宝「猜你喜欢」、抖音推荐都不同程度用 GNN。HGT 处理多种节点和关系。"},
	        {n: "分子/药物发现", d: "分子是图（原子是节点，化学键是边）。GNN 预测分子性质。", course: null,
	         note: "【直觉】分子天然是图——原子（C/H/O/N）是节点，化学键（单/双/三键）是边。GNN 学「分子能不能抗癌」「溶于水吗」——性质由原子排列和键连接决定。<br><br>【技术】分子→图表示（原子特征+键特征+邻接矩阵）→GNN（GCN/GAT/MPNN）→Readout→预测。AlphaFold 用 Attention 预测蛋白质结构。<br><br>【应用】AI 加速材料筛选 100-1000×。DeepMind GNoME 发现 220 万种新晶体材料。分子生成（VAE/GAN/Diffusion）用于新药发现。"}
	      ]}
	  ]
	},
	{
	  id: "l10", t: "第十层：时序与结构化数据", c: "#8a7f72", i: "📊", inCourse: false,
	  s: [
	    {t: "时间序列", desc: "预测未来、检测异常——从股票到传感器",
	      items: [
	        {n: "LSTM / GRU", d: "RNN 改进版，用门控机制解决长程依赖。", course: "第8节",
	         note: "【直觉】预测股价需要多天前的趋势，而非上一秒的波动。LSTM 的门控让网络自主决定「该长期记住」还是「可以忘了」——三扇门像开关控制信息流。<br><br>【技术】遗忘门 f_t、输入门 i_t、输出门 o_t 三元控制。细胞状态 c_t = f_t⊙c_{t-1}+i_t⊙c̃_t 是「长期记忆带」，只做线性操作（乘+加），梯度几乎不衰减。GRU 参数更少效果类似。<br><br>【应用】小数据量时序场景（几百个点）LSTM 仍然好用，计算量远小于 Transformer。虽然 Transformer 逐渐渗透，但 LSTM 的知识是理解序列建模的基础。"},
	        {n: "Informer / Autoformer", d: "Transformer 做时序。Informer 用 ProbSparse Attention 降复杂度。", course: null,
	         note: "【直觉】普通 Transformer O(L²) 对长时序（几千步）是灾难。Informer 发现长序列只有少数「关键时刻」需全局关注，稀疏注意力只关注这些时刻，复杂度降到 O(L log L)。<br><br>【技术】Informer 三大创新：(1) ProbSparse Self-Attention——Query 活跃度用 KL 散度衡，只保留 Top-U 活跃 Query，(2) Distilling——MaxPooling 压缩长度，(3) Generative Decoder——一次输出整个预测序列。Autoformer 用自相关替 Attention。<br><br>【应用】电力负载预测、天气预测、金融高频交易。时序 Transformer 是快速发展方向。"},
	        {n: "TimesNet", d: "2023 年提出的通用骨干，将 1D 时序转为 2D 空间处理。", course: null,
	         note: "【直觉】时序难点在「周期模式」——股票有日/周/月周期叠加。TimesNet 巧妙思路：把 1D 时序按不同周期 reshape 成 2D（每行一个周期），CNN 在 2D 空间同时捕获周期内和跨周期变化。<br><br>【技术】FFT 找 K 个主要周期→对每个周期 reshape 1D→2D→Inception Block（多尺度 CNN）提取特征→reshape 回 1D→加权融合。<br><br>【应用】TimesNet 在五种时序任务上均达 SOTA。PatchTST 是另一优秀设计——把时序切 patch 送 Transformer。"}
	      ]},
	    {t: "推荐系统", desc: "从协同过滤到深度推荐",
	      items: [
	        {n: "FM / DeepFM", d: "FM = 特征交叉。DeepFM = FM 和 DNN 结合，学习低阶+高阶特征交互。", course: null,
	         note: "【直觉】推荐核心是特征交叉——「年轻人+晚上+周末」更可能点「游戏」。FM 用向量内积建模二阶交叉。DeepFM = FM（低阶）+DNN（高阶）+共享 Embedding。<br><br>【技术】FM 分量 ŷ_FM = w₀+Σw_ix_i+Σ_{i&lt;j}⟨v_i,v_j⟩x_ix_j（向量内积建模二阶交叉）。DeepFM = FM 分量 + DNN 分量，同时捕获记忆性和泛化性。<br><br>【应用】工业推荐技术栈：LR→FM→Wide&Deep→DeepFM→DIN→DIEN→SIM。理解这条线就理解推荐系统核心脉络。"},
	        {n: "序列推荐", d: "SASRec = 用 Transformer 做序列推荐。BERT4Rec = 把行为序列当 NLP 处理。", course: null,
	         note: "【直觉】最近浏览序列（手机壳→贴膜→充电器→耳机）暗示短期意图。把行为序列当「文本」，用 NLP 方法预测「下一个 token」（下一个商品）。<br><br>【技术】SASRec：行为序列→Embedding+Position Embedding→单向 Transformer Decoder→最后 token 接 softmax。BERT4Rec：双向 Encoder（像 BERT），遮部分行为让模型猜。<br><br>【应用】电商、短视频、音乐等场景。LLM 崛起催生了 P5/RecLLM——推荐数据用自然语言描述，LLM 统一处理。"}
	      ]}
	  ]
	},
	{
	  id: "l11", t: "第十一层：工程与系统（工业落地必备）", c: "#c25e00", i: "⚙️", inCourse: true,
	  s: [
	    {t: "模型训练工程", desc: "让训练更快、更稳、可恢复",
	      items: [
	        {n: "分布式训练", d: "数据并行 = 每 GPU 算不同数据。模型并行 = 每 GPU 存部分模型。ZeRO = 优化器状态分散到多 GPU。", course: null,
	         note: "【直觉】70B 模型 FP32 权重占 280GB，加 Adam 优化器状态（一阶动量+二阶动量+梯度+参数副本）总共约需 4 倍权重空间——一台机器装不下。分布式训练「分而治之」——数据分多 GPU（数据并行），或模型拆多 GPU（模型并行），或组合。<br><br>【技术】数据并行（DDP）：每 GPU 有完整模型，各自处理不同 batch，AllReduce 同步梯度。模型并行：模型每层拆分到不同 GPU。ZeRO（DeepSpeed）：优化器状态/梯度/参数逐步分布。FSDP 类似 ZeRO-3。<br><br>【应用】训练 7B：8×A100 ZeRO-2 可行。训练 70B：需 ZeRO-3+张量并行+流水线并行。DeepSpeed 和 FSDP 是两大主流框架。"},
	        {n: "混合精度（FP16/BF16）", d: "用半精度计算加速，全精度存关键值。提速 2-3 倍，显存省一半。", course: "第15节",
	         note: "【直觉】FP32 像量体重精确到小数点后 8 位，FP16 只到后 2 位——多数情况够用。混合精度「粗算+精存」——计算用 FP16（快），关键值保 FP32（准），几乎无损 2 倍加速。<br><br>【技术】BF16 范围与 FP32 相同（不易溢出）。PyTorch AMP：`with autocast():`+`scaler.scale(loss).backward()` 两行搞定。NVIDIA Tensor Core 专门优化了 FP16/BF16。<br><br>【应用】现代大模型训练默认开混合精度。70B BF16 只需 140GB。AMP 让消费级 GPU 也能训练大模型。"},
	        {n: "梯度累积/裁剪", d: "累积 = 小 batch 模拟大 batch。裁剪 = 防止梯度爆炸。", course: null,
	         note: "【直觉】显存只够 batch=4，但 batch=64 效果更好。梯度累积跑 16 次 batch=4，梯度累加后一次性更新——等价 batch=64。梯度裁剪是「限速器」——梯度突然巨大直接限制住。<br><br>【技术】每 N 次 forward/backward 后 optimizer.step()+zero_grad()。梯度裁剪：梯度范数 > max_norm 则 g = g×max_norm/||g||。<br><br>【应用】梯度累积是有限显存下训练大模型的基础技巧。梯度裁剪在 RNN 训练中必需，Transformer 偶尔用。"}
	      ]},
	    {t: "模型部署与推理", desc: "训练好的模型要变成产品，需要部署优化",
	      items: [
	        {n: "服务化框架", d: "TorchServe/Triton = 把模型封装成 API。支持批量推理、动态批处理。", course: null,
	         note: "【直觉】.pt 文件离「产品」还有很远。服务化框架把模型包装成可扩展的 HTTP/gRPC 服务，自动处理请求排队、批量合并、版本切换、健康检查。<br><br>【技术】TorchServe：定义 Handler→打包 .mar→部署。Triton（NVIDIA）：多框架、动态批处理、多模型流水线。vLLM/TGI：专为 LLM 推理设计，PagedAttention 管理 KV Cache，吞吐比 HF 高 10-20 倍。<br><br>【应用】TorchServe 适合中小规模。Triton 是大规模生产标配。vLLM 是当前 LLM API 服务主流方案。"},
	        {n: "模型压缩", d: "剪枝 = 去不重要权重。量化 = 用 INT8/INT4 代替 FP32。知识蒸馏 = 大模型教小模型。", course: "第19节",
	         note: "【直觉】大模型像博学教授，手机上跑就太大了。三种瘦身：(1) 剪枝——删不重要知识点，(2) 量化——32 位→8/4 位，(3) 蒸馏——让教授（大模型）手把手教聪明学生（小模型）。<br><br>【技术】剪枝：非结构化（单权重置 0）vs 结构化（去整个 channel）。量化：PTQ（GPTQ/AWQ）vs QAT（训练模拟量化，精度高）。蒸馏：Student 学 Teacher 软标签（温度 T 放大差异）+ 硬标签。<br><br>【应用】INT8 几乎无损，INT4 轻微损失。QLoRA 用 NF4 量化做 4-bit 训练。DistilBERT 压缩 40% 保持 97% 性能。手机跑 LLM 靠量化+蒸馏（GGUF 格式）。"},
	        {n: "推理加速（TensorRT / ONNX）", d: "TensorRT = NVIDIA 专用推理引擎，自动算子融合。ONNX = 跨框架通用格式。", course: null,
	         note: "【直觉】PyTorch 每个操作是独立算子调用，推理需「快」——TensorRT 把多小算子「熔炼」成大算子（减 kernel 启动开销），自动选最快 CUDA kernel，INT8 量化加速。<br><br>【技术】TensorRT 优化：Layer Fusion（Conv+BN+ReLU 合并）、Tensor Core 自动调用、Kernel Auto-Tuning。ONNX：统一格式，PyTorch→ONNX→TensorRT/OpenVINO/CoreML。torch.compile() 是 PyTorch 2.0 原生加速。<br><br>【应用】云端 GPU 推理用 TensorRT（2-5×加速），边缘用 ONNX Runtime，苹果用 Core ML。"}
	      ]},
	    {t: "MLOps", desc: "让机器学习流程像软件工程一样可靠",
	      items: [
	        {n: "实验管理（WandB / MLflow）", d: "记录每次实验的超参、指标、模型版本。方便对比和复现。", course: null,
	         note: "【直觉】调参两周后对着十几个 checkpoint 发呆。WandB 是「实验日记」——自动记录超参、loss 曲线、GPU 占用、代码版本。一行 wandb.init()+wandb.log() 搞定。<br><br>【技术】WandB：所有指标自动上传云端，实时可视化，支持对比和报告。MLflow：开源替代，模块化设计（Tracking/Projects/Models/Registry）。<br><br>【应用】WandB 是学术和工业默认实验管理工具。做好实验管理是 DL 工程化第一步——没它调参就是在迷宫里盲走。"},
	        {n: "CI/CD for ML", d: "自动化测试、自动部署。模型更新时自动跑回归测试。", course: null,
	         note: "【直觉】ML 的「bug」往往不报错——模型默默输出错误结果（数据漂移、特征缺失）。ML CI/CD 在传统软件测试上加 ML 特有验证：模型精度是否下降、数据分布是否偏移。<br><br>【技术】CI：代码检查→单元测试→数据验证→模型训练测试→模型评估。CD：模型上线（蓝绿/金丝雀部署）→线上监控→自动回滚。Google TFX、Kubeflow 提供完整 MLOps 工具链。<br><br>【应用】先把实验管理做好，再逐步加自动评估和部署。ML 系统技术债真实存在——早做工程化后面不还债。"}
	      ]}
	  ]
	},
	{
	  id: "l12", t: "第十二层：前沿与交叉领域", c: "#8a7f72", i: "🚀", inCourse: false,
	  s: [
	    {t: "AI for Science", desc: "AI 正在成为科学发现的工具",
	      items: [
	        {n: "AlphaFold2/3", d: "用 Attention 预测蛋白质 3D 结构。AlphaFold3 扩展到更多生物分子。", course: null,
	         note: "【直觉】蛋白质功能由 3D 形状决定，从序列预测结构困扰生物学 50 年。AlphaFold2 用 Transformer 做多序列比对，Attention 找共同进化氨基酸对（说明空间靠近），结构模块迭代优化坐标，达实验级精度。2024 年诺贝尔化学奖一半授予 Hassabis 与 Jumper（AlphaFold2 蛋白质结构预测），一半授予 David Baker（计算蛋白质设计）。<br><br>【技术】核心：(1) Evoformer——处理 MSA 和 Pair 表示，三角注意力保证物理约束；(2) Structure Module——IPA（3D 空间 Attention）+ 预测每个原子坐标。AlphaFold3 扩展到所有生物分子。<br><br>【应用】预测了 2 亿+蛋白质结构。药物设计——知道靶标结构就能设计结合药物。酶设计——设计新催化蛋白。"},
	        {n: "分子生成 / 材料发现", d: "用生成模型设计新药物分子。AI 加速材料筛选 1000x。", course: null,
	         note: "【直觉】传统药物发现需要从数百万化合物实验筛选——平均 10 年+10 亿美元。AI 药物设计在计算机里「凭空生成」可能有药效新分子，虚拟筛选淘汰 99.9%，只验证最有希望的。<br><br>【技术】方法：VAE-based（隐空间采样）、GAN-based（博弈生成）、Diffusion-based（噪声→3D 构象）、RL-based（化学空间探索）。评估用分子对接模拟。<br><br>【应用】Insilico Medicine 管线（18 个月 vs 3-4 年）。MIT 用 GNN 发现新型抗生素。AI for Science 是下一个爆发点——物理、化学、生物学。"}
	      ]},
	    {t: "可解释性（XAI）", desc: "知道模型为什么做出某个决策——黑盒不是借口",
	      items: [
	        {n: "注意力可视化", d: "看模型关注输入哪里。CV 中看热力图，NLP 中看词对齐。", course: "第8节",
	         note: "【直觉】模型说「X 光片有肺炎」——可视化就是「圈重点」：Grad-CAM 高亮 CV 模型关注区域，Attention 权重展示 NLP 词间对齐。<br><br>【技术】Grad-CAM：取最后一层卷积梯度，加权平均通道得热力图。NLP：看 Attention 权重矩阵。局限：Attention 不是唯一信息流，高层 Attention 往往分散不够可解释。<br><br>【应用】Grad-CAM 是模型 debug 基本工具——模型看背景判类别 = 学到虚假相关性。教学中让学生直观理解「模型在看什么」的好工具。"},
	        {n: "SHAP / LIME", d: "SHAP = 基于博弈论计算每个特征重要性。LIME = 局部近似解释。", course: null,
	         note: "【直觉】模型拒绝贷款申请，申请人问「为什么？」SHAP 告诉——「收入低贡献 0.3 分拒绝，但高学历贡献 0.1 分接受。」博弈论 Shapley 值公平分配「功劳」。<br><br>【技术】SHAP：对每个特征算边缘贡献权重平均。LIME：预测点附近采样，线性模型局部近似。局限：计算量大、「特征重要性≠因果性」。<br><br>【应用】金融风控、医疗诊断等需解释场景。SHAP 是工业标准——让你知道模型在「想什么」。"},
	        {n: "机械可解释性", d: "打开 Transformer，看具体哪个神经元在做什么。Anthropic 前沿研究方向。", course: null,
	         note: "【直觉】传统可解释性研究「关注哪里」，机械可解释性研究「内部每个神经元执行什么功能」。像大脑扫描升级为单神经元插电极。目标是打开 AI 黑盒。<br><br>【技术】方法：(1) 特征可视化——找让特定神经元激活最大的输入，(2) 电路分析——追踪输入→输出计算路径，(3) 字典学习（Sparse Autoencoders）——把激活分解为稀疏可解释特征。Anthropic 从 Claude 提取了百万级可解释特征。<br><br>【应用】发现 AI 欺骗行为（内部「欺骗回路」）。理解 LLM 推理和知识存储方式。可通向「可验证 AI 安全」——像审查代码而非黑盒测试。"}
	      ]},
	    {t: "AI 安全与伦理", desc: "AI 越强大，安全越重要",
	      items: [
	        {n: "对抗样本（FGSM / PGD）", d: "给图片加微小扰动让模型误判。理解模型盲点。", course: null,
	         note: "【直觉】熊猫照片加肉眼看不见的噪声（每像素<1/255），人看不出变化，模型却识别为「长臂猿」——置信度 99%。暴露深度学习模型脆弱性——学的是统计相关性，非真正「理解」。<br><br>【技术】FGSM：给输入加 sign(∇_x loss)，一步生成对抗样本。PGD：迭代多次，每步投影回 ε 球内，更强攻击。对抗训练：训练时加入对抗样本——最有效防御。<br><br>【应用】AI 安全研究起点——自动驾驶路牌被贴纸误导、人脸识别被眼镜骗过。培养「批判性使用 AI」思维——模型不是万能的，有系统性盲点。"},
	        {n: "隐私保护", d: "联邦学习 = 数据不出本地只传梯度。差分隐私 = 给数据加噪防泄露。", course: null,
	         note: "【直觉】医院想联合训练 AI 诊断，但数据不能出医院。联邦学习：各医院本地训练，只传梯度到中央合并。差分隐私：梯度/输出加噪声，让人无法反推出某人信息。<br><br>【技术】联邦学习：Server 分发模型→Clients 本地训练→上传梯度→FedAvg 聚合。DP-SGD：梯度加噪声+裁剪，ε（隐私预算）越小越私密但精度越低。<br><br>【应用】Google Gboard 用联邦学习改进键盘预测。Apple 用差分隐私收集数据。医疗 AI 中联邦学习是「数据不出院」核心方案。"},

      ]},
    {t: "具身智能与机器人", desc: "让 AI 从数字世界走向物理世界——能看、能说、能动",
      items: [
        {n: "具身智能（VLA）", d: "RT-2 = 机器人动作当 token，Transformer 端到端生成。Sim-to-Real = 仿真训练真实部署。", course: null,
	         note: "【直觉】为什么 ChatGPT 能写代码却不能叠衣服？LLM 活在文本世界里。具身智能把 AI 身体还给它们——VLA（Vision-Language-Action）把图像、语言、动作全 token 化，同一 Transformer 处理。<br><br>【技术】Google RT-2：Web 预训练 VLM+机器人数据微调→看到图像和指令→输出动作 token（7-DoF 位置/旋转/夹爪）→解码为控制信号。Sim-to-Real：仿真器训练+域随机化迁移到真实世界。<br><br>【应用】Figure 01（OpenAI 合作）实现语音+视觉+动作端到端人形机器人。特斯拉 Optimus 在大力投入。具身智能被认为是 AGI 的必要环节——AI 需要在物理世界互动才能真正理解世界。"}
	      ]}
  ]
}
];

var pathsData = [
{n: "CV工程师", d: "从图像分类到生成，从2D到3D", r: "地基 -> CNN（第2-4节） -> ResNet（第3节） -> YOLO检测（第5节） -> U-Net/SAM分割（第6-7节） -> Diffusion生成（第11-13节）", cls: "cv"},
{n: "NLP/LLM工程师", d: "从Transformer到Agent，从理解到生成", r: "地基 -> Attention（第8节） -> Transformer（第9节） -> nanoGPT（第14节） -> LoRA微调（第15节） -> SFT/DPO（第16节） -> RAG/Agent（第17-18节） -> 量化推理（第19节）", cls: "llm"},
{n: "推荐/搜广推", d: "从传统ML到深度推荐", r: "地基 -> 经典ML -> 深度学习基础 -> GNN（第9层） -> 序列推荐 -> 部署优化（第11层）", cls: "rec"},
{n: "AI研究/PhD", d: "全图通览后，选定1-2个方向深耕", r: "全图通览（第1节） -> 选定方向 -> 读顶会论文 -> 复现SOTA -> 提出改进", cls: "research"},
{n: "快速落地/创业", d: "用预训练模型+微调快速构建AI产品", r: "地基 -> PyTorch（第2节） -> 预训练模型（第3节起） -> LoRA（第15节） -> RAG（第17节） -> 部署优化（第11层）", cls: "product"},
{n: "机器人/具身智能", d: "从感知到决策，从仿真到现实", r: "地基 -> CNN（第2-4节） -> Transformer（第8-10节） -> RL（第8层） -> 多模态（第20-21节） -> VLA模型", cls: "robot"}
];

var courseColors = {
  cv: "#2a9d8f",
  llm: "#c25e00",
  rec: "#5c5470",
  research: "#4a7c59",
  product: "#d49500",
  robot: "#b85c5c"
};
