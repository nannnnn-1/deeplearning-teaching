> 更新说明：根据第一节课反馈，第 2 节课改为以 Jupyter Notebook 为主的动手实战课，课件和讲义大幅压缩。

---

# 第 2 节课 · 从感知机到 CNN（动手版）

## 课程定位

- **类型**：动手实战课
- **时长**：90-120 分钟
- **目标**：理解神经网络从感知机到 CNN 的演进，亲手跑通 3 个 demo

## 材料清单

### 课件（`课件.md`）

5 页：封面 → 感知机 → MLP → CNN → 对比与预告

### 讲义（`讲义.md`）

教师手册：开场话术、每个 notebook 的易错点、巡视关注点、讨论问题

### Jupyter Notebooks

| 文件 | 内容 | 目标 | 预计时长 |
|------|------|------|:--------:|
| `02_1_perceptron.ipynb` | NumPy 手写感知机（AND/OR/XOR） | 理解权重更新、线性不可分 | 15 min |
| `02_2_mlp_mnist.ipynb` | PyTorch MLP 训练 MNIST | 5 epochs 达到 95%+ | 25 min |
| `02_3_cnn_mnist.ipynb` | 手写卷积 + CNN 训练 MNIST | 3 epochs 达到 98%+ | 35 min |

## 课堂节奏

| 环节 | 时长 | 说明 |
|------|:----:|------|
| 开场 | 5 min | 回顾上节 + 本节 3 个目标 |
| Notebook 1 | 15 min | 感知机 |
| Notebook 2 | 25 min | MLP on MNIST |
| Notebook 3 | 35 min | CNN on MNIST |
| 收尾讨论 | 10 min | MLP vs CNN 对比 |
| **合计** | **90 min** | 不含学员练习时间 |

## 质量检查点

- [ ] 3 个 notebook 在干净环境中可运行
- [ ] `02_2_mlp_mnist.ipynb` 5 epochs 测试准确率 > 95%
- [ ] `02_3_cnn_mnist.ipynb` 3 epochs 测试准确率 > 98%
- [ ] 卷积核和特征图可视化清晰
- [ ] 参数量计算与 `model.parameters()` 输出一致
