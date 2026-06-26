> 更新说明：第 4 节课改为以 Jupyter Notebook 为主的动手实战课，4 组消融实验合并到一个 notebook。

---

# 第 4 节课 · 训练技巧实战（动手版）

## 课程定位

- **类型**：动手实战课
- **时长**：90-120 分钟
- **目标**：通过消融实验理解优化器、BatchNorm、Dropout、学习率调度的作用

## 材料清单

### 课件（`课件.md`）

5 页：封面 → 优化器 → BatchNorm → Dropout+LR schedule → 总结

### 讲义（`讲义.md`）

教师手册

### Jupyter Notebooks

| 文件 | 内容 | 目标 | 预计时长 |
|------|------|------|:--------:|
| `04_1_training_tricks.ipynb` | 4 组消融实验 | 理解各训练技巧的作用 | 80 min |

## 课堂节奏

| 环节 | 时长 | 说明 |
|------|:----:|------|
| 开场 | 5 min | 回顾 + 本节目标 |
| 实验一：优化器 | 15 min | SGD / SGD+Momentum / Adam |
| 实验二：BatchNorm | 15 min | 有/无 BN |
| 实验三：Dropout | 15 min | 有/无 Dropout |
| 实验四：LR schedule | 15 min | Fixed / Step / Cosine |
| 汇总讨论 | 10 min | 各技巧效果对比 |
| **合计** | **90 min** | 不含 GPU 等待时间 |

## 质量检查点

- [ ] notebook 在干净环境中可运行
- [ ] 4 组实验趋势正确（如 BN 加速收敛、Dropout 降低过拟合）
- [ ] 学员能说出每个技巧的核心作用
