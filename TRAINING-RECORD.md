# 训练记录（完整链路）

本文把两段分别在不同机器上完成的训练汇总到一处，作为提交表单「训练记录链接」的落点。
原始导出见 [evidence/b-preliminary-runs-2026-11-24_11-27.csv](evidence/b-preliminary-runs-2026-11-24_11-27.csv)。

- 交付版实验：https://swanlab.cn/@sxb/rsi-abc-harness （公开，无需登录）
- 前期实验（队友 jpc25）：https://swanlab.cn/@jpc25/cpu-b-rul-harness

## 1. 结论

| 任务 | 指标 | 官方 baseline | 交付版 |
|---|---|---:|---:|
| C（主指标） | RMSE | 52.87 | **45.9189**（4 核 4 成员）/ 45.9522（2 核 3 成员） |
| B | RMSE | 681.1598 | **428.2294** |
| A | Macro-F1 | — | **0.934611**（Acc 97.75%） |

C 伴随指标：MAE 32.2011、高估比 56.12%（门槛 57%）、NASA 评分 41494（门槛 160000）。

## 2. 第一段：B 任务基线优化（2026-11-24 ~ 11-27，队友机器）

项目：https://swanlab.cn/@jpc25/cpu-b-rul-harness

| # | 实验 | 公开验证最佳 RMSE | 时间 | 运行时长 | 记录内容 |
|---|---|---:|---|---:|---|
| 1 | [bearing-rul-harness-20261124-210844](https://swanlab.cn/@jpc25/cpu-b-rul-harness/runs/0uz7fuof/chart) | 668.7688 | 11-24 20:08 | 34 s | 3 轮训练（subprocess） |
| 2 | [bearing-rul-harness-20261125-225436](https://swanlab.cn/@jpc25/cpu-b-rul-harness/runs/j5ibfaay/chart) | 668.2787 | 11-25 21:54 | 19 s | 3 轮训练，云端与本地一致 |
| 3 | [bearing-rul-optimized-20261125-230647](https://swanlab.cn/@jpc25/cpu-b-rul-harness/runs/m95jjyvc/chart) | 650.1219 | 11-25 22:06 | 4 min 22 s | 3 轮，对应旧提交包原始实验 |
| 4 | [bearing-rul-optimized-import-20261127-115511](https://swanlab.cn/@jpc25/cpu-b-rul-harness/runs/sepwi2if/chart) | 650.1219 | 11-27 10:55 | 7 s | 历史日志导入，对应旧包 **51 条逐 epoch 日志** |

第 4 条为历史导入（`import_type = local_optimized_record_replay`），最佳检查点
`sha256:4798c31912c64061e6e3c3b2b78a05a3cad7213cfe72e2fb5dcf42caf6f50def`，
未舍入 RMSE `650.1219420046133`，与旧包 `all_trials_history.jsonl` 逐 epoch 记录一致。

> 该项目为私有，需要登录才能打开页面。上表数值与本仓库附带的 CSV 是同一份记录的离线导出，
> 二者与云端记录一一对应（实验名、时间、时长、`baseline_rmse = 681.1598`、`best_round = 2`）。

## 3. 第二段：交付版 R12（2026-11-27，本机）

项目：https://swanlab.cn/@sxb/rsi-abc-harness

- B 任务 10 轮续训：https://swanlab.cn/@sxb/rsi-abc-harness/runs/abcbr12q/chart
- A / C 任务记录同在该项目内

逐轮轨迹（本地按平台规则复现「每轮采纳候选作为下一轮父检查点」）：

| 任务 | 轮 | 父 | 候选 | 决策 | 方案 |
|---|---:|---:|---:|---|---|
| A | 1 | 0.714030 | 0.911592 | 采纳 | standard |
| A | 2 | 0.911592 | 0.911746 | 采纳 | ethanol_ethylene_review |
| A | 3 | 0.911746 | 0.921593 | 采纳 | ethanol_ethylene_logistic |
| A | 4 | 0.921593 | **0.934611** | 采纳 | small_batch |
| B | 1 | 681.1598 | 428.3720 | 采纳 | vibration95 |
| B | 2 | 428.3720 | **428.2294** | 采纳 | vibration95 |
| C | 1（冷启动） | 52.87 | **45.9189** | 采纳 | 4 成员集成（工况分组标准化） |

C 在 2 核 / 4 GiB（正式评测规格）下按墙钟自适应降为 3 成员：RMSE 45.9522、MAE 32.2461、
高估比 55.45%、耗时 206.8 s，不超时。A、B 在 2 核下逐位不变。

## 4. 两段之间的关系

两条**独立训练线**，均从官方 baseline 冷启动，不存在权重继承：

- 第一段（队友）确定方向：工况分组建模、上尾加权、按 epoch 早停；
- 第二段（R12）在本机用交付镜像重跑满轮次，B 从 681.1598 直接落到 428.2294，A 从 0.714030 走到 0.934611。

交付镜像内不携带预训练权重，结果完全依赖平台 10 轮续训，符合「必须基于父检查点续训」的约束。
第一段是方向性证据，第二段是交付证据。

## 5. 复现

```bash
docker build --platform linux/amd64 -t zhixun-abc:0.3.0 .
docker run --rm zhixun-abc:0.3.0 describe
docker save -o harness-abc-0.3.0.tar zhixun-abc:0.3.0
```

C 的关键环境变量（缺省会导致只训完 1 个成员，RMSE 退到 46.1248）：

```
OMP_NUM_THREADS=4  OPENBLAS_NUM_THREADS=4  MKL_NUM_THREADS=4
```

## 6. 校验值

| 项 | 值 |
|---|---|
| 前期 B 最佳检查点 sha256 | `4798c31912c64061e6e3c3b2b78a05a3cad7213cfe72e2fb5dcf42caf6f50def` |
| 前期 B 未舍入 RMSE | 650.1219420046133 |
| 交付版 B | RMSE 428.2294 / MAE 367.7265（1679 条） |
| 交付版 C | RMSE 45.9189 / MAE 32.2011 / 高估比 56.12% / NASA 41494（23049 条） |
| 交付版 A | Macro-F1 0.934611 / Acc 97.75%（1244 条） |

## 7. 基础模型声明

提交表单声明的基础模型：`Qwen/Qwen3.5-4B`。

本方案评分路径为确定性数值算法（C：833 参数 MLP × 4 成员；A/B：平台基线检查点上的确定性策略序列），
未把模型生成内容作为预测结果来源。镜像按平台协议保留模型调用能力，但结果不依赖其输出。
