# 三任务工业预测 Harness（0.4.2）

本仓库对应参赛镜像 `zhixun-abc:0.4.2`：一个 Linux/amd64 镜像同时承接 **A**（六类气体识别）、
**B**（轴承剩余寿命）、**C**（C-MAPSS 涡扇发动机剩余寿命）。三项任务都从平台下发的父检查点续训，
镜像内不携带任何预训练权重。

- 官方 Harness SDK：`1.0.5`
- 运行时依赖：`numpy==2.5.5`、`swanlab==0.10.6`（实验记录，离线落盘）
- 提交镜像 `harness.tar`：85,893,120 字节，SHA-256 `2d3673fd862f2986048cbc534590c854ba9aa901e52974cc6f7fc615431fcf25`

## 结果

下表来自赛事平台 v5 与 v7 的试运行回执，隐藏评测覆盖 A 的 4,886 条、B 的 200 条、C 的 9,687 条预测。
v5 直接保留官方父产物，用于衡量 v7 续训带来的变化。回执中的正式总分字段为空。

| 任务 | 指标 | 官方基线（对照运行 v5） | 本版本（v7） | 变化 |
|---|---|---:|---:|---|
| C（主指标） | RMSE | 56.2259 | **52.6688** | −6.33% |
| C | MAE | 41.2866 | 37.4149 | −9.38% |
| C | NASA 非对称分 | 4.259e10 | 1.178e10 | −72.4% |
| B | RMSE（分钟） | 720.066 | **489.166** | −32.1% |
| B | MAE（分钟） | 518.721 | 375.873 | −27.5% |
| A | Macro-F1 | 0.4068 | **0.7753** | +90.6% |
| A | Accuracy | 0.4959 | 0.7839 | +58.1% |

- v7 回执 `trial/fb2c2573` 于北京时间 2026-11-28 11:52 创建，已完成训练和用量结算。
- 对照回执 `trial/2836cd99` 于北京时间 2026-11-27 22:32 创建，运行选项为 `BASELINE_CONTROL=1`。

v7 回执的任务 CPU 用量为 A 1,094.4 秒、B 430.9 秒、C 2,059.8 秒。账单字段 `training_seconds` 为 2,435.4 秒，模型调用共 43,897 token。

## 本地 2 核 / 4 GiB 十轮预检

按正式评测的最低配置，用同一份 0.4.2 镜像在只读、断网且设为 `--cpus=2 --memory=4g` 的容器内串行运行 A、B、C 各 10 轮。
预检由本地测试桩返回大模型决策，后续阶段按平台回执的样本量模拟。

| 项 | 结果 |
|---|---|
| 串行总墙钟 | 338.9 秒（5 分 39 秒） |
| C 十轮墙钟 | 243.7 秒 |
| 峰值内存 | 860,450,816 字节（821 MiB） |
| C 首轮 | 200 epoch，24.67 秒，选中 1 个成员 |
| C 轮次超时 | 无（`c_timed_out = false`） |
| 内存超限 | 无 |

C 十轮所选产物均为单成员，第 6 轮按策略直接保留父模型。首轮公开验证 RMSE 复现为 46.1248。[本地十轮预检 SwanLab](https://swanlab.cn/@sxb/rsi-abc-harness/runs/q96vjrsk)

## 构建

从赛事平台取得官方 `contest-harness-sdk-1.0.5.tar.gz`，放在仓库根目录的 `sdk/` 下：

```
sdk/contest-harness-sdk-1.0.5.tar.gz
sha256 bb5c616081de975d8f932a20cc00fd489e701e7585b69fdb52982bd669a305e6
```

构建前请核对 SDK 的 SHA-256。[赛事平台 SDK 1.0.5 下载](https://rsi-index.115.zone/downloads/contest-harness-sdk-1.0.5.tar.gz)

```sh
docker build --platform linux/amd64 -t zhixun-abc:0.4.2 .
docker run --rm --network none zhixun-abc:0.4.2 describe
docker save -o harness-abc-0.4.2.tar zhixun-abc:0.4.2
```

镜像入口为 `/app/harness.py`，分发逻辑在 `app/contest_submission.py`，三个任务的实现分别在
`app/a/`、`app/b/`、`app/c/`。源码与镜像逐字节核对的摘要见
[evidence/source-sha256.json](evidence/source-sha256.json)（29 个文件，全部可复算）。

## 设计要点

| 决策 | 依据 |
|---|---|
| C 首轮上限 200 epoch（此前 432） | 本地对照选中同一份检查点。v6 到 v7 的首轮平台训练耗时从 240.20 秒降至 124.86 秒，两次隐藏 RMSE 均为 52.6688 |
| C 选模 | v7 首轮采纳单成员，后续九轮保留它。本地 2 核 / 4 GiB 预检复现该序列 |
| BLAS 线程固定为 1 | 2 核下 1 线程与 2 线程的 15-epoch 墙钟都是约 1.42 秒，但 2 线程的 CPU 用时接近翻倍；4 线程在 2 核上反而慢到 2.50 秒 |
| 在同一轮公开验证集上比较父模型和候选 | 使用平台返回的候选验证摘要选模。C 先在镜像内筛选并重载候选，再核对平台评分 |
| SwanLab 离线记录 | v7 的 30 轮回执均记录 `swanlab_enabled = true` |
| 父检查点完整性 | 加载时按平台清单解析文件并核对摘要。C 支持在可选成员缺失或摘要不符时使用主检查点 |

## 训练记录

逐轮父/候选分数、决策、耗时与两版对照见 **[TRAINING-RECORD.md](TRAINING-RECORD.md)**。

- [交付版 0.4.2 平台试运行汇总](https://swanlab.cn/@sxb/rsi-abc-harness/runs/s07xeuwh)
- [0.4.1 联合方案逐轮记录](https://swanlab.cn/@sxb/rsi-abc-harness/runs/kvenydot)
- [B 任务 R12 三轮记录](https://swanlab.cn/@sxb/rsi-abc-harness/runs/abcbr12q/chart)
- [前期 B 任务优化（队友 `jpc25`）](https://swanlab.cn/@jpc25/cpu-b-rul-harness)

## 声明

- 提交表单声明的基础模型为 `Qwen/Qwen3.8-27B`（处于参数量不超过 27B 的限制内）。
- 基础模型参与选择当轮续训方案。独立预测入口加载选中的工业模型产物，对评测样本执行数值推理。
- 三项任务从平台下发的父检查点续训。训练使用当轮官方训练集，验证集用于选模和评分。
