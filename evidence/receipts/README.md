# 平台试运行回执（原始响应）

这两个文件是赛事平台 `/runs/{run_id}/report` 接口返回的 JSON，用于支撑
[../../TRAINING-RECORD.md](../../TRAINING-RECORD.md) 中的结论表与逐轮明细。
保存的是**接口原样响应**，除下表注明的两个字段外没有做任何改写。

| 文件 | 运行 | 提交版本 | 说明 |
|---|---|---:|---|
| `trial-fb2c2573.json` | `fb2c25733cd7b32998bf261d8bafede7` | v7 | 交付版 0.4.2，C 首轮上限 200 epoch |
| `trial-2836cd99.json` | `2836cd994dd1bb408f86b1c7fd8c2008` | v5 | 对照组，镜像内 `BASELINE_CONTROL=1`，每轮只保留平台父产物 |

## 1. 脱敏与摘要

只删除了两个字段：`run.team_name`、`run.team_id`。其余字段（含 `run.id`、
`submission_id`、`model_call_ids`、各类 `artifact_id`）保留，以便与提交记录对齐。
回执中不含令牌、不含下载链接，也不含本机路径。

| 文件 | 字节 | SHA-256（本仓库版本） |
|---|---:|---|
| `trial-fb2c2573.json` | 248,756 | `74398d270bdca8d0d02b203438f3bdbab3ead2cf611920c5ffb680c1c73b6b5f` |
| `trial-2836cd99.json` | 35,811 | `2c3d31dad57fa75a5067dd33a240af0d2be84982218e491cdbdcafef4be99e02` |

平台返回的未删字段版本（含团队信息）未入库，其摘要记录在此供追溯：

| 文件 | 字节 | SHA-256（接口原始响应） |
|---|---:|---|
| `trial-fb2c2573.json` | 349,856 | `db00f4793781c8e3e2188b04324492de144ca174505ae2b7ae7b85f9dbedd438` |
| `trial-2836cd99.json` | 36,789 | `514378e3dd5b1bdcea4fd61244fb630a37771eacbf65c5b0a641ba3edcd379f2` |

## 2. 文档表格与回执字段的对应关系

`TRAINING-RECORD.md` 第 1、3 节的每一个数字都来自下面的字段，可直接复算：

| 文档中的列 / 指标 | 回执字段路径 |
|---|---|
| 任务隐藏集 RMSE / MAE / NASA 分 | `report.workspace_report.datasets[key].metrics` |
| 预测条数 | `report.workspace_report.datasets[key].prediction_count` |
| 任务 CPU 秒 | `report.workspace_report.datasets[key].cpu_seconds` |
| 轮次、方案、epoch 数 | `…datasets[key].rounds[i].configuration.plan` / `.epochs_done` |
| 训练秒 | `…rounds[i].configuration.training_seconds` |
| 父模型 RMSE（当轮公开验证） | `…rounds[i].configuration.parent.rmse` |
| 镜像内最佳候选 RMSE | `…rounds[i].configuration.best_observed.candidate.rmse` |
| 平台候选 RMSE | `…rounds[i].configuration.candidate.rmse` |
| 决策 | `…rounds[i].decision`（`promote` / `retain`） |
| 训练总时长 | `report.costs.training_seconds` |
| 模型调用 token | `report.costs.prompt_tokens` / `completion_tokens` / `total_tokens` |

## 3. 复现

回执来自赛事平台的登录态接口，需要本队账号：

```sh
python scripts/fetch_report.py fb2c2573     # 交付版 v7
python scripts/fetch_report.py 2836cd99     # 对照组 v5
```

脚本落盘到 `runs/reports/trial-<前 8 位>.json`，与本节的两份文件同源。
若要核对数值，直接读取 `report.workspace_report.datasets` 即可，不需要重跑镜像。

## 4. 阅读提示

- `v5` 的 `trial_note` 记录了它的性质：`BASELINE_CONTROL=1`，每轮不训练、直接保留平台下发的父产物。
  因此它的 `costs.training_seconds`（1,093.9 秒）是**容器空转与评测开销**，不是训练时间，不能与 v7 的
  2,435.4 秒直接比较。
- 两份回执的 `run.score` 均为空：试运行阶段平台不回填正式总分，文档里也没有引用任何总分。
- 回执中的 `…rounds[i].configuration.parent.rmse` 是**当轮公开验证切片**上的分数，与最终隐藏集分数不同一个口径。
  例如第 1 轮父模型 52.8686 是平台 baseline 在该切片上的分，而 baseline 的隐藏集 RMSE 是 56.2259。
