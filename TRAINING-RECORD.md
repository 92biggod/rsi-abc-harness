# 训练记录（0.4.2）

- [交付版 0.4.2 平台试运行汇总](https://swanlab.cn/@sxb/rsi-abc-harness/runs/s07xeuwh)
- [0.4.2 本地 2 核 / 4 GiB 十轮预检](https://swanlab.cn/@sxb/rsi-abc-harness/runs/q96vjrsk)
- [0.4.1 联合方案逐轮记录](https://swanlab.cn/@sxb/rsi-abc-harness/runs/kvenydot)
- [B 任务 R12 三轮记录](https://swanlab.cn/@sxb/rsi-abc-harness/runs/abcbr12q/chart)
- [前期 B 任务优化（队友 `jpc25`）](https://swanlab.cn/@jpc25/cpu-b-rul-harness)

---

## 1. 结论（平台回执口径）

下表对比 v5、v7 两次平台试运行的隐藏集指标。评测覆盖 A 的 4,886 条、B 的 200 条、C 的 9,687 条预测，两份回执的正式总分字段均为空。

| 任务 | 指标 | 官方基线（v5 对照） | 0.4.2（v7） | 变化 |
|---|---|---:|---:|---|
| C（主指标） | RMSE | 56.2259 | **52.6688** | −6.33% |
| C | MAE | 41.2866 | 37.4149 | −9.38% |
| C | NASA 非对称分 | 4.259e10 | 1.178e10 | −72.4% |
| B | RMSE | 720.066 | **489.166** | −32.1% |
| A | Macro-F1 | 0.4068 | **0.7753** | +90.6% |

「官方基线」列取自对照运行 v5（`trial/2836cd99`）。该镜像设为 `BASELINE_CONTROL=1`，每轮直接保留平台下发的父产物。

v7 回执（`trial/fb2c2573`）的任务 CPU 用量为 A 1,094.4 秒、B 430.9 秒、C 2,059.8 秒。
回执 `costs.training_seconds` 字段为 2,435.4 秒，模型调用共 43,897 token（提示 37,936，生成 5,961）。
A、B、C 各 10 轮的回执均记录了 `swanlab_enabled = true`。

> C 首轮公开验证 RMSE 为 46.1248，用于当轮选模。最终选中产物的隐藏集 RMSE 为 52.6688。

---

## 2. 0.4.2 的训练预算与线程测试

| 项目 | 证据 | 结论 |
|---|---|---|
| C 首轮 epoch 上限 432 → 200 | v6（`trial/ad125f23`）首轮 393 epoch、240.20 秒。v7 首轮 200 epoch、124.86 秒。两者首轮公开验证 RMSE 都是 46.1248，最终隐藏 RMSE 都是 52.6688 | 首轮训练耗时下降 48.0%。本地 200 和 432 epoch 对照选中了 SHA-256 完全相同的检查点 |
| BLAS 线程对照 | 2 核容器下跑 15 epoch，1 线程墙钟 1.421 秒 / CPU 1.416 秒，2 线程墙钟 1.415 秒 / CPU 2.827 秒，4 线程墙钟 2.502 秒 / CPU 4.996 秒。4 核容器下，1、2、4 线程墙钟分别为 1.401、1.419、1.462 秒 | 2 核实测 1 和 2 线程的墙钟只差 0.006 秒，单线程的 CPU 用量约为两线程的一半。镜像采用 1 线程 |

---

## 3. C 任务逐轮明细（v7 回执）

| 轮 | 数据阶段 | 方案 | epoch | 训练秒 | 父 RMSE | 镜像内最佳候选 RMSE | 平台候选 RMSE | 决策 |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | 01 | anchor | 200 | 124.86 | 52.8686 | 46.1248 | **46.1248** | 采纳（promote） |
| 2 | 01 | short | 16 | 11.82 | 46.1248 | 46.2580 | — | 保留父模型 |
| 3 | 01 | medium | 24 | 17.24 | 46.1248 | 46.2278 | — | 保留父模型 |
| 4 | 02 | short | 16 | 11.87 | 46.1403 | 46.2715 | — | 保留父模型 |
| 5 | 02 | medium | 24 | 18.41 | 46.1403 | 46.2399 | — | 保留父模型 |
| 6 | 02 | hold | 0 | — | 46.1403 | — | — | 保留父模型 |
| 7 | 03 | short | 16 | 11.83 | 46.0012 | 46.1145 | — | 保留父模型 |
| 8 | 03 | medium | 24 | 18.51 | 46.0012 | 46.1112 | — | 保留父模型 |
| 9 | 04 | short | 16 | 12.71 | 46.3461 | 46.4658 | — | 保留父模型 |
| 10 | 04 | medium | 24 | 18.03 | 46.3461 | 46.4540 | — | 保留父模型 |

父 RMSE 来自当轮公开验证。第 4、7、9 轮切换数据阶段，首轮采纳的同一份检查点在不同验证切片上得分有变化。
镜像内最佳候选取自回执配置中的 `best_observed`。只有通过镜像内筛选并提交的产物才有平台候选分。
第 2 至 10 轮均保留首轮检查点，第 6 轮执行 `hold`。该检查点的最终隐藏集 RMSE 为 52.6688。

---

## 4. 本地 2 核 / 4 GiB 十轮预检

按正式评测最低配置，在同一份 0.4.2 镜像中串行预检 A、B、C 各 10 轮。容器只读、断网，限制为 `--cpus=2 --memory=4g`。

| 项 | 结果 |
|---|---|
| 串行总墙钟 | 338.9 秒（5 分 39 秒） |
| 峰值内存 | 860,450,816 字节（821 MiB） |
| C 十轮墙钟 | 243.7 秒（首轮 43.2 秒） |
| C 首轮 | 200 epoch，训练 24.67 秒，选中 1 个成员，RMSE 46.1248 |
| 每轮成员数 | 全部为 1 |
| 第 6 轮 | hold，0 训练步 |
| 超时 / 内存超限 | 均未发生 |

各轮墙钟（C，秒）：43.2 / 16.1 / 20.2 / 15.7 / 16.9 / 14.6 / 16.3 / 17.9 / 20.1 / 18.8。
A 十轮合计 55.6 秒，B 十轮合计 39.5 秒。

预检用本地测试桩返回模型决策，并按 v7 回执的样本量模拟后续阶段。正式耗时还取决于平台调度、模型调用与阶段数据。[SwanLab 预检记录](https://swanlab.cn/@sxb/rsi-abc-harness/runs/q96vjrsk)

---

## 5. 历史线索（保留，作为方向证据）

### 5.1 B 任务前期优化（2026-11-24 ~ 11-27，队友机器）

[前期 B 任务 SwanLab 项目](https://swanlab.cn/@jpc25/cpu-b-rul-harness)

| # | 实验 | 公开验证最佳 RMSE | 时间 | 运行时长 | 内容 |
|---|---|---:|---|---:|---|
| 1 | bearing-rul-harness-20261124-210844 | 668.7688 | 11-24 20:08 | 34 s | 3 轮训练 |
| 2 | bearing-rul-harness-20261125-225436 | 668.2787 | 11-25 21:54 | 19 s | 3 轮，云端与本地一致 |
| 3 | bearing-rul-optimized-20261125-230647 | 650.1219 | 11-25 22:06 | 4 min 22 s | 3 轮，旧提交包原始实验 |
| 4 | bearing-rul-optimized-import-20261127-115511 | 650.1219 | 11-27 10:55 | 7 s | 历史日志导入（51 条逐 epoch） |

第 4 条为历史导入（`import_type = local_optimized_record_replay`），最佳检查点
`sha256:4798c31912c64061e6e3c3b2b78a05a3cad7213cfe72e2fb5dcf42caf6f50def`，
未舍入 RMSE `650.1219420046133`，与旧包 `all_trials_history.jsonl` 逐 epoch 记录一致。
离线导出见 [evidence/b-preliminary-runs-2026-11-24_11-27.csv](evidence/b-preliminary-runs-2026-11-24_11-27.csv)。

### 5.2 0.4.1 联合方案（A/B/C 十轮）

0.4.1 的逐轮父/候选与决策记录：https://swanlab.cn/@sxb/rsi-abc-harness/runs/kvenydot

0.4.2 把 C 首轮的训练上限压到 200 epoch。v6 与 v7 两份回执中，三个任务的方案序列逐轮一致。

| 任务 | 十轮方案序列（v6 = v7） | 十轮决策（v6 = v7） |
|---|---|---|
| A | `standard` × 10 | promote × 8（第 2、7 轮 keep） |
| B | `vibration95` × 3 → `curve80` / `curve95` / `curve80` / `curve50` / `curve95` / `curve80` / `curve50` | promote × 2（第 1、2 轮），其余 keep |
| C | `anchor` → `short` / `medium` 交替，第 6 轮 `hold` | promote × 1（第 1 轮），其余 retain |

B 的第 1、2 轮采纳候选，第 3 至 10 轮保留父模型。A 在同轮验证 Macro-F1 提高时采纳候选，两次试运行各采纳 8 次。
两份回执的最终隐藏集指标一致。

---

## 6. 复现

```sh
# 从赛事平台取得 SDK 放到 sdk/，然后
docker build --platform linux/amd64 -t zhixun-abc:0.4.2 .
docker run --rm --network none zhixun-abc:0.4.2 describe
docker save -o harness-abc-0.4.2.tar zhixun-abc:0.4.2

# 在正式最低资源限制下检查入口
docker run --rm --network none --cpus=2 --memory=4g zhixun-abc:0.4.2 describe
```

`describe` 可离线检查镜像入口与协议描述。十轮续训预检还需要赛事方提供的 baseline 数据包以及 SDK 测试桩。

C 镜像内固定使用 `OMP_NUM_THREADS=1 / OPENBLAS_NUM_THREADS=1 / MKL_NUM_THREADS=1`。线程对照实测见第 2 节。

---

## 7. 校验值

| 项 | 值 |
|---|---|
| 提交镜像 harness.tar | 85,893,120 字节，SHA-256 `2d3673fd862f2986048cbc534590c854ba9aa901e52974cc6f7fc615431fcf25` |
| 官方 SDK | 1.0.5，`bb5c616081de975d8f932a20cc00fd489e701e7585b69fdb52982bd669a305e6` |
| 源码清单 | [evidence/source-sha256.json](evidence/source-sha256.json)，29 个文件全部可复算 |
| v7 回执 | `trial/fb2c2573`，`costs.training_seconds` 为 2,435.4 秒，模型调用共 43,897 token |
| v5 对照回执 | `trial/2836cd99`，`BASELINE_CONTROL=1`，C 56.2259 |
| 前期 B 最佳检查点 | `4798c31912c64061e6e3c3b2b78a05a3cad7213cfe72e2fb5dcf42caf6f50def` |

---

## 8. 基础模型声明

提交表单声明的基础模型为 `Qwen/Qwen3.8-27B`，处于「参数量不超过 27B」的限制内。

基础模型用于从预设方案中选择当轮续训策略。返回值通过本地检查后用于执行训练，无法采用时走默认方案。
独立预测入口加载选中的工业模型产物执行数值推理，C 使用 833 参数 MLP。
