# 三任务工业预测 Harness

本仓库对应参赛镜像 `zhixun-abc:0.3.0`，包括 A 组气体分类、B 组轴承寿命 R12、C 组发动机寿命 R7。
训练通过平台宿主读取父检查点和历史，预测通过同一镜像的独立入口运行。

## 构建与检查

从赛事平台取得官方 `contest-harness-sdk-1.0.4.tar.gz`，放在仓库根目录下的 `sdk/`。
本版本使用的 SDK 压缩包 SHA-256 为 `7afd282d6b5fd889d64d0515a67d169806970e36200689b18d2040026167a4af`。
在仓库根目录执行以下命令。

```sh
docker build --platform linux/amd64 -t zhixun-abc:0.3.0 .
docker run --rm --network none zhixun-abc:0.3.0 describe
docker save -o harness-abc-0.3.0.tar zhixun-abc:0.3.0
```

镜像入口为 `/app/harness.py`，分发逻辑在 `app/contest_submission.py`。A、B、C 的实现分别在
`app/a/`、`app/b/`、`app/c/`。数据和父检查点由赛事平台提供。程序文件已逐字节核对导出的镜像，
源码及 Dockerfile 的摘要见 [source-sha256.json](evidence/source-sha256.json)。

## 公开验证与训练记录

| 方法 | 公开验证结果 | 样本数 |
|---|---|---:|
| A 预设方案重放 | Macro-F1 0.934611 | 1,244 |
| A 本地 5090 自主决策 | Macro-F1 0.921593 | 1,244 |
| B R12 本地 10 轮续训 | RMSE **428.1466**（第 2 轮 428.2294） | 1,679 |
| C R7 四成员 | RMSE **45.9189** | 23,049 |

逐轮轨迹（每轮父模型、候选分数、是否采纳）与前期 4 条实验的原始导出见
**[TRAINING-RECORD.md](TRAINING-RECORD.md)**。

[SwanLab 实验索引](https://swanlab.cn/@sxb/rsi-abc-harness) 汇集本版本已有的记录。
A 的两次实验和 B R12 的逐轮指标来自队友交付文件，可在 [ab-rounds.json](evidence/ab-rounds.json)
中查到来源摘要。C 的 [R7 采纳配方](https://swanlab.cn/@sxb/rsi-abc-harness/runs/cx2gt2cb) 从原始离线运行同步。
平台封闭评测结果以正式回执为准。
