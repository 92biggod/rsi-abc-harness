# sdk/

本目录用于放置赛事方下发的官方 Harness SDK 压缩包，**不入库**（见根目录 `.gitignore` 的 `*.tar.gz`）。

0.4.2 使用：

```
contest-harness-sdk-1.0.5.tar.gz
sha256 bb5c616081de975d8f932a20cc00fd489e701e7585b69fdb52982bd669a305e6
```

取得后放成 `sdk/contest-harness-sdk-1.0.5.tar.gz`，`Dockerfile` 会自动解包并安装。
SHA-256 不符时请不要构建——镜像与提交物必须同源。
