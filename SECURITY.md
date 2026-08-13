# Security Policy

## 支持范围

|版本|状态|
|-|-|
|`main` 分支最新 commit|✅ 会修|
|已发布 tag（v1.0 及以上）|✅ 会修|
|开发中的 feature 分支|❌ 不承诺|

## 报告漏洞

**请不要在公开 Issue 里报漏洞**——公开报会让攻击面在修复前就暴露给别人。

请通过以下渠道私密报告：

- **邮箱**：<在这里填一个你能收到邮件的地址>
- 或：GitHub 的 [Private Vulnerability Reporting](https://github.com/zhoujiaxin11/StatCraft_bench/security/advisories/new)（右上角 Security → Report a vulnerability）

报告里请包含：

1. **漏洞类型**（例如：远程代码执行、路径穿越、任意文件读、prompt 注入导致的评分绕过、groundtruth 泄漏路径……）
2. **受影响的文件/模块**（如 `framework/docker_executor.py` 的 exec 命令拼接）
3. **复现步骤**（最小化 PoC，最好带一条 shell 命令）
4. **潜在影响**（本地误评分？CI 环境泄漏？groundtruth 被 exfil？）
5. 可选：你的联系方式（如果希望后续沟通）

## 响应时间

|阶段|承诺时间|
|-|-|
|确认收到|3 个工作日内|
|初步分级（严重/一般/信息级）|7 个工作日内|
|发布修复|严重级 30 天内 / 一般级下一个 minor 版本|

如果你在 30 天内没收到任何回复，可以直接开一个不含细节的 Issue 提醒（如 "已通过邮箱报送安全问题，等待响应"）。

## 已知的**非**漏洞

以下几种情况不是本项目的安全问题，请不要报告：

- **Agent 通过 API 联网访问外部服务**——`--agent http_api` 本来就是通过 HTTP API 调外部模型的，暴露 API key 是使用方的责任（配置在 `configs/models.yaml` 的 `api_key_env`，通过环境变量注入）
- **Docker 容器逃逸类风险**——本项目使用 Docker 作为沙箱只是"隔离评测环境"，不是"防御恶意 agent"。如果你的评测场景需要抵抗恶意 agent，请自行加固（gVisor / Kata Containers / 更严格的 seccomp）
- **模型输出中包含敏感信息**——那是模型问题，不是本 bench 问题
- **性能问题 / DoS**（除非能证明是远程可触发的资源耗尽）

## 我们**会**认真对待的漏洞类型

- Runner / scorer / verifier 的**代码执行**漏洞（能让贡献的 PR 在 CI 里跑出 shell）
- **groundtruth 泄漏**路径（例如某 PR 通过特殊命名让 hidden groundtruth 意外暴露）
- **评分绕过**（例如 agent 只需要写某个特殊字符串就能拿满分）
- **CI 环境凭据泄漏**（例如 pytest fixture 意外把 secrets 写到 stdout）
- 依赖的第三方库有**已知 CVE**且我们能升级修

## 免责声明

本项目**不是生产系统**，是学术 benchmark。使用者应假定：

- 评测结果仅用于研究比较，不作任何决策依据
- 任务/instruction/groundtruth 可能被 fork 者修改，官方仓库的 tag 才代表定版
- Docker 沙箱**不足以**抵抗恶意 agent 的持久化攻击，请勿在评测方式不受控的场景使用

## 致谢

我们会在 CHANGELOG / release notes 里公开致谢有效的漏洞报告（除非你要求匿名）。
