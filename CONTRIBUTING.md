# Contributing

感谢有兴趣给这个数据分析 AgentBench 提交贡献。本文件只讲**流程**——具体的任务规范（目录结构、字段格式、评分协议）在 [TASK_AUTHORING.md](TASK_AUTHORING.md)，它才是 PR 的实质标准。

---

## 可贡献的类型

|类型|示例|入口|
|-|-|-|
|**新任务**|新增一道数据分析题（instruction.md + solution/solve.py + groundtruth）|遵循 [TASK_AUTHORING.md](TASK_AUTHORING.md)|
|**修 bug**|scorer 逻辑错误、runner 崩溃、状态桶分类错|直接 PR，附最小复现|
|**框架改进**|新的能力标签、新的评分维度|**先开 Issue 讨论**再动手，避免白干|
|**文档**|README / TASK_AUTHORING 表述澄清、跑通示例|直接 PR|

**不接受**：仅仅重命名、批量格式化、加空行的 PR。

---

## 提 PR 之前——本地必过二件套

**这就是 CI 会跑的东西**，本地过了 CI 也不会挂：

```bash
# 1. Oracle 必过（solve.py 能拿到 ≥ 0.99 分）
python3 -m framework.runner \
  --task tasks/education_academia/159_gaokao_reform \
  --agent oracle --check-oracle

# 2. 单测全绿（框架守门 + 任务级 pytest）
python3 -m pytest -q
```

**这两条任何一条挂 = PR 挂**。对新任务，把上面的 task_id 换成你新加的那个。

---

## PR 流程

1. Fork 本仓库，从 `main` 拉一个 feature 分支：
   ```bash
   git checkout -b feat/task-<task_id>
   ```
2. 完成改动，本地跑上面二件套确认全绿
3. 提交 PR，模板会自动填充（如果没有模板，PR 描述里说明三件事）：
   - **改了什么**（一句话）
   - **为什么改**（引用 issue 编号或说明动机）
   - **验证方式**（贴出上面二件套的最后一行输出）
4. 等待 CI 完成 + 至少一位 maintainer review
5. Review 意见落地 → CI 重新绿 → 合并

---

## 新任务贡献者额外提醒

**这是最容易翻车的部分**，仔细读一遍 [TASK_AUTHORING.md](TASK_AUTHORING.md)，尤其：

- **零泄漏**：`instruction.md` 里不能出现真值、评分权重、控制组、公式实现——CI 会用禁词扫描
- **单一 factoid**：主输出必须是一份 `outcome.json`，字段由 `schema.json` 声明
- **groundtruth 隔离**：真值放在 `groundtruth/public/<task_id>.json`，**不能进 `tasks/` 目录**（这个不是习惯，是防止训练数据污染）
- **能力标签必填**：`task.toml` 里的 `capability_map` 只能选[受控词表](TASK_AUTHORING.md#能力标签词表)里的值

**新任务 PR 必须自带**：一份 oracle 分数截图/日志（证明 ≥ 0.99）。

---

## 代码风格

- Python 3.11+，标准库优先。引入新依赖需要在 PR 描述里说明必要性
- 缩进 4 空格，字符串双引号
- 函数 docstring 用简中文或英文都行，但**同一文件内保持一致**
- 提交信息用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：
  ```
  feat(tasks): add 160_stroke_prediction task
  fix(scorer): handle NaN in weighted aggregation
  docs(readme): fix quick-start command
  ```

---

## 报告漏洞 / 安全问题

**不要开 public issue 报安全问题**——用 [SECURITY.md](SECURITY.md) 里的私密渠道。

---

## 行为规范

对事不对人。技术分歧可以很直接，但不针对贡献者本人。恶意 PR / 拒不改错 / 反复提交同一被拒版本，maintainer 有权关闭并屏蔽。

---

## 许可

提交 PR 即视为你同意将该贡献以 [Apache-2.0](LICENSE) 许可发布。
