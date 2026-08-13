# Harbor-Compatible Data Analysis Bench Sample

---

## 一、设计定位

面向"数据分析垂直领域"的 Agent 评测基准，兼顾三个目标：

1. **未来兼容**：题目难度分层（Easy / Hard / Extreme），Extreme 题保留天花板给未来模型
2. **公平比较**：跨模型 HTTP API 统一 harness，Docker 沙箱隔离
3. **可归因**：Factoid 化 JSON 答案 + 能力级加权部分分，能定位模型能力短板

**参考的四篇论文**：
- **DABStep** (Adyen/HuggingFace, 2025)：Factoid 答案 + 灵活容差评估协议
- **AgenticDataBench** (Tsinghua/Ant, 2026)：Skill 树 + 技能级归因
- **StatABench** (南科大, 2026)：垂直领域深挖 + 双轨（闭卷 + 开放）
- **SkillsBench** (Harbor 团队, 2026)：任务打包结构 + oracle 必过 CI + 失败分类学

**沿用了原 159 项目**：
- 加权分档判分思路 → framework/scorer.py（升级为通用打分器）
- HTTP API 跨模型运行 harness → framework/agent_adapter.py
- `groundtruth.json` 里的 tolerance / 分档容差字段设计
- `task.toml` 元数据抽象

---

## 二、目录结构

```
harbor_sample/
├── README.md                         # 本文件
├── framework/                        # 通用框架层(全 bench 共享)
│   ├── scorer.py                     # Factoid + 加权部分分打分器
│   ├── normalize.py                  # 数值/字符串/列表归一化
│   ├── runner.py                     # 单题执行入口
│   ├── batch.py                      # 多题并发/串行调度(带磁盘门)
│   ├── agent_adapter.py              # HTTP API 通用 Agent 接口
│   ├── docker_executor.py            # 沙箱执行器(Docker/subprocess 双通道)
│   ├── aggregate.py                  # 多 seed 聚合(稳定分+bootstrap CI+配对检验)
│   ├── summary.py                    # trials/ 目录成绩汇总
│   ├── verifier_wrapper.py           # pytest → 加权分数桥接
│   ├── statuses.py                   # 单一真源:infra/model_fail/ok 状态桶
│   ├── taxonomy.py                   # 单一真源:scenario/difficulty/capability 白名单
│   ├── compat_matrix.py              # 模型×网关兼容性预检(v1.0 新增)
│   └── ci_validate.py                # 离线结构+内容校验(CI L1/L2)
├── docker/                           # 分层镜像(场景级,不是每题级)
│   ├── base.Dockerfile
│   └── social.Dockerfile             # 159 目前挂在这上面;后续按 scenario 拆
├── tasks/                            # 目前只 ship 1 道 sample task;
│   └── education_academia/           # 其余 11 个 scenario 是白名单占位,
│       └── 159_gaokao_reform/        # 逐步补题.
│           ├── task.toml             # 元数据 + 能力标签 + image 声明
│           ├── instruction.md        # 去泄漏版本
│           ├── schema.json           # 答案字段定义(给 agent 看)
│           ├── environment/
│           │   └── README.md         # 数据说明(实际数据挂载运行时提供)
│           ├── solution/
│           │   └── solve.py          # Oracle 参考解(CI 必过判分器)
│           └── tests/
│               └── test_outputs.py   # pytest 断言(补充图/CSV 检查)
├── groundtruth/                      # 与 tasks/ 物理隔离
│   ├── public/                       # dev 集真值(公开)
│   │   └── 159_gaokao_reform.json    # 真值 + 权重 + tolerance + 分档规则
│   └── hidden/                       # 隐藏 test 集(不进 git)
├── registry/
│   └── v1.0.json                     # 数据集版本清单
├── configs/
│   ├── models.yaml                   # 模型 HTTP API 配置(密钥用 env 引用)
│   └── evaluation-plans.yaml         # 分层评测计划
├── tests/                            # 框架自测(pytest -q)
└── .github/workflows/ci.yml          # L1 结构/L2 内容/L3 oracle 三层门禁
```

---

## 三、关键设计决策

### 1. 任务打包格式 ≈ Harbor 风格（暂未 100% 兼容）

采用 `task.toml + instruction.md + environment/ + solution/ + tests/` 的类 Harbor
目录结构。**注意**：SkillsBench/BenchFlow 现网规范用的是 `task.md` (YAML frontmatter,
`schema_version "1.3"`) + `oracle/solve.sh` + `verifier/test.sh` 写 reward 到
`/logs/verifier/reward.txt`，跟本仓库这一版的 `task.toml` + `solution/solve.py` +
`tests/test_outputs.py` 是**不同协议**——所以 `harbor run -p ./tasks` 目前**跑不了**。
后续要接入的话有两条路：

1. 写一个 `export_to_skillsbench.py` 转换器，把本仓库的 task 包翻译成 `task.md`
   骨架 + `oracle/solve.sh`（包一层 `python solution/solve.py`）+
   `verifier/test.sh`（包一层 `pytest tests/`）。
2. 直接把本仓库的 task 目录改造成 native `task.md` 结构，`framework/runner.py`
   保持自研，只在需要上 Harbor leaderboard 时用转换器。

### 2. 答案接口 = Factoid JSON（DABStep 风格）

Agent 只需产出一份 `outcome.json`，字段类型和结构由 `schema.json` 声明。
辅助交付物（`predictions.csv` 等）只做存在性检查，主判分权重放在 factoid 字段。

### 3. Groundtruth 物理隔离

真值放在 `groundtruth/public/` 与 `groundtruth/hidden/`，**绝不进入任务目录**。
Agent 挂载 `tasks/**/` 时看不到真值。防泄漏。

### 4. 能力级归因

每题 `task.toml` 声明能力标签（capabilities），`groundtruth.json` 的 weights 按能力分组。
Scorer 输出既有总分，也有 per-capability 分数，画雷达图定位能力短板。

标签来自一份**受控词表**（42 个）。词表的唯一定义在 `framework/taxonomy.py::CAPABILITIES`，
文档描述在 `TASK_AUTHORING.md` §七；新增标签需要提 spec 变更 PR，两处同步改。

**注意**：本项目里的"能力标签 / capability"是**打分聚合用的字符串**，跟 Anthropic AgentSkills（SkillsBench 论文里那种"给 agent 的操作手册包"）是**完全不同**的概念。

### 5. Docker 分层：场景级镜像

不是每题一个 Dockerfile，而是每个场景一个 domain image（social/medical/timeseries...）。
20 题共用 6-8 个 image，维护成本低。

### 6. 判分层保留自主控制

Harbor 的 verifier 是 pytest 全或无。我们通过 `verifier_wrapper.py` 桥接：
Harbor 跑完 pytest 后，wrapper 读 `groundtruth.json` 的权重表做二次加权聚合。
最终得分同时暴露给 Harbor（作为 reward）和自研 leaderboard。

### 7. Agent 层解耦

`agent_adapter.py` 把 159 原有的 `minimal_agent.py` 包装成通用 HTTP API Agent 接口。
一行命令跑多模型：`python -m framework.runner --agent http_api --model claude-sonnet-4.5 ...`。
天花板测评时可切换为 native CLI（Claude Code / Codex）。跟 Harbor CLI 的对接依赖前述的
转换器工作。

---

## 四、快速开始

**最短路径（10 秒能跑，不需要 Docker 也不需要 API key）**：

```bash
# 用 oracle 参考解跑一次判分器,验证仓库自洽
python3 -m framework.runner \
  --task tasks/education_academia/159_gaokao_reform \
  --agent oracle --check-oracle
# 期望输出: score >= 0.99  (CI 的 L3 硬门槛)
```

**提交前的本地二件套**（跟 CI 一致）：

```bash
# L1 + L2 结构 + 内容校验(<5s)
python3 -m framework.ci_validate --structure --content

# 框架自测
python3 -m pytest -q
```

**Docker 沙箱 + 真实模型评测**：

```bash
# 0. 一次性:装 Docker Desktop
#    brew install --cask docker

# 1. Build 场景镜像(一次性,~5-15 分钟)
docker build -f docker/base.Dockerfile   -t bench-base:v1   .
docker build -f docker/social.Dockerfile -t bench-social:v1 .

# 2. 跑一次真实模型评测(密钥用环境变量注入,不入库)
export ANTHROPIC_API_KEY=sk-ant-...
python3 -m framework.runner \
  --task tasks/education_academia/159_gaokao_reform \
  --model claude-sonnet-4.5 --seed 0 --use-docker

# 3. 未来上并发/云沙箱(v2 阶段接入 Harbor CLI)
#    需要先写 export_to_skillsbench.py 转换器把本仓库的 task 包翻成
#    native task.md + oracle/solve.sh + verifier/test.sh, 才能:
# harbor run -p ./tasks -a http_api_agent -m claude-sonnet-4.5 -n 8 --env daytona
```

**并发多 seed 聚合**：跑完 ≥5 个 seed 后，用 `framework/aggregate.py` 收敛成
稳定分 + bootstrap CI：

```bash
# 单模型单题聚合
python3 -m framework.aggregate task \
  --task tasks/education_academia/159_gaokao_reform --model claude-sonnet-4.5
# 两模型配对差异检验(paired-diff bootstrap, 判 CI 是否含 0)
python3 -m framework.aggregate compare \
  --tasks-root tasks --model-a claude-sonnet-4.5 --model-b gpt-5.5
```

**Docker 是可选开关**：
- **加 `--use-docker`**：agent 生成的 Python 代码在容器里跑，pandas/sklearn 版本严格锁定，跨机器可复现
- **不加**：走宿主机 subprocess，快，方便 debug
- LLM API 调用永远在宿主机，API key 不进容器

**镜像 build 一次就好**，改 Dockerfile 才需要重 build（加 `--force` 或 `--no-cache` 到 `docker build`）。

---

## 五、CI 三层门禁

`.github/workflows/ci.yml` 在每个 PR / push 上自动跑：

| 档次 | 时长 | 内容 |
|---|---|---|
| **L1 · structure** | ~1 分钟 | 目录/文件/TOML/JSON 有效性、`weights`↔`scoring`↔`values` 键集自洽、`registry/*.json` ↔ `tasks/` 双向一致、`difficulty_distribution` 与实际计数相符 |
| **L2 · content & docs** | ~3 分钟 | `scenario` / `difficulty` / `capability` 白名单、`veto` 引用完整性、`scoring.type` 是 `scorer.py` 支持的类型 + 全仓 `pytest -q` + `instruction.md` 泄漏禁词扫描 |
| **L3 · oracle gate** | ~30 分钟 | 逐题 `--check-oracle`（`combined_score >= 0.99`） |

L1 / L2 单机也能跑，就是 `framework/ci_validate.py` + `pytest -q`，跑一遍 <5 秒。L3 需要 numpy / pandas / scipy / sklearn / statsmodels 全套装好，本地首跑约几分钟。

三个词表（`SCENARIOS` / `DIFFICULTIES` / `CAPABILITIES`）+ 状态桶（`INFRA_STATUSES` / `MODEL_FAIL_STATUSES`）的**唯一定义**分别在 `framework/taxonomy.py` 和 `framework/statuses.py`；CI 就是从这两处 import 校验，改文档必须同时改这两处。

---

## 六、路线图

v1.0 只 ship 1 道 sample task（`159_gaokao_reform`，Extreme）作为 framework 的旗舰参考。

- **v1.x**：分批补齐 12 个 scenario 的 easy / hard / extreme 题目。目标 20 题左右首个稳定版
- **v2**：接入 Harbor CLI —— 写 `export_to_skillsbench.py` 把本仓库的 task 包翻译成 native `task.md` + `oracle/solve.sh` + `verifier/test.sh`，就能直接 `harbor run` 跑
- **hidden 集**：Extreme / 关键题补一份不同数据的 hidden 版（如"研究生入学考试分数分布"）

题目按目录 scenario 分类；新任务贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md) 与 [TASK_AUTHORING.md](TASK_AUTHORING.md)。
