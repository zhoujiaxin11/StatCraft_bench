# Harbor-Compatible Data Analysis Bench Sample

以 `159-Gaokao_Score_Distribution` 为改造样例，演示 **数据分析 AgentBench** 的目录结构、Harbor 兼容任务格式、评估协议、以及 framework 层设计。

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

**沿用 159 原有的**：
- `outcome_grader_v2.py` 的加权部分分思路（升级为 framework/scorer.py）
- `minimal_agent.py` 的 HTTP API 跨模型能力（包装成 Harbor Agent Adapter）
- `groundtruth.json` 中的 tolerance 容差字段设计
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
│   ├── agent_adapter.py              # HTTP API 通用 Agent 接口
│   ├── aggregate.py                  # 多 seed 聚合(稳定分+bootstrap CI+配对检验)
│   └── verifier_wrapper.py           # pytest → 加权分数桥接
├── docker/                           # 分层镜像(场景级,不是每题级)
│   ├── base.Dockerfile
│   └── social.Dockerfile
├── tasks/
│   └── social_stats/
│       └── 159_gaokao_reform/        # 改造样例
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
└── configs/
    ├── models.yaml                   # 5 个模型的 API 配置
    └── evaluation-plans.yaml         # 分层评测计划
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

## 四、与 159 原版的对比

| 维度 | 159 原版 | Harbor Sample |
|---|---|---|
| 任务结构 | 半标准（有 `environment/tests/task.toml` 但缺 `solution/`） | Harbor 标准 5 目录 |
| Groundtruth 位置 | 同目录（易泄漏） | 独立 `groundtruth/public/` |
| Instruction 泄漏 | 方法学 + 评分标准全暴露 | 去泄漏，只描述"要什么"不指定"怎么做" |
| 判分器 | 每题一份 `outcome_grader_v2.py` | 全 bench 一份 `framework/scorer.py` |
| 能力归因 | 无（只有粗粒度 taxonomy） | 13 个能力标签 + 权重按能力分组 |
| 难度分级 | `complexity="hard"` 单档 | Easy / Hard / Extreme 三档 |
| Docker | 无（宿主机直接跑） | 分层场景 image |
| 版本管理 | 无 | `registry/v1.0.json` 版本清单 |
| Agent 层 | `minimal_agent.py` 强耦合 | Harbor Agent Adapter，可切换 |

---

## 五、快速开始

```bash
# 0. 一次性:装 Docker Desktop(可选,不加 --use-docker 就不需要)
#    brew install --cask docker

# 1. Build 场景镜像(一次性,~5-15 分钟)
docker build -f docker/base.Dockerfile   -t bench-base:v1   .
docker build -f docker/social.Dockerfile -t bench-social:v1 .

# 2. 校验 oracle 能过判分器(CI 硬规则)
#    --use-docker 走容器沙箱; 不加就走宿主机 subprocess
python -m framework.runner \
  --task tasks/social_stats/159_gaokao_reform \
  --agent oracle --check-oracle --use-docker

# 3. 跑一次真实模型评测
export ANTHROPIC_API_KEY=sk-ant-...
python -m framework.runner \
  --task tasks/social_stats/159_gaokao_reform \
  --model claude-sonnet-4.5 --seed 0 --use-docker

# 4. 未来上并发/云沙箱(v2 阶段接入 Harbor CLI)
#    需要先写 export_to_skillsbench.py 转换器把本仓库的 task 包翻成
#    native task.md + oracle/solve.sh + verifier/test.sh, 才能:
# harbor run -p ./tasks -a http_api_agent -m claude-sonnet-4.5 -n 8 --env daytona
```

**并发多 seed 聚合**：跑完 ≥5 个 seed 后，用 `framework/aggregate.py` 收敛成
稳定分 + bootstrap CI：

```bash
# 单模型单题聚合
python -m framework.aggregate task \
  --task tasks/social_stats/159_gaokao_reform --model claude-sonnet-4.5
# 两模型配对差异检验(paired-diff bootstrap, 判 CI 是否含 0)
python -m framework.aggregate compare \
  --tasks-root tasks --model-a claude-sonnet-4.5 --model-b gpt-5.5
```

**Docker 是可选开关**：
- **加 `--use-docker`**：agent 生成的 Python 代码在容器里跑，pandas/sklearn 版本严格锁定，跨机器可复现
- **不加**：走宿主机 subprocess，快，方便 debug
- LLM API 调用永远在宿主机，API key 不进容器

**镜像 build 一次就好**，改 Dockerfile 才需要重 build（加 `--force` 或 `--no-cache` 到 `docker build`）。

---

## 六、v1 阶段（20 题）落地建议

- **8 Easy 题**：单一能力，SOTA 应能 60%+，来自 200 题池中 taxonomy `complexity=easy/medium` 的题
- **9 Hard 题**：2-3 种能力，SOTA 应在 25-45%，主要负载
- **3 Extreme 题**：5+ 种能力堆叠 + 高阶方法，SOTA 应在 5-20%，给未来模型留天花板
  - **159 是 Extreme 题之一**（本样例）
  - 权重可以更高（Easy 平均权 5，Hard 权 8，Extreme 权 12），防止未来分数饱和

---

## 七、下一步

- 用真实数据在 159 样例上跑通 oracle → scorer 闭环
- 挑 1-2 道 Easy 题按同套模板改造，验证结构通用性
- 补齐 hidden 版：把 159 结构复制一份用不同数据（如"研究生入学考试分数分布"）作为 hidden test
