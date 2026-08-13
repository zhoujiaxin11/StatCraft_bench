# 任务贡献规范（Task Authoring Spec）

面向数据分析 AgentBench 的**新任务贡献者**。所有新加入的题目**必须**遵循本规范；不符的 PR 会被 CI 拒绝。

---

## 一、总体原则（先记住这 5 条）

1. **零泄漏**：`instruction.md` 里不含真值、评分标准、控制组名单、公式实现
2. **单一 factoid**：主输出是**一份** `outcome.json`，字段类型由 `schema.json` 声明
3. **oracle 必过**：`solution/solve.py` 必须能拿到 `combined_score ≥ 0.99`（CI 硬门槛）
4. **groundtruth 隔离**：真值放在 `groundtruth/public/<task_id>.json`，**不进** `tasks/` 目录
5. **能力标签必填**：从受控词表挑，用于 per-capability 归因（**注意**：本文档里的"能力标签 / capability"跟 Anthropic AgentSkills 是完全不同的概念，仅是打分聚合用的字符串标签）

---

## 二、目录结构（新任务的 5 个必要文件 + 1 个真值文件）

```
tasks/<scenario>/<task_id>/
├── task.toml               # ✅ 必要 - 元数据 + 能力标签 + 镜像声明
├── instruction.md          # ✅ 必要 - 干净任务描述(无泄漏)
├── schema.json             # ✅ 必要 - 答案字段类型
├── environment/            # ✅ 必要 - 数据文件目录
│   └── ...
├── solution/
│   └── solve.py            # ✅ 必要 - Oracle 参考解
├── tests/
│   └── test_outputs.py     # ⚠️ 可选 - 艺术品/结构断言(pytest)
└── Dockerfile              # ⚠️ 可选 - 仅在需要特殊依赖时

# 真值(单独存放)
groundtruth/public/<task_id>.json     # ✅ 必要
groundtruth/hidden/<task_id>.json     # ⚠️ 可选(Extreme/关键题建议做隐藏版)
```

---

## 三、命名规范

**task_id**：`<数字>_<英文短名>`（用下划线，不用中文/空格）
- `159_gaokao_reform`
- `042_km_survival_bmc_dataset`
- `007_customer_churn_lgbm`

**scenario**（一级目录）：从下面 12 个中选一个（Chinese 说明仅供参考，实际入库的是英文 slug）
- `medical_health` — 医疗·生命健康
- `finance_economics` — 金融·经济
- `retail_ecommerce` — 零售·电商·消费营销
- `industrial_energy` — 工业·能源·制造
- `transport_logistics` — 交通·物流
- `environment_climate` — 环境·气候·生态·农业监测
- `geo_hazards` — 自然灾害·极值事件·地球物理
- `education_academia` — 教育·学术
- `society_policy` — 社会·公共政策·人口
- `tech_internet` — 科技·互联网·媒体信息
- `sports_entertainment` — 体育·娱乐·文旅
- `bio_chem_materials` — 生物·化学·材料

白名单的**唯一定义**在 [framework/taxonomy.py](framework/taxonomy.py) 的 `SCENARIOS`；改这里必须同步改那里，反之亦然。

**数据文件**：保留原始文件名。避免超长中文（部分容器/CI 环境编码差）。

---

## 四、`task.toml` 规范

```toml
[task]
id = "159_gaokao_reform"        # 与目录名一致
title = "全国高考分数分布——改革影响、位次预测与因果推断"

[taxonomy]
scenario = "education_academia"  # 上面 12 个之一
subscenario = "education_reform_analysis"  # 自由文本,便于分组
difficulty = "extreme"           # easy / hard / extreme(见第八节)
version = "v1.0"                 # 加入时的 bench 版本

# 从能力标签受控词表挑,3-15 个(见第六节)
capabilities = [
  "data_cleaning",
  "weighted_statistics",
  "did_estimation",
  ...
]

[environment]
image = "bench-social:v1"        # 从场景 image 中选(见第七节)
# dockerfile = "./Dockerfile"    # 仅在有专属 Dockerfile 时启用

[resources]
timeout = 900                    # 单次 agent 运行秒数上限
max_turns = 50                   # ReAct 循环最大轮次
memory_gb = 8                    # 容器内存上限
cpus = 4                         # 容器 CPU 上限

[required_outputs]
files = ["outcome.json"]         # 必须产出的文件(最少 outcome.json)
# 例:additional artifacts
# files = ["outcome.json", "output/predictions.csv", "output/plot.png"]

[scoring]
factoid_alpha = 0.9              # 主判分权重(0.9=90% factoid + 10% pytest)
pass_threshold = 0.3             # Extreme=0.3, Hard=0.5, Easy=0.6
```

**字段要求**：
- ✅ `[task]`、`[taxonomy]`、`[environment]`、`[resources]`、`[required_outputs]`、`[scoring]` 全部必填
- ✅ `capabilities` 至少 1 个，最多 15 个
- ❌ 别加自定义 section（等 spec 更新后再扩）

---

## 五、`instruction.md` 规范

### 必写内容

1. **任务背景**：1-2 段，说明业务场景
2. **数据说明**：字段列表、规模、已知的脏数据类型
3. **需要做什么**：分部分描述（Part 1/2/3 …），每部分只描述"要什么"
4. **输出要求**：告诉 agent 保存到 `outcome.json` 和 `output/`，schema 见 `schema.json`
5. **注意事项**：禁止访问 `tests/`、`groundtruth*`；只用相对路径

### 严禁写入（防泄漏红线）

| 类型 | 反例 | 正确做法 |
|---|---|---|
| ❌ 具体真值 | "weather_sensitivity = 2.3883" | 只说"报告天气敏感度" |
| ❌ 完整公式 | "p = 1 − erf(\|t\|/√2)" | 只说"检验平行趋势" |
| ❌ 控制组名单 | "控制组:内蒙古、北京、吉林..." | "自行判断控制组，说明选择依据" |
| ❌ 评分权重 | "Part3 占 42 分" | "5 部分独立评分，权重不披露" |
| ❌ 实现细节 | "特征命名用 hist_ 前缀" | 只说"避免目标泄漏" |
| ❌ 硬阈值 | "R² > 0.95" | "报告测试集 R²、RMSE、MAE" |

### 长度指引

- Easy 题：< 30 行
- Hard 题：< 80 行
- Extreme 题：< 100 行（159 现在已经压到 ~70 行）

**关键测试**：让另一个人读 instruction.md，如果他能**猜到真值或方法**，就是泄漏了。

---

## 六、`schema.json` 规范

只声明**字段名 + 类型**，绝对不含具体值。

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "<task_id> answer schema",
  "type": "object",
  "properties": {
    "part1_overview": {
      "type": "object",
      "properties": {
        "total_records": {"type": "integer"},
        "province_count": {"type": "integer"},
        "year_min": {"type": "integer"},
        "year_max": {"type": "integer"}
      },
      "required": ["total_records", "province_count"]
    },
    ...
  }
}
```

**规则**：
- 只有 `type` / `enum`（有限选项时）/ `description`（可选，简短说明字段含义）
- 不含 `default`、不含具体 min/max、不含示例值
- 嵌套用 `properties`，跟 outcome.json 结构一致

---

## 七、能力标签受控词表

> **命名说明**：本文档里的"能力标签 (capability)"是打分聚合用的字符串标签，跟 Anthropic AgentSkills（SkillsBench 论文里那种"给 agent 的操作手册包"）**没有任何关系**。为避免混淆，字段名统一用 `capabilities` / `capability_map` 而非 `skills`。

新任务的 `[taxonomy].capabilities` 必须从以下 42 个标签中挑。词表的**唯一定义**在 [framework/taxonomy.py](framework/taxonomy.py) 的 `CAPABILITIES`；改这里必须同步改那里，反之亦然。**要加新标签需要提 spec 变更 PR**。

**数据处理**：
- `data_cleaning` · `data_format_handling` · `data_manipulation` · `data_deduplication` · `missing_value_handling`

**描述性统计**：
- `descriptive_statistics` · `weighted_statistics` · `grouped_comparison` · `cross_tabulation`

**推断统计**：
- `hypothesis_testing` · `parametric_test` · `nonparametric_test` · `multiple_comparison` · `power_analysis`

**回归与建模**：
- `regression_modeling` · `time_series_features` · `feature_engineering` · `model_selection` · `hyperparameter_tuning` · `cross_validation`

**因果推断**：
- `did_estimation` · `synthetic_control` · `placebo_test` · `parallel_trend_test` · `propensity_matching` · `iv_estimation`

**高阶统计**：
- `evt_tail_index` · `panel_ols_inference` · `bayesian_inference` · `survival_analysis` · `mixed_effects_model`

**距离与分布**：
- `wasserstein_distance` · `kl_divergence` · `distribution_test`

**不确定性**：
- `conformal_prediction` · `bootstrap_ci` · `bayesian_posterior`

**工程与安全**：
- `leakage_prevention` · `code_correctness` · `numerical_stability` · `reproducibility` · `output_format_compliance`

---

## 八、难度分级标准

| 档次 | 标准 | 预期 SOTA 分数 | 涉及能力数 | 建议题占比 |
|---|---|---|---|---|
| **easy** | 单一场景、公开教科书方法、5-15 分钟能解 | 60-80% | 1-2 种能力 | 40% |
| **hard** | 2-3 种能力组合、需要判断方法选择 | 25-45% | 3-5 种能力 | 45% |
| **extreme** | 5+ 种能力堆叠、涉及高阶统计/因果、模糊指定 | **5-20%** | 5+ 种能力 | 15% |

**判定原则**：如果一个熟练的数据分析师 15 分钟能做完 → easy；一个下午能做完 → hard；需要一整天思考 + 试错 → extreme。

**校准方法**：新加的题跑 3-5 个头部模型（gpt-5.5 / claude-sonnet / gemini），观察分数区间：
- 全都 > 80% → 太简单，改为 easy 或补充难度
- 全都 < 5% → 可能定义不清或超出模型能力范围，需要调整
- 分数在 20-60% 分散 → 好的 hard/extreme 题

---

## 九、`groundtruth/public/<task_id>.json` 规范

```json
{
  "task_id": "<task_id>",
  "difficulty": "hard",
  "notes": "自由描述来源、算法、注意事项",

  "capability_map": {
    "part1.total_records": ["data_cleaning"],
    "part2.weighted_mean_old": ["weighted_statistics"],
    ...
  },

  "values": {
    "part1": {"total_records": 535326, ...},
    "part2": {...},
    ...
  },

  "weights": {
    "part1.total_records": 1,
    "part2.weighted_mean_old": 2,
    ...
  },

  "scoring": {
    "part1.total_records": {"type": "exact_number", "gt": 535326},
    "part2.weighted_mean_old": {
      "type": "graded", "gt": 474.78,
      "levels": [
        {"tol": 0.005, "score": 1.0},
        {"tol": 0.02, "score": 0.7},
        {"tol": 0.05, "score": 0.3}
      ]
    },
    ...
  },

  "pytest_weight_map": {
    "test_outcome_json_exists": 1,
    ...
  },

  "factoid_alpha": 0.9
}
```

### 支持的 scoring rule types

| type | 用途 | 必填参数 |
|---|---|---|
| `exact_number` | 整数/浮点精确匹配 | `gt` |
| `exact_string` | 字符串精确匹配（归一化后）| `gt` |
| `graded` | 数值分档给分（对称容差，越接近 gt 越好） | `gt` + `levels: [{tol/abs, score}]` |
| `min_threshold` | 越高越好的指标（R²、覆盖率、准确率） | `levels: [{min, score}]`（top-down 匹配） |
| `max_threshold` | 越低越好的指标（RMSE、MAE、错误率） | `levels: [{max, score}]`（top-down 匹配） |
| `bool` | 布尔字段（`true`/`false`/`1`/`0`/`"yes"`/`"是"` 等） | `gt: true` 或 `false` |
| `enum` | 枚举匹配 | `allowed: [...]` |
| `accept_set` | 多个可接受答案（如省份选择） | `accept: [...]` |
| `list_ordered` | 有序列表匹配 | `gt: [...]` |
| `list_set` | 无序集合匹配 | `gt: [...]` |
| `object_keys_match` | 字典所有键值匹配 | `gt: {...}` |
| `no_leakage_keywords` | 检查禁词 | `forbidden: [...]` |
| `presence` | 字段非空即得分 | 无 |

**关键选型原则**：R²/覆盖率/准确率这类"越高越好"的指标**必须**用 `min_threshold`，不要用 `graded`——`graded` 是双边对称容差，会惩罚超过 gt 的更好结果。RMSE/MAE 同理用 `max_threshold`。

**要加新 rule type**：需要 PR 修改 `framework/scorer.py::_score_field()` + 更新本表。

### 可选：`veto` 一票否决块

groundtruth JSON 可以加一个可选的 `veto` 块，用于"某关键字段没做对，整题算 0"的场景（例如做了导致目标泄漏的特征、因果效应方向搞反）：

```json
"veto": {
  "essential":     ["part2.reform_direction"],
  "unacceptable":  ["part3.leaked_feature_used"]
}
```

- `essential`：字段得分必须 > 0，否则整题清零。
- `unacceptable`：字段得分必须 = 0（表示禁忌行为**没**发生），否则整题清零。

规则：veto 字段必须已在 `weights` 或 `scoring` 中声明；`aggregate.py::validate_veto()` 会在加载时校验，拼错字段名会直接报错而不是静默清零。

### 权重设计原则

- 单字段权重区间 **1-6**
- Easy 字段平均权重 = 1-2
- Hard 字段平均权重 = 2-4
- Extreme 关键判分点 = 4-6

**总权重 = sum(weights)**。对 Extreme 题应在 60-120 范围。

---

## 十、`solution/solve.py` 规范

- 用 `./environment/xxx` 相对路径读数据（与 agent 完全一致的路径规则）
- 输出 `./outcome.json` 到工作目录根
- 其他产物按 `[required_outputs].files` 声明的路径保存
- **不允许**读取 `groundtruth*`、`tests/`、任何评分文件
- **不允许**硬编码真值（除非必要，否则应真实计算）—— 硬编码的部分要在文件顶部注释说明

### CI 硬门槛

```bash
python -m framework.runner \
  --task tasks/<scenario>/<task_id> \
  --agent oracle --check-oracle
# 必须输出:score >= 0.99
```

Oracle 拿不到 0.99 的 PR 不能合并。**26.7% 的历史通过率就来自这条**（SkillsBench 数据）。

---

## 十一、可选的 `tests/test_outputs.py`

只用于**主判分覆盖不了的检查**：
- 交付文件是否存在（`outcome.json` / `predictions.csv` / `.png`）
- CSV 列名/行数正确性
- 反泄漏关键词扫描（feature_names 里不含"人数"这种）
- 结构性完整（如 panel OLS 至少 3 个系数）

**不要**在 pytest 里判分主要 factoid 结果，那是 `groundtruth.json` 的活。

pytest 权重记在 `groundtruth.pytest_weight_map`，最终得分 `factoid_alpha * factoid + (1-alpha) * pytest`。

---

## 十二、可选的单题 Dockerfile

**大多数任务不需要**——用 `[environment].image` 指向已有场景镜像即可（`bench-social:v1` / `bench-medical:v1` / …）。

**需要单题 Dockerfile 的场景**：
- 特殊依赖冲突（比如你需要 `pandas < 2.0`）
- 异构语言（R / Julia / Stata）
- 特殊系统库

```dockerfile
FROM bench-social:v1
RUN R -e "install.packages('lme4', repos='https://cloud.r-project.org')"
RUN pip install rpy2==3.5.14
```

然后 `task.toml`：
```toml
[environment]
image = "bench-<task_id>:v1"
dockerfile = "./Dockerfile"
```

---

## 十三、CI 校验清单（合并前必过）

新任务 PR 会自动跑：

1. ✅ 目录结构完整（5 个必要文件都在）
2. ✅ `task.toml` schema 合法（所有必填字段 present、capability 都在受控词表内）
3. ✅ `instruction.md` 无泄漏关键词扫描（禁词表：`groundtruth`、`控制组:`、`权重 =`、"必须"数值等）
4. ✅ `schema.json` 是合法 JSON Schema
5. ✅ `groundtruth/public/<task_id>.json` 存在且结构合法
6. ✅ `weights` 里所有 field key 都能在 `values` 或 `scoring` 里找到
7. ✅ Docker 镜像存在（`task.toml [environment].image`）
8. ✅ **oracle 必过判分器**（`combined_score ≥ 0.99`）
9. ✅ pytest 全部通过
10. ✅ 至少 1 个已跑过的 baseline 模型分数（附在 PR 描述里）
11. ✅ 若 `groundtruth` 声明了 `"veto"` 块，则 `framework.aggregate.validate_veto` 必须通过（essential/unacceptable 引用的 field key 都能在 weights/scoring 里找到），否则 typo 会静默把每个 seed 归零
12. ✅ 单跑 score.json 必须带非空的 `veto_status` 字段（由 verifier_wrapper 盖章，暴露 configured/valid/would_zero_task）。注意：**单跑 combined_score 不应用一票否决**——零分只在聚合层 aggregate.py 触发；CI 第8条只验单跑 ≥0.99，故 essential/unacceptable 的实际生效靠本条+聚合层共同保证

失败任一条 → PR block。

---

## 十四、任务生命周期

`task.toml` 加一个 `[lifecycle]` section（可选）：

```toml
[lifecycle]
status = "active"        # active / deprecated / retired
added_in = "v1.0"
deprecated_in = null     # "v1.5" 时表示 v1.5 开始不再进 leaderboard
retired_in = null        # "v2.0" 时表示 v2.0 起彻底移除
reason = null            # 若 deprecated/retired,说明原因
```

- **active**：正常参与所有评测计划
- **deprecated**：仍在仓库,不再进 leaderboard（如题目被发现有 bug 或 SOTA 都 100%）
- **retired**：仅历史保留，不再运行

---

## 十五、Registry 版本管理

每次 bench 版本升级（v1.0 → v1.5 → v2.0），在 `registry/v<version>.json` 里声明该版本包含的题：

```json
{
  "dataset_id": "data-analysis-bench",
  "version": "1.5",
  "release_date": "2026-09-01",
  "tasks": [
    {"task_id": "159_gaokao_reform", "status": "active"},
    {"task_id": "042_km_survival",   "status": "active"},
    ...
  ],
  "changelog": {
    "1.5": "新增 20 道 hard 题;将 007_bike_sharing 标记 deprecated(SOTA 全部 >90%)"
  }
}
```

**关键作用**：让"跑 bench v1.0"和"跑 bench v1.5"是**两次可复现的评测**。

---

## 十六、快速模板

想加新题？先复制这个骨架，改字段：

```
tasks/<scenario>/<NNN_short_name>/
├── task.toml              # 复制 159_gaokao_reform/task.toml,改 id/title/capabilities/difficulty
├── instruction.md         # 参照第五节写,只描述任务不含真值
├── schema.json            # 声明输出字段类型
├── environment/
│   └── <data.csv>         # 放数据
└── solution/
    └── solve.py           # 参考解,必须能通过判分器

groundtruth/public/<NNN_short_name>.json  # 真值 + 权重 + scoring 规则
```

然后本地验证：
```bash
python -m framework.runner --task tasks/<scenario>/<NNN_short_name> \
  --agent oracle --check-oracle
# 必须输出:score >= 0.99
```

通过就可以 PR。

---

## 十七、常见错误 & 修复

| 症状 | 原因 | 修复 |
|---|---|---|
| Oracle 拿 0.82 而不是 0.99 | 某字段的 `scoring` type 没实现 | 检查 scorer.py 支持的 rule types |
| Agent 都拿 0 分 | outcome.json 没被产出 | 看 trajectory.jsonl,检查 [FINAL_ANSWER] 与代码执行顺序 |
| 所有模型分数一样 | instruction.md 泄漏方法 | 用第五节红线表逐条核对 |
| 分数差异集中在 20% 附近 | 分档 tolerance 太宽 | 收紧 `levels` 里的 tol |
| pytest 权重加起来大于 factoid | factoid_alpha 太低 | 调回 0.9,确保 factoid 主导 |

---

## 十八、Spec 变更流程

想改本规范（比如加 capability 标签、加 rule type、调难度定义）？

1. 开 issue 描述提议
2. 提 PR 修改 `TASK_AUTHORING.md` + 相关代码
3. Spec 版本号 bump（如 v1.0 → v1.1）
4. 已有任务无需回填，新任务从下个版本起遵守新规

---

**Spec 版本：v1.0**
**最后更新：2026-07-31**
**问题反馈：本仓库 issues**


## Trial 状态字段语义

每个 trial 的 `score.json` 同时携带两个看起来相似的终止信息，但它们的**权威性不同**，报告与聚合必须只认一个：

- **`status`（权威判分字段）** —— 回答"这一 trial 是否拿到了可计分的有效分数"。`framework/aggregate.py`
  只用 `status` 把 trial 归入三个桶：
  - `ok`：已测、产出可评分答案 → 计入能力均值（按实际分）。
  - `model_fail`（如 `max_turns / no_output / bad_json / no_progress / stuck_loop`）：测了但模型没给出有效答案 → 计 0 分但仍计入均值。
  - `infra`（如 `api_error / executor_crash / scorer_error / docker_unavailable / image_missing / bad_io / wall_timeout`）：**没有真正测量到模型** → 从均值中剔除。

- **`_meta.terminated_by`（诊断字段）** —— 记录 agent *为什么* 停下
  （`final_answer / stuck_loop / wall_timeout / max_turns_exhausted …`），仅供事后审计与排障，
  不直接决定打分。它可能与 `status` 取值相同，也可能不同；当二者冲突时一律以 `status` 为准。

### 单一真源与登记义务

三组状态字符串的**唯一定义**在 [framework/statuses.py](framework/statuses.py)
（`INFRA_STATUSES / MODEL_FAIL_STATUSES / ALL_KNOWN_STATUSES`）。runner 与 aggregate 都从该模块导入；
**任何新增的状态字符串都必须先在此处登记**。未登记的 status 在 `_bucket()` 里会触发告警并按
infra（保守丢弃）处理——绝不会静默当成 ok 计进均值。配套单测见
[tests/test_status_universe.py](tests/test_status_universe.py)，
它会断言关键 infra/model_fail 状态仍在册，防止后续重构把它们漏掉而污染排行榜数字。

