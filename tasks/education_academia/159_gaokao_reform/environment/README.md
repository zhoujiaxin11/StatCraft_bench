# Environment: 159_gaokao_reform

## 数据文件

- `2000_2025_gaokao_score_distribution_11_11_2025.csv`

数据实际挂载点：运行时由 `framework/runner.py` 从原 159 任务目录（`159-Gaokao_Score_Distribution/environment/`）符号链接进来，或从 `data/education_academia/` 中央数据目录加载。

## 数据规模

- 约 53 万行
- 覆盖 31 个省级行政区
- 年份跨度 2000–2025

## 字段

详见 task 目录下的 `instruction.md`。

## 使用约定

- Agent 通过相对路径读取：`./environment/2000_2025_gaokao_score_distribution_11_11_2025.csv`
- 挂载为 **只读**，agent 无法修改
- 不得读取工作区之外的数据
