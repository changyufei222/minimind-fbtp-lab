# FBBP 查询编译器算法报告

## 结论摘要

这是一条面向算法实习/论文写作的 **小模型查询编译器** 线，而不是普通 prompt demo。任务是把自然语言的 FBBP 候选检索请求，编译成可执行的结构化查询计划。

核心结论是：

> 在固定协议下，`baseline_v23` 已经能生成合法结构化计划，但语义执行还不够稳定；`lora_v23` 在 reserved `v15 true holdout` 上把 `slot_accuracy` 从 `0.8141` 提升到 `1.0`，把 `result_overlap_at_k` 从 `0.28` 提升到 `1.0`，并达到 `first_pass_perfect_rate = 1.0`、`used_repair_rate = 0.0`、`used_projection_rate = 0.0`。

## 任务定义

输入：

- 自然语言候选搜索请求

输出：

- 可执行的结构化 query plan

这使它不只是生成 JSON，而是一个包含以下成分的算法任务：

| 维度 | 含义 |
|---|---|
| 结构化输出学习 | 模型需要生成符合协议的 plan |
| 语义槽位对齐 | 请求中的语义要映射到正确字段 |
| 可执行计划正确性 | 输出不仅要合法，还要能执行 |
| 下游结果对齐 | 计划执行后要命中正确候选集合 |

## 协议与数据划分

这个项目已经不是随便拼提示词，而是有固定协议的。

| 项目 | 说明 |
|---|---|
| train / dev / test_seen / test_hard | 固定划分 |
| reserved true holdout | 独立保留的最终门控集 |
| no-hints | 无提示版本 |
| farther no-hints | 更远离模板表述的鲁棒性版本 |
| validator-backed execution | 用执行结果验证计划正确性 |

可用于论文的话术：

> 我们将查询编译任务设计为带执行验证的结构化输出学习问题，并在固定 train/dev/test 之外保留 true holdout 作为最终晋升门槛。

## 方法

方法主线可以概括为：

> 小模型 + LoRA 微调 + 受控协议 + 执行验证 + repair/projection 追踪。

训练与推理中最关键的设计包括：

- rule baseline
- base model 历史轮次
- LoRA-tuned 轮次
- repair / projection 使用情况记录
- 不是只看 JSON 合法性，而是看执行后的候选集是否正确

## 结果

最强对比来自同一 reserved `v15 true holdout`：

| 模型 | plan_valid_rate | slot_accuracy | result_overlap_at_k | used_repair_rate | used_projection_rate |
|---|---:|---:|---:|---:|---:|
| `baseline_v23` | 1.0 | 0.8141 | 0.28 | - | - |
| `lora_v23` | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |

补充解释：

- `baseline_v23` 已经能稳定产出合法计划，但还会在语义槽位与候选集合上犯错。
- `lora_v23` 的真正提升不是“更像 JSON”，而是“语义执行正确性”。
- 最终结果不依赖 repair 或 projection，说明模型本身已经学会了编译规则。

## 可靠性说明

这份结果为什么不像“糊出来的”：

1. 使用的是 reserved holdout，不是训练集或普通 dev。
2. 有 no-hints / farther no-hints 变体，不是只记模板。
3. 评价指标包含 `slot_accuracy`、`execution_success_rate`、`result_overlap_at_k`，不是只看 JSON 是否能 parse。
4. 最终晋升结果显示 `used_repair_rate = 0.0`、`used_projection_rate = 0.0`，说明不是靠后处理补出来的。

## 论文可直接使用的方法段

我们将 FBBP 候选检索请求建模为结构化查询编译任务，输入为自然语言请求，输出为可执行 query plan。为避免仅凭格式合法性判断模型能力，我们采用 validator-backed execution protocol，对计划的字段、过滤条件、排序和执行结果进行联合验证，并在固定 train/dev/test 之外保留 reserved true holdout 作为最终晋升门槛。模型采用 MiniMind2 104M 作为基础模型，通过 LoRA 微调学习查询编译行为，并在 no-hints 与 farther no-hints 评测族上检验鲁棒性。

## 论文可直接使用的结果段

在 reserved `v15 true holdout` 上，`baseline_v23` 已达到 `plan_valid_rate = 1.0`，但其 `slot_accuracy` 仅为 `0.8141`，`result_overlap_at_k` 仅为 `0.28`。相比之下，`lora_v23` 将 `slot_accuracy` 提升至 `1.0`，将 `result_overlap_at_k` 提升至 `1.0`，并且实现了 `first_pass_perfect_rate = 1.0`，`used_repair_rate = 0.0`，`used_projection_rate = 0.0`，表明模型在该协议下可以一次性生成可执行且语义正确的结构化查询计划。

## 简历表述

- 训练了一个面向 FBBP 候选检索请求的 MiniMind2 104M 查询编译器，将自然语言请求映射为可执行结构化 query plan。
- 设计固定 train/dev/test/true-holdout 协议，并用 validator-backed execution 评估计划的语义正确性与候选集对齐效果。
- 在 reserved no-hints true holdout 上，将 `slot_accuracy` 从 `0.8141` 提升到 `1.0`，将 `result_overlap_at_k` 从 `0.28` 提升到 `1.0`，同时消除了 repair / projection 依赖。

## 面试解释

短版：

> 我做的是一个小模型 query compiler，而不是普通的提示词模板。它把自然语言候选检索请求编译成结构化计划，并且不是只看 JSON 合法性，而是看执行后的候选集合是否对。最终在 reserved holdout 上，LoRA 把 slot 对齐和结果重叠都提到了 1.0，而且不再依赖 repair / projection。

## 证据文件

| 文件 | 作用 |
|---|---|
| `reports/algorithm_resume/query_compiler_resume_summary.md` | 项目总摘要 |
| `reports/algorithm_resume/query_compiler_protocol_reliability.md` | 协议与可靠性说明 |
| `reports/algorithm_resume/query_compiler_baseline_matrix.md` | baseline / LoRA 对比矩阵 |
| `reports/algorithm_resume/query_compiler_error_analysis.md` | 误差分析 |
| `reports/algorithm_resume/query_compiler_training_card.md` | 训练卡片 |

