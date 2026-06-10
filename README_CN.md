# MiniMind FBBP Query Compiler Lab

[English](./README.md) | **中文**

这是一个可在单张 RTX 4090 上运行的小模型微调实验仓库。当前主线任务是把自然语言的 FBBP 候选检索请求编译为受约束、可验证、可执行的查询计划，而不是训练通用问答模型。

## 项目主线

用户请求 -> 模型生成查询草案 -> DSL 校验/标准化 -> 数据库执行 -> 固定协议评估

## 快速导航

| 目标 | 入口 |
|---|---|
| 了解训练流程 | [TRAINING_README.md](./TRAINING_README.md) |
| 查看查询编译器 | [query_compiler/](./query_compiler/) |
| 查看配置版本 | [configs/](./configs/) |
| 查看评估脚本 | [scripts/run_query_compiler_eval.py](./scripts/run_query_compiler_eval.py) |
| 查看最终结果 | [FINAL_RESULT_SUMMARY.md](./FINAL_RESULT_SUMMARY.md) |
| 理解上游边界 | [UPSTREAM.md](./UPSTREAM.md) |

## 当前结果边界

正式结果来自 23 在保留的 15 true holdout 协议上。该结果说明当前固定任务和约束下达到稳定首轮语义落地，不代表任意未来自然语言表达都已解决。数据库实体层是真实的，但自然语言监督仍以程序化构造为主。

## 复现要求

- Python 与 PyTorch 环境
- MiniMind2 104M/768dim 对应上游模型资源
- 单张 RTX 4090 或同等级显存环境
- 按配置文件固定随机种子、数据划分和评估协议

详细界面说明见 [INTERFACE_GUIDE_CN.md](./INTERFACE_GUIDE_CN.md)。
