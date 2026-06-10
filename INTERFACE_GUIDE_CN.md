# MiniMind FBTP 实验仓 界面说明

[English](./INTERFACE_GUIDE_EN.md) | [中文](./INTERFACE_GUIDE_CN.md)

## 这个仓库是做什么的

带 DSL、validator 和 holdout gate 的小模型 Query Compiler 实验。

## 谁应该先看这个说明

关注小模型适配能力的模型训练评审者和面试官；本仓不是从零重写 MiniMind。

## 仓库阅读顺序

- 先读 README.md、TRAINING_README.md、UPSTREAM.md 和 FINAL_RESULT_SUMMARY.md。
- query_compiler/ 包含 DSL、validator、repair、scoring 和 executor 逻辑。
- reports/algorithm_resume/ 保存简历可用证据。
- 仓库没有上传模型权重或原始私有训练数据。

## 上传边界

这个仓库是已经整理过的公开上传版本。上传前已经排除了本机路径、运行缓存、日志、原始私有数据、模型权重和临时工作文件。

## 中英文切换

本文件顶部提供 English / 中文链接，可在英文界面说明和中文界面说明之间切换。