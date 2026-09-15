# 如何复现 GSM8K GRPO 实验

类型：实验专属复现指南。对应 [报告](report.md)。

## 先确定需要哪一种复现

| 目标 | 输入与入口 | 能得出的结论 |
|---|---|---|
| 查验保存材料 | [机器胶囊](../../../../repro/experiments/gsm8k-qwen3-0.6b-grpo/experiment.json)，仓库根运行 `python3 repro/verify.py` | 固定文件、配置、命令与来源是否匹配 |
| 复算原有结果 | 保存的两份完整 predictions、历史统计／验证实现；见 [repro/README](../../../../repro/README.md) | 原始记录能否重新得到相同分数与配对统计 |
| 重新训练与评价 | 原始模型／数据、历史环境、各阶段 checkout、[原始 argv](../../../../repro/experiments/gsm8k-qwen3-0.6b-grpo/commands.json) | 一次新的运行；需要自己的完整证据，不能预先视为成功 |

训练使用原始 0.6B 模型，评价后模型使用它加 step 64 adapter。重建时按胶囊模型／数据锁核对真实文件，在新目录运行；旧训练结果和数据不得覆盖。数据与权重不包含在 Git 中。

恢复代码应使用对应提交的 Git checkout。代码快照供离线查阅／字节核验，不能假装自身就是原 Git 根目录。历史命令中的 `/workspace/experiment` 是当时的实际路径，迁移时必须明确新的资产绑定和输出位置，保存新命令。

正式顺序为：准备并验证资产 → 训练 64 步并保存各 dev 点 → 选定 checkpoint → 原始模型完整 audit → 所选模型完整 audit → 统计与实际权重验收。数学基线 audit 的实际日期在训练之后，但使用未训练的原始权重。

运行环境和归档边界见 [公共操作说明](../../../operations/historical-reproduction.md)。本次审阅只复核保存材料，没有重新运行上述 GPU 链路。
