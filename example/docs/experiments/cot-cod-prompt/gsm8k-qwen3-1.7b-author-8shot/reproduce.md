# 如何复现作者 CoT／CoD 提示对照

类型：实验专属复现指南。对应 [报告](report.md)。

| 目标 | 材料 | 可验证范围 |
|---|---|---|
| 查验胶囊 | [experiment.json](../../../../repro/experiments/gsm8k-qwen3-1.7b-author-cot-cod/experiment.json) 与 `python3 repro/verify.py` | 配置、源代码、原命令及文件身份 |
| 公开数值复算 | 两份 `author_metrics.jsonl`、原配对结果；见 [repro 入口](../../../../repro/README.md) | 由已保存分数／token 重算计数、成本、同题翻转与统计；不重新判断原文答案 |
| 私有全文重评分 | 原始 `author_predictions.jsonl`、作者固定输入、原 tokenizer、canonical 数据、原比较器 | 重解码、重建输入与重新评分；必须取得实际原文材料 |
| 重新生成 | 原始1.7B权重、模型／数据锁、作者来源与历史vLLM用户态 | 一次新的1319×2生成；其完整性和结果需单独核验 |

作者代码源为 `61ff6a476c3cacec1fe87c846c9559a010f499dd`。恢复该 Git checkout 后，用来源锁取得固定作者文件；实际命令的 `--source-dir /opt/cod-author-a7dbf5d` 是历史 guest 路径，不能在宿主中原样假设存在。

同样，原命令中的模型、数据与输出绝对路径属于当时的机器。迁移需显式绑定资源；新输出目录必须与旧结果区分。两模式必须保持相同模型、模板、采样、预算、题序与后端，不能只复制配置文件名。

原始完整响应曾回显作者示例，因此仅在私有归档与服务器保存；Git 中的 `author_metrics.jsonl` 是公开数值投影，不能拿它冒充完整原始回答。没有外部输入或原始响应时，报告相应能力缺失，不补造内容。[详细边界](../../../../evidence/gsm8k-author-comparison-001/publication_boundary.md)。

历史环境、重新运行和资源退出的一般步骤见 [公共操作说明](../../../operations/historical-reproduction.md)。本次文档整理未再次生成答案。
