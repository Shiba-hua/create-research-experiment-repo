# GSM8K 实验管理示例

本示例保留两项真实实验及必要对照：

| 问题 | 结果 | 报告 |
|---|---|---|
| GRPO 能否提高数学答题正确率？ | Qwen3-0.6B，837/1319 → 970/1319 | [GRPO](docs/experiments/grpo/gsm8k-qwen3-0.6b/report.md) |
| CoD 提示能否在保持质量时节省 token？ | Qwen3-1.7B，严格正确969/1319 → 922/1319，输出 token 减少10.26% | [CoT/CoD](docs/experiments/cot-cod-prompt/gsm8k-qwen3-1.7b-author-8shot/report.md) |

从报告进入设计、图表、复现配方、prompt 和数据/verifier，再追到 results 与 evidence。CoT 是 CoD 的必要基线；原始模型是 GRPO 的必要基线。未纳入其他数据集、冷启动/SFT、速记或系统实验。

## 入口

- [机器协作规则](readme2machine/README.md)
- [实验登记](repro/catalog.json)与[复现说明](repro/README.md)
- [GSM8K 数据与判分](datasets-and-verifiers/gsm8k/README.md)
- [实际提示](prompts/README.md)
- [来源与脱敏说明](PROVENANCE.md)

此例是历史结果的筛选导出，不是新的模型训练。原始模型权重、canonical 数据及作者八例需从固定外部来源取得；CoD 的全文响应未公开，公开数值投影只支持统计复算，不支持全文重新评分。GRPO 的完整保存回答与逐步训练记录在包中。

## 本地核验

在本目录运行，Python 3.11+，数值复算需 numpy，绘图另需 matplotlib：

```sh
python3 repro/verify.py
python3 repro/recompute.py grpo
python3 repro/recompute.py author
```

核验导出文件身份、复算旧数据与重跑 GPU 是三个不同层次。命令及限制详见 [复现说明](repro/README.md)。
