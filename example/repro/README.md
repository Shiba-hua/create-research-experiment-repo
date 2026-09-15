# 两项实验的复现入口

## 1. 查验公开材料

`python3 repro/verify.py` 验证导出清单的当前文件身份。历史哈希与当前导出哈希通过清单区分。查看每个 `experiments/<id>/experiment.json`、配置、commands、environment 和 assets。

## 2. 从保存记录复算

```sh
python3 repro/recompute.py grpo
python3 repro/recompute.py author
```

复算使用保存的全题预测/数值投影、历史统计函数，核对正确数、配对翻转、区间与成本。GRPO 以导出清单验证当前文件，统计复算不再次认证已脱敏的旧签名。提供固定 canonical audit 可用 `--data /path/to/audit.jsonl` 对 GRPO 逐条重新判分；缺少 canonical 数据时不声称全文重评分。

作者 CoD 投影没有回答文本；只能复算已有分数，不能替代完整响应重评分。完整响应需额外获得原件。本例没有这些文件，也不保证外部维护者提供访问。

## 3. 复绘

在新目录输出，避免覆盖历史图：

```sh
python3 repro/accuracy_decode_cost.py --help
python3 repro/trajectory_jumps.py --help
python3 repro/experiments/gsm8k-qwen3-0.6b-grpo/plotting/training_diagnostics.py --help
python3 repro/experiments/gsm8k-qwen3-1.7b-author-cot-cod/plotting/plot_cod_author.py --output /tmp/new-cod-figures
```

每项绘图输入来自 results；新图的 manifest 记录新的输入/脚本身份。报告内历史图、图数据及绘图版本保留。

## 4. 重新训练/生成

参照每项报告的 reproduce.md。先取得原始阶段 checkout、固定模型与数据、作者输入（仅CoD）、匹配的 Python/CUDA 用户态依赖，再将 commands.json 中的示例路径替换为实际路径。训练64步→按dev选step64→原始与所选权重完整audit；CoD保持两臂同一模型、题序、采样和预算。每次新运行用新目录并保存实际命令及环境。

这里不自动下载、启动训练或改动驱动。源码裁剪不能证明重新执行与原历史行为相同；本交付没有做 GPU 重跑。原模型、数据、作者输入与检查点可用性分别看资产锁和报告，不能以文件名假定权重存在。

独立新运行也可复制code/<stage>到新目录、记录为新Git项目，显式绑定模型/数据及资产锁后执行。它是改编代码的新实验，不是还原历史SHA；本次仅验证CPU入口，未验证此GPU路径。严格历史checkout若不可访问，该级复现即有访问缺口。
