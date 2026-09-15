# GSM8K GRPO 实验设计

类型：历史实验设计说明。对应 [实验报告](report.md)，不是新的待执行任务。

## 问题与比较

检验从原始 Qwen3-0.6B 出发的 64 步 LoRA GRPO，能否在完整 GSM8K test 上至少提升 5 个百分点。对照为原始权重与按 dev 选择的 adapter，固定题目、题序、CoT 提示、原生 thinking、采样及 1536 输出预算。

训练从 6961 题池采样，G=8、B=16，64 步产生 8192 个训练回答；每 16 步评价固定 128 条 dev。选点规则是最早最高，实际选到 step 64。test 的 1319 题不用于选点。

## 判据与可能的混杂

统计采用同题配对 bootstrap 与精确 McNemar，并为两个领域保留校正范围。效果之外还要验证训练步骤、梯度、参数变化、完整评价、评分与实际所选权重。完整数值判据保存在 [acceptance.json](../../../../results/gsm8k-grpo-formal-001/acceptance.json)。

最终答案正确性同时受推理、格式和预算内结束影响，本设计不分离这些成分。仅一个训练 seed；不把未训练的原始基线、训练中 dev、最后 checkpoint 与最终所选 checkpoint 混为同一对象。

## 程序版本

训练源为 `75e05a621004e1e29bfd05b4f6b859afc89fc8e4`；完整 before/after audit 源为 `9359731bc1fcdeeef867ff4c5dd1144ff9cdc11f`。审阅分支的新文字不改变两者的历史版本。每项的实际参数与 argv 见 [复现胶囊](../../../../repro/experiments/gsm8k-qwen3-0.6b-grpo/experiment.json)。
