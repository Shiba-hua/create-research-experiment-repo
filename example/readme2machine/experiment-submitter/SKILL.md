---
name: experiment-submitter
description: 在本仓库提交实验报告、结果、复现材料及相关文件时使用。建立源自 experiment 的实验分支，整理可追溯的完整交付，并交给主代理审核；不负责合并到 experiment。
---

# 实验提交者

先读[目录规范](../../docs/standards/documentation-system.md)与[报告标准](../../docs/standards/experiment-report.md)，按所属研究问题更新实验档案。两份规范定义内容格式；本 skill 定义提交责任与分支流程。

## 分支与责任

`experiment` 是已审核成果的入口。实验 agent 不得在整合分支上直接 commit、push、强推或代替主代理合并。

开始编辑前 fetch `origin` 的 `experiment`，从其确切提交创建独立分支，例如 `codex/gsm8k-grpo-run-003`。已有工作树有未提交改动时，保留它并使用独立 worktree；不要清理别人的改动。

```sh
git fetch origin experiment
git switch --create codex/gsm8k-grpo-run-003 --no-track origin/experiment
```

实验源码版本、文档提交版本、分支起点是不同身份：逐阶段保存实际执行时的源码 SHA，不能用本次报告提交 SHA 代替历史运行源码。新增运行使用独立 run ID；同一比较的重试、失败和预算点归入同一实验档案。

## 交付材料

每项材料放到规范指定的位置，形成报告→机器配方→原始证据的可追溯链：

- 完整 `report.md` 使用 0—6 主干，配齐设计、复现和必要诊断。元数据分别标明执行状态、证据是否有效、结论是否支持；失败、无效对照、负结果与缺测不隐藏。
- `repro/experiments/<experiment-id>/` 绑定实际模型及初始化权重、数据划分、verifier、prompt、配置、源码、环境、命令和产物。GRPO 明示 G、B、优化步数及真实 SFT 起点；重试版本逐项绑定。没有保留权重时据实披露可重新训练的范围。
- prompt 模板引用 `prompts/` 的精确组装入口；数据简介、典型点、可靠外部来源及实际评分代码引用 `datasets-and-verifiers/`。禁止以概念模板替代实际输入，禁止把外部受限材料当作可再分发材料。
- `results/` 与 `evidence/` 保存原始数值和运行证据，不因报告改写而回写历史。体积大或受限材料使用来源锁、哈希、恢复步骤及可用性说明，不创建空文件冒充交付。已有历史证据发现错误时增加纠正记录并保留原件。
- Matplotlib 图必须附原始数值与可复算代码，逐图核对分母、单位、图例和可见图注。同组策略画在同一比较中；缺测留空。decode cost 包括 reasoning、final 与实际返回的 EOS，所有失败／截断题计入分母，输入 token 单列，不能重复计入 reasoning。独立预算评测、单次轨迹回看和单点明确区分；长度 CDF 不是 val_acc 曲线。
- 误差条披露估计对象、单位、方法、名义水平、假设及不覆盖的变异；单次训练不能声称多 seed 区间。若使用条件 Wilson 区间，说明假想 IID 题目总体假设未经本实验验证。无合适统计对象时只报点估计。

## 自检与移交

按仓库现有复现入口运行与本次改动相关的检查，例如 `python repro/verify.py`；保存检查命令、结果与适用范围。检查失败不得标为已通过。机械一致性通过并不证明实验成功，未运行的验证标明未运行。

完成文件后，在 topic 分支逐交付项 commit。再次 fetch 最新目标；如目标前进，把它 merge 或 rebase 到 topic，解决冲突后重做受影响的验证。采用保守规则：**当前目标必须是提交源的祖先**，仅存在较早共同祖先不足以移交。

```sh
python scripts/check_submission.py --source codex/gsm8k-grpo-run-003 \
  --target origin/experiment --require-clean
```

检查器只读本地 refs，不联网。保存输出的完整 `source_sha` 和 `target_sha`（审核基点），以及报告／复现入口、变更摘要、验证结果、已知限制和仍缺少的证据，交给主代理。审核材料写到工作树外，避免污染 `--require-clean`。得到的 PASS 仅是分支机械检查，不是主代理批准。

有向本仓库提交的授权时，仅 push 当前 topic 分支，并核对远端 SHA；不得推到 `experiment`。审批后 source 或 target 任一 SHA 改变，都要重新移交审核。最后停在移交状态，不能通过调用 reviewer skill 给自己的提交签发主代理批准。
