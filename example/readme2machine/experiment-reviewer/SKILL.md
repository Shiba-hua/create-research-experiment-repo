---
name: experiment-reviewer
description: 审核本仓库实验分支的报告、机器材料、原始证据和图表是否符合约定时使用。冻结源与 experiment 的 SHA，区分文档合规和科学结论；只有承担主代理职责并获合并授权者可执行最终 merge。
---

# 实验审核者

先读[目录规范](../../docs/standards/documentation-system.md)与[报告标准](../../docs/standards/experiment-report.md)。审核输出须给出具体证据和处置，不能仅写“检查通过”。独立审核子代理可以提出通过／驳回建议；它不能冒充主代理完成最终批准，提交者也不能通过切换 skill 自行批准。

## 冻结审核对象

从移交记录读取 topic 分支名 `SOURCE_BRANCH`、完整 `SUBMITTED_SOURCE_SHA` 与 `SUBMITTED_TARGET_SHA`。这些是提交身份，尚未获准；不能用现场最新值替换过期移交。fetch 后检查远端 topic 是否与移交 SHA 相同，并用移交值执行初审检查。

```sh
python scripts/check_submission.py --source "$SOURCE_BRANCH" \
  --target origin/experiment --expected-source-sha "$SUBMITTED_SOURCE_SHA" \
  --expected-target-sha "$SUBMITTED_TARGET_SHA" --require-frozen --require-clean
```

检查器只读取本地工作树和 refs；运行前 fetch 才能使 `origin/experiment` 代表刚查到的远端。要求当前目标是 source 的祖先。若任一端前进，退回提交者更新移交并重审；尚未纳入 topic 的最新版目标需要先 merge/rebase，已纳入则不用重复合并。只检查当前祖先关系的 PASS 不能证明原移交仍有效。即使 Git 显示无冲突，也不能沿用旧批准。

## 审核内容与结论

逐项检查本次 diff 和受影响的完整实验档案，而非只看新增文件：

- 0—6 报告及设计／复现入口完整；研究线、实验、run ID 的归属明确；执行、证据有效性和结论状态相互一致。将数据准备、CPU smoke、训练完成和正式科学验收分别表达。
- 模型版本与权重链、数据身份及污染／划分边界、评分代码、实际 prompt、配置、运行代码、环境、命令与产物能相互定位。逐阶段真实源码身份可查；选点规则没有借 audit 事后挑选最优。
- 原始证据与统计分母一致，失败、截断、无效重试、负结果和缺测都披露；重复上传或重算不增加独立实验数量。不可恢复材料标明缺口，不能补造人工标注、测量、时长或种子结果。
- 图能由保存数值和 Matplotlib 代码重建，且已实际视觉检查。成本口径包括 reasoning 与失败题，不把输入混入 decode 或重复计数；单点、独立多预算与轨迹完成门控含义清楚。同组策略可比较，下降和缺测保留。
- CI 有可解释的统计对象、算法、水平与假设。固定题集本次分数是点估计；条件 Wilson 不能写成训练稳定性或配对增益区间。质量保持、因果加速、显著提升等结论有相应设计支持；仅 token/s 或训练曲线不足以证明。
- 相关复现、哈希、链接和代码检查实际通过，例如 `python repro/verify.py`；检查日志注明范围。链接／字段／哈希正确不代表科学结论通过，合成测试不得当作真实实验结果。

审核记录绑定两个完整 SHA、审核者职责、审阅文件与证据、实际检查命令及结果、问题处理和结论。分别给出“仓库交付是否合规”与“实验支持什么结论”：有效负结果可以合规合入，无效结果可作为明确标记的故障记录保存；缺少关键材料或夸大结论须退回修订。

## 主代理合并

只有主代理审核通过且处于用户授权的提交范围，才执行以下整合流程。下面变量必须来自审核记录，不能在合并前改成当前值掩盖过期批准：`SOURCE_BRANCH`、`APPROVED_SOURCE_SHA`、`APPROVED_TARGET_SHA`。

1. fetch 后复核远端 source 和 target 均等于批准 SHA；切换干净的本地 `experiment`，使其与已批准的远端目标一致（可做 `git merge --ff-only origin/experiment`；若本地有分叉或新提交则停止重审，不 reset／强推）。
2. 用冻结值再次检查；先检查 `origin/experiment`，再检查本地 `experiment`。两个检查都必须通过：

   ```sh
   python scripts/check_submission.py --source "$SOURCE_BRANCH" \
     --target origin/experiment --expected-source-sha "$APPROVED_SOURCE_SHA" \
     --expected-target-sha "$APPROVED_TARGET_SHA" --require-frozen --require-clean
   python scripts/check_submission.py --source "$SOURCE_BRANCH" \
     --target experiment --expected-source-sha "$APPROVED_SOURCE_SHA" \
     --expected-target-sha "$APPROVED_TARGET_SHA" --require-frozen --require-clean
   ```

3. 在 `experiment` 用 `git merge --no-ff --no-commit "$APPROVED_SOURCE_SHA"` 合入批准的精确源码版本。形成审核记录并运行合并后的相关检查，再提交 merge commit。父提交应依次是已批准 target SHA 和 source SHA；不要 squash、cherry-pick 或直接复制报告来代替这次分支合并。
4. 仅由主代理普通 push `experiment`，核对远端 SHA 与合并提交一致。若远端前进导致 push 拒绝，停止沿用本次批准，重新同步、冻结和审核，禁止强推覆盖。

检查器不会执行 fetch、push、merge 或修改托管端权限，也不能认证谁是主代理。以上身份与批准是工作流约束；没有实际配置并验证服务器端保护时，不声称它是不可绕过的权限控制。
