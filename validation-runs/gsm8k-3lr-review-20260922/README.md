# GSM8K三学习率无人值守案例验收

日期：2026-09-22。源实验仓库：用户的私有gsm8k-grpo-lab。审阅本地提交 `cff186f88b54307a2a70c4249761511589648cd6`，工作树干净；双父为 `ece16b7557a4b6fade4442564375141c7733fcbd` 与 `0cb5ea5bfa6e54f6e294e8731ab4d1f3f6b51197`。

## 结论

**可以作为“正式实验阶段无人值守串行三次，并交付、审核和merge”的经验案例收录。** 用户在本次验收中明确：数值是否贴近旧结果不作为收录这种使用经验的前提。该决定不追溯修改旧的0.5pp复现门槛，也不免除已批准的轮询和报告契约。

| 维度 | 独立结论 |
|---|---|
| 正式三次执行 | 通过：各64步、独立两臂1319题、阶段和实验臂串行 |
| 原文评分与统计 | 通过：7914条回答重新调用锁定数值verifier，correctness/format/parsed/reason全部匹配；独立bootstrap/McNemar/Holm复算一致 |
| 无人值守执行区间 | 通过：首个baseline至最后cleanup为2026-09-21 06:27:37 UTC至09-22 00:58:52 UTC，无新用户消息；Sol只有18次等待调用，Luna-Max负责执行 |
| 实际模型 | Sol / medium主代理，Luna / max执行者；符合后续用户改为Sol的选择，不宣传成Terra-Ultra实测 |
| 本地Git整合 | 通过：真实双父merge和干净工作树；通过既有认证只读核验远端main，SHA与本地一致 |
| 轮询约定 | 未通过：1128条记录，26条本地监测命令/传输错误；同阶段间隔中位数50—51秒，未遵守分钟序列 |
| 报告与图契约 | 待修订：F3/F6实际内容替代了合同要求；报告仍写待审核/合并，主要图片仅作链接，没有逐图可见中文图注 |
| 原0.5pp复现门槛 | 未通过：同学习率1e-4为938/1319，相对历史970差2.426pp；此项单列，不否定自动执行事实 |

## 数值

| 学习率 | Before | After | 增益pp | 最早最高dev step | Holm-3 p |
|---|---|---|---:|---:|---:|
| 1e-4 | 837/1319 | 938/1319 | 7.6573 | 48 | 8.6027e-10 |
| 5e-5 | 837/1319 | 950/1319 | 8.5671 | 48 | 8.3963e-12 |
| 2e-4 | 837/1319 | 889/1319 | 3.9424 | 32 | 0.00250376 |

九个baseline/train/after作业总墙钟17.9753小时；训练板级功率采样梯形积分4.4807kWh，不含主机CPU与audit，也不是电表读数。

## 复核范围与证据

- [observations.json](observations.json)：全部原文重评分、222项远端历史文件哈希映射、dev选择、统计、清理回执、轮询摘要及历史结果差异。
- [autonomy.json](autonomy.json)：本地完整运行转录与阶段时间交叉核验后的脱敏角色/交互摘要；完整转录不公开。
- [execution-order-and-cost.json](execution-order-and-cost.json)：15个阶段的串行时间与资源采样积分。
- [verify_observations.py](verify_observations.py)：只读审核脚本，需要获得源实验库和固定canonical audit，示例命令如下。

```sh
python verify_observations.py --repo /path/to/private-lab --canonical /path/to/gsm8k/audit.jsonl --reference /path/to/skill/example/results/gsm8k-grpo-formal-001 --output /tmp/new-audit.json
```

原验收脚本audit_three_arm.py也实际重跑通过，但它主要验证保存分数和身份，不能单凭该PASS声称做了全文重评分；本次额外核对标准答案和每条response。原始before的文件hash不同但生成内容可能一致，不能仅凭时间字段导致的字节差异证明独立生成；启动器、独立PID、串行时间和执行源码共同支持执行来源。

本轮查看了代表性F3/F6/F4实际图像，足以确认图文契约问题；没有对全部21张图重新签发独立视觉通过结论。未重新训练、未修改源实验库、未删除权重、未实时重新查询GPU；清理和释放核对的是所保存的终态回执，当前设备状态不属于这些历史回执能证明的事实。

## 历史差异的边界

1e-4与旧实验的初始LoRA身份相同，训练参数序列化差异只有输出/日志路径；最初256条rollout的ID、回答、token与评分也一致，第一次分岔出现在记录step2。数据/Arrow兼容层、机器与运行时仍有差异，目前不能把性能差归因到某一项，也不能由相同seed保证数值一致。

## 后续改进

将smoke/环境准备交给主代理，正式实验交给执行者；用持久化时钟执行轮询；保持固定stage身份和重复启动拒绝；同步索引不覆盖原始manifest；权重评测和身份核验后按本run白名单清理；合并后做报告状态与图契约收尾。具体用法见 [指南](../../example/guides/gsm8k-grpo/README.md)。
