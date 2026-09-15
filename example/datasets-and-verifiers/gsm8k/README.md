# GSM8K：数据简介、典型题与本项目划分

GSM8K 是英文小学数学应用题集，由人工编写题目和分步解答，主要考查用若干步四则运算解决文字题。本项目使用 `main` 版本，不使用附带子问题的 Socratic 版本。[作者仓库](https://github.com/openai/grade-school-math) · [原始论文](https://arxiv.org/abs/2110.14168) · [官方数据卡](https://huggingface.co/datasets/openai/gsm8k)。

## 数据长什么样

原始 JSONL 每行包含 `question` 和 `answer`。`question` 是题目原文；`answer` 包含参考推导，最后以 `####` 引出最终数值，还可能包含 `<<表达式=结果>>` 计算标注。参考推导用于保存标准答案，不会拼进本项目的待测问题。

下面是两个真实数据点的中文转述，便于阅读；不是实验实际发送的英文 prompt。

| 原始数据点 | 问题（中文转述） | 解答与标准数值 |
|---|---|---|
| train 第0题，Natalia 卖发夹 | 四月卖48个，五月卖四月的一半，两个月合计多少？ | `48 + 48/2 = 72`；标准数值 `72` |
| test 第0题，Janet 卖鸭蛋 | 每天16枚蛋，早餐用3枚、做松饼用4枚，剩余每枚卖2美元，每天收入多少？ | `(16−3−4)×2 = 18`；标准数值 `18` |

第一题可在 [官方数据卡的首条训练记录](https://huggingface.co/datasets/openai/gsm8k) 核对；第二题在 [固定官方 test.jsonl 的首行](https://github.com/openai/grade-school-math/blob/b0bb162abedc65e1fdd8e93ed090fd7598ee68bc/grade_school_math/data/test.jsonl#L1)。原始英文与完整参考解答以这些来源为准。两例展示先确定数量关系、再计算最终值的题型。

## 本项目实际用了哪些题

| 划分 | 题数 | 用途 |
|---|---:|---|
| 官方 train | 7473 | 本项目再划出训练池与 dev |
| 本项目训练池 | 6961 | GRPO 采样训练题 |
| 本项目 dev | 512 | 其中固定128题用于此次 GRPO 选 checkpoint |
| 官方 test／本项目 audit | 1319 | 两项实验完整最终评价，每臂每题一次回答 |

数据固定在 `openai/gsm8k@740312add88f781978c0658806c59bc2815b9866`，来源及文件锁见 [gsm8k.json](../../evidence/assets/gsm8k.json)。canonical audit SHA256 为 `437f2042d9d210e34834d1f596a0b6289f89414672ede7fe47727041dc11da62`。题序按 seed=20260910 固定打乱；`gsm8k:test:0` 表示原始 test 索引0，不意味着它在评估文件第1行。

canonical 记录将最终数值规范化为可精确比较的字符串，保留题目／原始记录／原始参考答案的哈希；参考长推导不作为模型输入。公开数据可能出现在预训练中，本实验不能排除这种污染。

## 答对／答错到底如何判

参考答案是 `18`，不代表回答中任何位置出现“18”都算对。必须先按本实验的格式提取最终答案，再比较数值。两项实验的具体规则与常见边界见 [verifiers.md](verifiers.md)，实际代码见 [code/](code/)。

原论文也研究训练学习式 verifier；本项目这两项实验使用的是确定性的最终答案规则，没有训练该论文的学习式 verifier。
