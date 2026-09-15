# GSM8K 的三种实际判分入口

本项目保留两套历史实现、三个分数入口。它们不检查推导过程是否正确，也不执行模型写出的表达式或代码。数据背景与标准答案格式见 [数据说明](README.md)。

| 入口 | 使用实验 | 主要返回字段 | 代码 |
|---|---|---|---|
| `grpo` | Qwen3-0.6B GRPO 的训练 reward、dev 和 audit | `correctness`、`format_ok`、`parsed`、`reason` | [project_math.py](code/project_math.py) |
| `author-strict` | Qwen3-1.7B 作者 CoT／CoD 的主结果 | `strict_correct`、`strict_format_ok`、`strict_parsed` | [author_math.py](code/author_math.py) |
| `author-compatible` | 同一作者提示实验的辅助诊断 | `author_correct`、`author_method` | 同上 |

## GRPO：显式最终答案与精确数值

**先重放调用端，再调用纯 verifier。** 原生 Qwen3 的训练、dev、audit 在 `thinking=true` 且返回文本去除前导空白后不以 `<think>` 开始时，先给检查文本补上 `<think>\n`。本例不包含关闭原生思考的实验。这不是改写模型原始输出，也不增加实际 decode token 成本。比如原始 `Answer: 42` 在原生 thinking 入口会因未闭合思考而失败。CLI 要求显式选择 `--native-thinking true|false`，不能仅因底层函数逐字相同就把两类调用视为相同。

先检查思考是否闭合，再只处理最后思考标签之后的 final。最后一行必须是 `Answer: ...`，或输出以合法 `\boxed{...}` 结束；普通正文里的最后一个数字不算提交答案。

解析接受受限的整数、小数、科学计数、分数及简单 LaTeX 分数，用 `Fraction` 精确比较，不使用浮点容差。逗号分组须合法；数字表达式如 `2+3` 不会被求值。可接受单层 `<18>` 这种由模板占位符带出的外括号。

final 中若有多个显式 `Answer:` 或 boxed 答案，它们必须一致。未闭合思考、缺少最终答案、解析失败、显式答案冲突都得0；格式合格但数值错误也得0，但 `format_ok=true`，与格式失败区分。

## 作者严格评分：原生思考结构与一个 `####`

本次 `thinking_opening_prefilled=false`，所以生成必须以 `<think>` 开始，并且标签序列恰好是 `<think>`、`</think>`。少标签、多标签或不合法起点先判失败。opening 若由其他实验的模板预填，必须显式传入相应布尔，不能自动猜测。

只在 closing 之后评分。final 中必须恰好有一个 `####`，其后全部内容必须是一个带可选正负号的整数／小数，可有合法千位逗号；不接受单位、货币或百分号后缀、分数表达式、科学计数或数字回退。数值仍用 `Fraction` 精确比较。`####` 之前允许解释文字，因此这项检查也不能证明中间推导正确。

## 作者兼容评分：保留原方法的宽松回退

它与严格评分共用思考结构检查，但 final 的答案提取较宽松：优先取首个 `####` 后片段（没有分隔符就取整个 final），清理逗号、美元和百分号，再做字符串比较；不相等时回退到首个数字及浮点比较。

原数字回退忽略负号，且可能只取长整数的前三位。我们保留这个历史行为以解释原报告，同时把严格评分作为主结果。两种评分的正确样本集合不一定相互包含，不能把“兼容分数”称为严格分数的简单宽松超集。

以下是自造输出示意，用于解释实现，不是模型实测案例：

| 标准答案与输出 | GRPO | 作者严格 | 作者兼容 | 原因 |
|---|---:|---:|---:|---|
| 标准18；`<think>…</think>` 后 `Answer: 18` | 1 | 0 | 1 | GRPO 格式正确；作者兼容可回退找数字 |
| 标准18；`<think>…</think>` 后 `#### 18` | 0 | 1 | 1 | 两类主输出格式不同 |
| 标准18；思考未闭合，即使中途出现18 | 0 | 0 | 0 | 没有合格最终输出 |
| 标准18；final 先写 `Answer: 17` 再写 `Answer: 18` | 0 | 0 | 0 | GRPO 检测冲突；作者没有严格分隔符，兼容先读17 |
| 标准42；合法思考后 `#### -42` | 0 | 0 | 1 | 作者兼容回退丢失负号，产生假阳性 |
| 标准42；合法思考后 `#### 00042` | 0 | 1 | 0 | 严格数值为42，兼容回退可能只取前三位000 |

## 格式、EOS、截断和准确率是不同字段

上面的 verifier 处理文本；EOS 和截断由生成记录另行标记，不自动改变判分。在本次原始记录中，四臂的**主严格分数**均没有“判对但截断”的样本；作者兼容分数则有 CoT 3题、CoD 2题判对但截断。不能偷偷加上 EOS 条件后仍沿用原来的兼容准确率。

普通准确率以全部1319题为分母，包括格式失败和截断。新的轨迹跃变图使用严格分数，并先检查所有判对记录均正常结束；其终点才能与原严格准确率保持一致。

## 直接运行与历史来源

```bash
python datasets-and-verifiers/gsm8k/code/verify.py \
  --mode grpo --completion-file /path/to/answer.txt --answer 18 --native-thinking true
python datasets-and-verifiers/gsm8k/code/verify.py \
  --mode author-strict --completion-file /path/to/answer.txt \
  --answer 18 --opening-prefilled false
```

辅助作者评分将 `--mode` 改为 `author-compatible`。CLI 输出所选分数及原始完整 verdict，不将不同字段合并成一个模糊的“正确”。

GRPO 文件来源为 `9359731:src/rlvr_lab/rewards.py`，导出时删除其他数据集分支，GSM8K数值逻辑保留；作者文件从 `61ff6a4:src/rlvr_lab/cod_author.py` 提取原四个纯评分函数，补标准库 import，未改逻辑。源文件及函数身份见 [source-lock.json](code/source-lock.json)，验证用例见 [测试](../../tests/test_verifiers.py)。
