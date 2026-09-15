# 作者八样例 CoT／CoD 的实际提示模板

对应 [Qwen3-1.7B 提示工程实验报告](../../docs/experiments/cot-cod-prompt/gsm8k-qwen3-1.7b-author-8shot/report.md)。固定模板 ID 为 `gsm8k-author-cot-8shot-v1` 与 `gsm8k-author-cod-8shot-v1`。

## 两种提示具体怎样不同

[template.json](template.json) 保存实际英文指令：`styles.cot` 要求逐步作答，`styles.cod` 要求每步只保留最小草稿、至多五词；两者共用 `answer_instruction`，要求把答案放在 `####` 之后。`qa_format` 保存每个问答的准确换行格式。

两组还分别使用作者配置中的八段推导。八道示例问题及顺序相同，推导表达不同。因此处理变量包括风格指令与八例答案文本，不能解释成只改一句系统指令。

## 完整组装模板

以下是实际组装算法的可读模板。`qa_format` 各段已有尾换行，段间仍额外加入一个换行；不能擅自压缩空行。

```python
instruction = styles[mode] + "\n" + answer_instruction + "\n"
examples = [qa_format.format(**item) for item in fixed_fewshot[mode]]
payload = instruction + "\n" + "\n".join(examples) + "\n"
payload += qa_format.format(question=question, answer="")
messages = [{"role": "user", "content": payload}]
```

`fixed_fewshot[mode]` 必须恰好是下面固定文件中的八例，当前问题的 `answer` 为空。**整个 payload 放入一条 user 消息**；作者配置名 `system_prompt` 并不意味着本实验有独立 system 消息，八例也不是八轮聊天。

| 固定输入 | 可直接阅读的原文件 | SHA256 |
|---|---|---|
| CoT 的指令、格式与八例 | [gsm8k_cot.yaml](https://github.com/sileix/chain-of-draft/blob/a7dbf5dea808b1aa1e12f7a90ea573321581df78/configs/gsm8k_cot.yaml) | `77fda02c1c68dc9e1f0143fed83e78c9f18febd838bfe2b4efcf86e90acf420d` |
| CoD 的指令、格式与八例 | [gsm8k_cod.yaml](https://github.com/sileix/chain-of-draft/blob/a7dbf5dea808b1aa1e12f7a90ea573321581df78/configs/gsm8k_cod.yaml) | `c83a151f36af1a491080df441044d6dfeb2ada15cefeb98368c2f9e3563f13cb` |

八例作为固定外部输入读取，完整八例文本不另行打包进仓库；作者版本未提供再分发许可。这里保存可读模板、实际风格指令、精确来源及验证式重建程序，不以自造例子替换实验使用的八例。

最后应用 [实际 Qwen3 chat template](../qwen3/chat_template.jinja)，`enable_thinking=true`、`add_generation_prompt=true`。本次模板停在 assistant header，**不预填 `<think>`**；模型自行生成思考的 opening 和 closing。最终输出的严格规则见 [verifiers.md](../../datasets-and-verifiers/gsm8k/verifiers.md)。

## 重建完整实际输入

```bash
python repro/experiments/gsm8k-qwen3-1.7b-author-cot-cod/code/author/scripts/fetch_cod_author_sources.py --output-dir /external/author-inputs
python prompts/render.py --template gsm8k-author-cod \
  --author-source-dir /external/author-inputs \
  --question-file /path/to/question.txt --output /external/rendered-cod.json
```

CoT 把 `--template` 换为 `gsm8k-author-cot`。程序检查源文件 SHA、八例数、指令和格式后生成完整消息与 chat-template 文本；不执行上游代码，也不调用模型。本例只保留上述两种提示家族；重建后可与对应运行中的摘要校验。
