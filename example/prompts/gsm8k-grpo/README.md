# GSM8K GRPO 的实际提示模板

模板 ID：`gsm8k-grpo-cot-v1`。对应 [Qwen3-0.6B GRPO 报告](../../docs/experiments/grpo/gsm8k-qwen3-0.6b/report.md)。

实际消息有两条：`system` 放任务及最终答案格式要求，`user` 放当前题目的英文原文。完整指令逐字保存在 [template.json](template.json) 的 `messages[0].content`；唯一数据变量是 `question`，组装时去除首尾空白。

```text
system: {{template.messages[0].content}}
user:   {{question.strip()}}
```

没有 few-shot 示例、标准答案或参考解答进入消息。指令要求逐步推理，并在最后一行使用 `Answer: <number>`。训练前后、dev 和 audit 沿用此模板；它不是作者 CoD 实验的 `####` 模板。

随后应用锁定的 [Qwen3 chat template](../qwen3/chat_template.jinja)，设置 `add_generation_prompt=true, enable_thinking=true`，最后停在 assistant 的生成起点。原生 thinking 与指令中的 CoT 表达要求分别记录，不混为一个设置。

CPU 重建任一问题的实际输入：

```bash
python prompts/render.py --template gsm8k-grpo \
  --question-file /path/to/question.txt --output /new/path/grpo-prompt.json
```

历史来源为 `9359731:src/rlvr_lab/data.py` 的 `build_prompt`；训练阶段使用相同构造。模型模板来自固定 tokenizer 配置，身份见 [模板来源](../qwen3/source.json)。判分规则见 [GSM8K verifier](../../datasets-and-verifiers/gsm8k/verifiers.md)。
