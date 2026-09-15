# 模型与数据

GRPO 使用 Qwen3-0.6B，CoD提示使用原始Qwen3-1.7B。精确 revision、文件哈希见各自机器胶囊 assets。两项采用 [GSM8K](../../datasets-and-verifiers/gsm8k/README.md) 同一锁定划分：6961 train、512 dev、1319 audit；GRPO选点只看固定128 dev，CoD不训练。权重与canonical数据未打包，数据划分去重不能证明无预训练污染。
