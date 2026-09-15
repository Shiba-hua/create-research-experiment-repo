# 作者八样例 CoT／CoD 对照的设计

类型：历史实验设计说明。结果见 [report.md](report.md)。

## 研究变量

保持原始 Qwen3-1.7B 权重、1319 道问题、原生 thinking、采样与预算不变，改变风格指令和八例推导。两组的八道示例问题与顺序相同，但 CoD 示例使用简写。控制的是整个提示变体，不是只替换一个词的消融。

## 实际输入怎样构造

作者配置固定在 [`sileix/chain-of-draft@a7dbf5d`](https://github.com/sileix/chain-of-draft/tree/a7dbf5dea808b1aa1e12f7a90ea573321581df78)。程序依次拼接：风格说明、八个问题与推导答案、当前未作答问题。整个 payload 放进**一条 user 消息**，再应用 Qwen3 的原生模板。配置字段名 `system_prompt` 不等于实际存在独立 system 消息，八例也不是八轮聊天。

每步不超过五词只是一条提示目标；没有 grammar、强制删词或后处理缩短。本适配开启原生思考，允许锁定模板只预填 assistant header，由模型生成思考 opening 和 closing。

完整模板、实际风格指令、八例来源及重建程序集中在 [prompts/gsm8k-author-8shot](../../../../prompts/gsm8k-author-8shot/README.md)。八例原文作为固定外部输入，来源锁包含文件 hash，不能用未来更新的 main 配置代替。模板重建会逐题核对原运行的提示哈希。

## 评价与适配差异

主要严格分数要求思考频道合法闭合，final 中只有一个 `####`，后面为有效数值。作者兼容分数保留其首数字等回退，单独列为诊断。两者均保留全量题目与失败成本。

作者默认客户端用 temperature=0 和普通 API response；本适配使用原生 Qwen3、temperature=.6、top-p=.95、top-k=20、min-p=0、固定 vLLM0.10.2，并明确切分 thinking/final。因此只能称方法适配。完整数据、模型和采样细节见报告第3节及 [机器胶囊](../../../../repro/experiments/gsm8k-qwen3-1.7b-author-cot-cod/experiment.json)。

没有注册非劣界限，也没有观察其他预算或多 seed。结果不应按事后选定的阈值升级为“质量保持”。
