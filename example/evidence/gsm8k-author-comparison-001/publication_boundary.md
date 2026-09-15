# 原始输出私有保存，公开数值投影

原始预测没有 `generation_prompt` 或 `generation_prompt_token_ids` 字段，但这不足以证明文件不含作者输入：模型自身会回显示例。对标准化后的作者完整问句作精确包含检查，长度大于120字符的问句在 CoT 的24条、CoD的55条输出中出现。这个计数只描述所列长字符串的精确匹配，不是语义重复率或法律判断。[仅含ID／哈希的扫描记录](public_input_exclusion_check.json)

完整原始 JSONL 和输出 token IDs 没有删词、裁短或改分；用户服务器原件保持不变，gzip 按原SHA保存在 Git 忽略的 `private/author-original-outputs/{run}/`。公开发布分支不包含这两个 gzip，也不包含可还原输出文本的解压文件或 token 序列。原始 SHA／字节数／行数及恢复 receipt 继续公开，用于验证持有的私有原件。

公开的是两个 `author_metrics.jsonl` 与各自 manifest：保留全部1319题的标识、固定枚举、哈希、两种分数、实际 token 数、时间和终止标记，额外绑定每条原始物理行的 SHA256。删除全部输出文本、解析文本、标准答案文本和 token ID 序列。每行只含标量，没有可重构文本的数组；manifest 的字段集合／生成程序身份也被固定。

[生成程序](build_public_projection.py)先验证完整原始文件SHA与summary／receipt，再进行白名单投影；拒绝未知字段和非预期 parser/mode/config 身份，receipt 仅复制固定元数据键。输出是**数值投影，不是原始预测文件**，不能提供给原 tokenizer／parser 比较器冒充完整证据。

[公开数值验证](public_projection_validation.json)由[可复跑程序](verify_public_projection.py)核对所有行、字段结构、原题序、counts、token成本、翻转和20k配对区间，并绑定comparison／builder／verifier／两投影及其manifest身份。完整文本／tokenizer／parser复核则已经在 masked guest 对私有原件完成，二者没有相互替代。[独立复核](public_projection_review.json)

原始 `author_plan.json`、`author_summary.json`、`author_status.json`、请求、job／release及日志字节保持不变；它们仍引用原始预测文件名和SHA。原先的归档校验与路径记录作为历史材料保留。当前公开文件校验请使用各run的 `PUBLIC_SHA256SUMS`，原 `SHA256SUMS` 仅表示私有迁移前的历史归档。


本导出额外进行了路径脱敏；当前字节身份用根EXPORT-MANIFEST.json验证。上文程序与PASS描述原历史上下文；公开数值复算使用repro/recompute.py，不要求私有原件。
