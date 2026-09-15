# 作者 CoT / CoD 真实实验图：图例修复版 002

本目录基于已通过完整 CPU 比较验证的 `figures-001/plot_data.json` 重画五组真实结果图，每组提供 PNG、SVG、PDF。所有绘图数值、样本、标签与评分口径沿用原数据；`plot_data.json` 的字典及完整字节均与 001 一致，SHA256 为 `4294d7697a1199820a9290f5a394d0f0d772f0a7f584822e4d88f7dbc35992e6`。

1. `01_accuracy`：两种评分的正确率及配对区间。
2. `02_paired_flips`：逐题改善与退步矩阵。
3. `03_actual_token_cost`：全部问题的实际输入与输出成本。
4. `04_length_and_termination`：长度分布、配对 token 差及终止诊断。
5. `05_timing`：串行运行计时；左图图例移到坐标区域上方，避免遮住 `778.7`。

唯一布局干预是 timing 左图的图例位置。五组图全部在本地 Matplotlib 3.10.9 / NumPy 2.4.6 重新导出，原 001 使用 Matplotlib 3.10.3 / NumPy 2.2.6，因此不声称其他图文件字节不变。原 001 全部图、数据与 manifest 保留原字节。

本次只重画布局，没有重新加载 tokenizer、调用比较器或使用 GPU。`plot_manifest.json` 中 `full_comparison_recomputed=false`、`full_comparison_recomputed_in_this_render=false`、`comparison_verified_upstream=true` 明确区分这一步与原 001 的完整验证；原比较报告、绘图数据及 manifest 的 SHA256 均已绑定。

[修复与数字一致性证据](../../evidence/gsm8k-author-figures-002/README.md)说明修复脚本、原 renderer 身份和检查结果。正式可读性结论由主任务后续的完整 Luna 视觉审计给出，本目录的生成记录不冒充视觉验收。
