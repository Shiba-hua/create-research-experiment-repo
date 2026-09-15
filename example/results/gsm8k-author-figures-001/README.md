# 初轮真实数据图：保留的视觉修复前版本

这组五图由固定源码 `61ff6a476c3cacec1fe87c846c9559a010f499dd` 在 masked CPU guest 中运行 `scripts/plot_cod_author.py` 生成。脚本重新验证完整配对数据、评分、输入输出 tokens 与 comparison JSON 后，输出 PNG／SVG／PDF。`plot_data.json` SHA256 为 `4294d7697a1199820a9290f5a394d0f0d772f0a7f584822e4d88f7dbc35992e6`。

**初轮 timing 单图视觉审计为 FAIL**：左图 legend 遮挡 CoD generation batch wall 的 `778.7` 注释。该审计完成1/1任务，Luna模型正确，发现1项 blocking layout collision；未据此宣称本目录其余所有图位已完成视觉验收。原始数值与15个图文件保留，不回写为修复后版本。

最终报告使用 [figures-002](../gsm8k-author-figures-002/README.md)。新版本保持绘图数据逐字节相同，仅有 timing legend 的布局干预；没有重新运行模型，也没有改变统计结果。[初轮 CPU 与审计记录](../../evidence/gsm8k-author-figures-001/README.md)
