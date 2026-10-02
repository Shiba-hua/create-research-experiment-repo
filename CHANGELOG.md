# Changelog

本仓库的变更记录。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased] — topic/unattended-experiment-orchestrator

### 新增

- **新 skill：`skills/unattended-experiment-orchestrator/`**，面向计算机实验的无人值守规划与编排。
  - **分工**：AI 只在开跑前（配环境、debug、最小冒烟测试证明代码能跑、确认记录会落盘）和收尾后（负责人明确指示时写报告）工作。运行期间由编排脚本和看门狗脚本无人值守，AI 不轮询、不定时唤醒、不中途介入。
  - **工作流**：对齐目标与边界 → 有异常先查根因 → 可行性预算 → 环境与 debug → 最小冒烟测试 → 编排器 → 两层看门狗 → 开跑前确认 → 提交并下线 → （运行中改条件的处理流程）→ 收尾报告。
  - **参考文档**：`references/orchestration-patterns.md`（目录布局、矩阵文件、系列作业生命周期、单点程序约定、部署与源码提交号、Slurm 常见坑），`references/watchdog-design.md`（作业内/矩阵两层看门狗的信号与动作、可疑数据判定）。
  - **模板**：`assets/templates/`，Slurm 版本，包括 `matrix.tsv`、`submit_matrix.sh`、`series.sbatch`、`job_watchdog.py`、`matrix_watchdog.py`、`watchdog.sbatch`、`cleanup_node.sbatch`、`status.sh`。
- README 增加该 skill 的说明、安装命令，以及与 create-research-experiment-repo 协作模式的选择表。
- 本 CHANGELOG。

### 来源

提炼自 2026-09 至 10 月在 BLCU-HPC（Slurm，L40S/H20/H20-3e）上的两轮实验：
- 三个玩具实验（RLVR、OPD、CoD）；
- Qwen3.5-9B 单卡并发吞吐实验。

吞吐实验中，最初的“子代理轮询 + 逐任务回执审批”流程跑了三天，没有得到一个完整测量点。改为“开跑前冒烟 + 编排脚本一次提交 + 看门狗无人值守”后，三种卡型的作业在一次提交内全部启动。

模板中的脚本是从真实运行过的版本泛化而来，去掉了项目特定内容。本次提交前做过以下验证：
- 全部脚本通过 `bash -n` 和 `py_compile`；
- `series.sbatch` 在本地模拟了一个系列：一档测完、一档容量失败、其后的档位记为“未测”，并能写出 `run.json` 和 DONE；
- `matrix_watchdog.py` 用伪造的 squeue/scontrol/sbatch 跑了一轮：正确报出可疑点 ALERT、提交同节点清理、从第一个未完成档位续提一次；
- `job_watchdog.py` 在卡死时能向单点程序发送 SIGTERM。

模板在新集群上使用前，仍需按项目修改分区、路径和钩子函数。
