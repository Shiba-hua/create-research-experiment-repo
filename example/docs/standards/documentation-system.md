# 本例目录与协作

README面向读者；docs/experiments/<研究线>/<实验>/含report、design、reproduce、figures与必要diagnostics。repro/experiments/<实验ID>保存实际命令、配置、环境、输入版本与源码。results/<run>是原始观测，evidence/<run>是运行与异常。两类复用信息prompts、datasets-and-verifiers放一级目录。

研究问题、实验、一次运行分别登记，重试不增加独立研究数量。报告连接机器配方和原始证据，不能复制失去身份的“最终版”。历史源码SHA与本次导出SHA不同。

本例展示experiment分支整合与topic提交，审核绑定source/target精确SHA，过期后重审，主协调者按已有授权merge。新项目在审批单确认是否采用此流程；文本不等于平台权限。
