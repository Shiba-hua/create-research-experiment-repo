# 待实例化的科研项目模板

这是建仓材料，不是已批准项目。先选择仅本地Git，或本地＋远程关联同步；后者需先检查平台、归属、仓库与建仓/推送认证条件。先用 [审批单](approvals/approval.md) 对齐并获批，再实例化；删除此段模板说明并填写项目问题、结果入口、负责人和状态。

```text
README.md                         人类入口：问题、主要结果和导航
STATUS.md                         截止日期；执行、证据与结论分开
AGENTS.md                         机器协作入口
approvals/                        版本化方案、批准记录与变更单
readme2machine/                    负责人批准的协作配置、执行及审核说明
asset/                            反复复用信息；按种类分目录
docs/standards/                    报告及目录契约
docs/experiments/<experiment-id>/  report、design、reproduce、必要 figures
repro/experiments/<experiment-id>/ 协议、配置、版本、命令、环境、输入身份
results/<run-id>/                  原始观测及产物
evidence/<run-id>/                 运行、校准、异常、人工记录
```

按批准方案按需建立实验/运行子目录。先阅读 [asset 规则](asset/README.md)、[报告契约](docs/standards/report-contract.md)、[证据说明](repro/README.md)。不创建空结果冒充已实验。

协作方式先对齐：多智能体可用时推荐主代理派发/审核、子代理执行/交付；具体模型与低成本路由在 [collaboration.md](readme2machine/collaboration.md) 中确认，人类/设备执行也使用相同任务与移交边界。
