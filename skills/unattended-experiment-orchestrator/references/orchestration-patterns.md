# 编排模式

目录：
1. 目录布局
2. 矩阵文件
3. 系列作业的生命周期
4. 单点的约定（被编排器调用的程序要满足什么）
5. 部署与源码提交号
6. 常见调度器坑

模板在 `assets/templates/`（Slurm 版本，复制到项目的 `scripts/` 后按项目改）：

| 文件 | 作用 |
|---|---|
| `matrix.tsv` | 每行一个系列：id、任务、硬件类型、调度器节点参数、优先级 |
| `submit_matrix.sh` | 一次提交全部系列 + 矩阵看门狗，写 `registry.tsv` |
| `series.sbatch` | 一个系列：启动服务（可选）→ 按档位依次调用 `RUN_POINT_CMD` → 停测规则 → 清理 |
| `job_watchdog.py` | 作业内看门狗（随系列作业在后台运行） |
| `matrix_watchdog.py` + `watchdog.sbatch` | 矩阵看门狗（独立 CPU 作业） |
| `cleanup_node.sbatch` | 同节点清理一个已结束作业的临时目录 |
| `status.sh` | 给人看的状态查看命令 |

## 1. 目录布局

```
<集群数据根>/runs/
  series/<series_id>/job-<jobid>/
    run.json              # 配置、源码提交、版本、硬件身份
    series.log            # 作业主日志
    series_status.jsonl   # point_start / point_done / capacity_stop / not_measured / series_end
    heartbeat             # 每 30 秒更新的时间戳
    alerts.log            # 作业内看门狗写的 ALERT
    p-<level>/            # 每个档位一个目录
      events.jsonl        # 每个数据点的终态（含起止时间）
      metrics.jsonl       # 服务指标时序（如有）
      point_summary.json  # 这个点的汇总和 flags
  matrices/<matrix_name>/
    matrix.tsv  registry.tsv  status.md  events.log  DONE
```

原始记录只在集群；仓库只放汇总和图。

## 2. 矩阵文件

`matrix.tsv` 一行一个系列（制表符分隔）：

```
# series_id  task   hw_type  node_args                 priority
taskA-h100   taskA  H100     --nodelist=node3           1
taskA-a100   taskA  A100     --exclude=node3,node6      1
taskB-h100   taskB  H100     --nodelist=node3           2
```

- 一个系列 = 同一硬件 × 同一任务，档位在作业内依次测（服务只起一次）。
- 优先级小的先提交；超配额的由调度器排队（例如 Slurm 的 `QOSMaxGRESPerUser`）。
- 换模型尺寸/换数据集就是加行，或者加一个维度列，`submit_matrix.sh` 里拼进 `--export`。

## 3. 系列作业的生命周期

```
start → 身份检查（硬件型号对不对）→ 心跳/硬件监控后台进程
      → 启动服务（可选，等健康检查）→ 写 run.json
      → 启动作业内看门狗
      → for level in LEVELS:
            已触发停测？→ 记 not_measured，continue
            写 current_level → 后台运行 RUN_POINT_CMD → 记录 pid → wait
            服务死了？查日志里的 OOM 关键词 → 容量停测 / 服务崩溃
            按退出码记 point_done / capacity_stop / point_failed
      → series_end → trap 清理（停服务、杀后台、删临时目录）→ DONE
```

关键约定：
- `trap cleanup EXIT` 加 `trap ... TERM`：`scancel` 和超时都先发 SIGTERM，要能走到清理。
- `wait` 在后台进程上，否则 TERM 要等前台命令结束才处理。
- 服务的端口按作业号派生（如 `40000 + jobid % 10000`），只绑 127.0.0.1。
- 编译缓存、临时目录放到作业私有的本地目录（`/tmp/<uid>-<jobid>`），多个作业互不干扰。

## 4. 单点程序的约定

`RUN_POINT_CMD` 是项目自己写的测量程序，编排器只要求：

| 约定 | 说明 |
|---|---|
| 退出码 | 0 = 测完；10 = 容量失败（停更高档位）；11 = 服务死了；12 = 基础设施故障；其他 = 失败 |
| 输出目录 | 写到 `$POINT_DIR`，包含 `events.jsonl`、`point_summary.json`、自身 pid 文件 |
| SIGTERM | 收到后停止接新任务、取消在途任务（让子进程清理自己的容器）、写出汇总后退出 |
| flags | 汇总里带 `flags` 列表；以 `suspicious_` 开头的会被矩阵看门狗报 ALERT |
| 冒烟模式 | 支持一个参数让它“第一个数据点完成就停、不预热”，冒烟和正式用同一套代码 |

点的“有效”判据（最少完成数、窗口时长、前后半段一致性、硬上限）属于实验设计，写在设计文档里；本 skill 只要求它们在开跑前定好、由程序自动打标记。

## 5. 部署与源码提交号

- 本地改代码 → 测试 → commit → `deploy.sh`（rsync 到集群并写 `.source-commit`）。
- 工作区和提交不一致时写 `<sha>+dirty`，正式矩阵开跑前必须是干净的提交号。
- 作业运行中不要部署会改变测量语义的代码：正在跑的系列会在下一个点读到新代码。

## 6. 常见调度器坑（Slurm）

- `sbatch` 会把脚本拷到 spool 目录执行，脚本里不要用 `dirname $0` 找同目录文件，写绝对路径。
- `sbatch --export` 按逗号切分值；JSON 之类的参数写进文件，只传路径。
- `srun`/`salloc` 不一定支持 `--parsable`；拿作业号用 `sbatch --parsable`。
- 计算节点上可能不能用 `sacct`（连不上数据库），看门狗用 `squeue` + `scontrol show job` + 自己写的状态文件判断。
- 一个 `srun` step 结束时它启动的后台进程会被回收；常驻服务放在 sbatch 主脚本里。
- 显式写 `--cpus-per-task` 和 `--mem`，否则可能吃到分区默认值（往往很大，更难排上）。
