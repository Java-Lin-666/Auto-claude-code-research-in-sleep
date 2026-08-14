# TANGS v4.6 交接方案（2026-08-14）

这份文件是新聊天窗口的首要入口。接手 AI 应先完整阅读本文，再读取：

1. `refine-logs/FINAL_PROPOSAL.md`
2. `refine-logs/EXPERIMENT_PLAN.md`
3. `refine-logs/EXPERIMENT_TRACKER.md`
4. `gradvax_experiments/reported_results_from_papers.md`
5. `refine-logs/FIRST_TRY_2026-08-14.md`

用户目标：尽最大努力提升 TANGS 在长尾半监督学习上的效果，形成可信的
CCF-C 论文证据。用户只有单卡；实验必须串行、可恢复，并在运行期间帮助
用户监控。未经明确授权，不连接服务器、不启动正式实验。

## 1. 当前结论

v4.5 不能继续作为当前方法。

- C10 v4.5 full：bACC 79.59%，GM 78.66%，Head 93.87%，Tail 69.93%。
- C100 v4.5 full：bACC 36.53%，GM 7.55%，Head 71.12%，Tail 5.61%，
  Worst 0%，共有 15 个零准确率类别。
- 文献 FixMatch C100 对照为 37.6 ± 0.48 bACC，但 v4.5 没有对应的本地
  250k FixMatch 配对，因此不能仅凭文献值判断全部损失来自手术。
- v4.5 的 C10 50k 开发筛选选择了 `tau=inf`。相对本地 FixMatch：
  `tau=2/5/10/inf` 的 bACC 差分别为 -12.12/-10.37/-9.68/+1.77 pp。
- `tau=inf` 使 v4.5 full 与 PCGrad/no-cap 完全相同，旧方法没有独立的
  norm-aware 组件证据。
- C100 共 250k 步，eligible 248,067（99.23%），anchor update 63,547
  （25.42%），eligible 中 conflict 233,163（93.99%），invalid anchor update
  只有 2 次。不要把根因简单归为 `tau=5`、锚点不可用或支持数不足。
- sampled deep diagnostics 中，accepted predicted-head 样本约 96–97% 确实
  是 true-head。旧方法修改它们的完整分类器梯度，缺乏尾部特异性。
- `within_tail_cancellation_ratio = ||sum g_c|| / sum ||g_c||`，值越高表示
  抵消越少。此前“高值意味着抵消严重”的解释是错误的。

历史原始证据：

- `gradvax_experiments/results/first_try_2026-08-14/`
- archive：`tangs_first_try_20260814.tar.gz`
- SHA-256：`74bf4947378377b31f9441de5fafdd83d5dcc0412ab5609fd106e2a9d08372d0`

不要恢复旧 STL v4.5 任务（停在 131,981/250,000）。

## 2. 当前方法：Classwise Tail-Row TANGS v4.6

旧 CLI 名 `tangs` 永久表示 v4.5，保证旧 artifact/checkpoint 可审计。新方法
使用显式名称：

- `tailrow-observer`：只测量，绝不修改梯度；
- `tailrow-group`：仅限制尾部输出行，但使用一个 group anchor；
- `tailrow-classwise`：每类 anchor，无修正预算；
- `tangs-v46`：每类 anchor + 修正预算，当前 full；
- `oracle-tangs-v46`：以后机制实验使用，当前不运行。

设最终线性分类器为 `z = Wf + b`。对每个尾类 c：

- `u_c`：accepted predicted-head unsupervised loss 对 `(W_c,b_c)` 的精确贡献；
- `s_c`：监督 batch 中真实类别 c 样本对 `(W_c,b_c)` 的精确 self-row 贡献；
- `a_c = EMA(s_c)`；
- `m_c = EMA(||s_c||)`。

若 `<u_c,a_c> < 0`：

```text
q_c = <u_c,a_c> / (||a_c||^2 + eps) * a_c
lambda_c = min(1, rho * m_c / (||q_c|| + eps))
u'_c = u_c - lambda_c * q_c
```

full 使用 `rho=1`。只替换 tail classifier rows；head/medium rows 和 backbone
梯度保持 base FixMatch 更新。`beta=0.99`、warm-up=2500、eps=1e-12、FP32。
每类出现至少一个监督样本即可更新自己的 anchor。

解析梯度保留了实际 loss `-log(softmax + 1e-8)` 中 eps 引起的导数因子，
并保留原始 batch denominator、mask、两张 strong view 和 lambda_u。

主要实现：

- `gradvax_experiments/tangs/surgery.py`
- `gradvax_experiments/tangs/trainer.py`
- `gradvax_experiments/tangs/config.py`
- `gradvax_experiments/train.py`

## 3. 开发协议

v4.6 允许 P-C100-100 使用 development mode。验证集来自 CIFAR-100 训练集：

- 对每类，从 source unlabeled prefix 之外取 50 张；
- 共 5,000 张，类别平衡；
- 与 active labeled/unlabeled loader 均不重叠；
- 不删除训练样本；
- 不读取官方 test set。

实现位置：`gradvax_experiments/tangs/data.py`。

## 4. 已完成验证

本机已有专用环境，无需安装：

```text
C:\lintao\envs\fixmatch\python.exe
Python 3.10.19
torch 2.5.1
torchvision 0.20.1
pytest 8.2.2
CUDA build 11.8
GPU NVIDIA GeForce RTX 4060 Laptop GPU
```

执行：

```powershell
Set-Location D:\Auto-claude-code-research-in-sleep\gradvax_experiments
& C:\lintao\envs\fixmatch\python.exe -m pytest -q
```

结果：`53 passed in 5.24s`。包括解析梯度/autograd parity、只改 tail row、
修正预算、observer no-op、数据划分、协议、门禁和脚本测试。

Python `py_compile` 通过；三个 Bash 脚本使用
`C:\msys64\usr\bin\bash.exe -n` 检查通过。

没有启动任何 v4.6 smoke、50k 或 250k 训练。

## 5. 接手后的第一项工作：服务器 CUDA smoke

用户已确认 CUDA smoke 可以在服务器做。意义是验证目标执行环境中的
驱动、CUDA、DataLoader、多进程、数据路径、显存、checkpoint/resume 和完整
训练链路，不用于判断精度。数据已缓存时，全部 smoke 预计 5–15 分钟。

连接服务器前必须由用户提供或确认连接方式与授权。不要寻找、复制或推测
新的凭据。已知旧记录提示服务器 venv 曾位于：
`/root/rivermind-data/tangs/venv`，但接手 AI 必须先只读确认实际路径。

服务器顺序：

1. 检查 repo、分支/dirty state、Python、torch、torchvision、CUDA、GPU、数据缓存。
2. 在服务器项目环境执行完整 `python -m pytest -q`；必须全部 PASS。
3. 跑 5-step `fixmatch --tailrow-observer` smoke。
4. 跑 5-step `tangs-v46 --tangs-correction-rho 1` smoke。
5. 做一次受控中断/恢复验证，确认 checkpoint、controller state、JSONL rollback
   和 status/event 记录正常。
6. 审查两个 smoke 的 resolved config、split manifest、diagnostics、summary、显存
   和 finite 值；observer 必须没有非零梯度修改。
7. 只有全部通过才启动 50k queue。

当前没有单独的 `run_v46_smoke.sh`。接手 AI 可以先补一个可恢复的脚本，或使用
`train.py --mode smoke` 的直接命令；必须使用新的 run ID，不能覆盖历史目录。

建议 smoke 命令骨架：

```bash
python -u train.py \
  --protocol P-C100-100 \
  --method fixmatch \
  --mode smoke \
  --manual-seed 0 \
  --data-root ./data/cache \
  --output-root ./results \
  --run-id v46-smoke-c100-fixmatch-observer-seed0 \
  --tailrow-observer \
  --checkpoint-every 1
```

```bash
python -u train.py \
  --protocol P-C100-100 \
  --method tangs-v46 \
  --mode smoke \
  --manual-seed 0 \
  --data-root ./data/cache \
  --output-root ./results \
  --run-id v46-smoke-c100-tangs-rho1-seed0 \
  --tangs-correction-rho 1 \
  --checkpoint-every 1
```

注意：默认 smoke 是 5 步；不要用 `--max-steps` 改写 locked smoke budget。

## 6. smoke 之后：单卡 50k 开发队列

入口：

```bash
cd gradvax_experiments
CUDA_VISIBLE_DEVICES=0 bash scripts/run_v46_development.sh
```

固定顺序：

1. `v46-dev-c100-100-fixmatch-observer-seed0`
2. `v46-dev-c100-100-legacy-pcgrad-seed0`
3. `v46-dev-c100-100-tailrow-group-seed0`
4. `v46-dev-c100-100-tailrow-classwise-seed0`
5. `v46-dev-c100-100-tangs-rho1-seed0`

脚本单卡串行执行；有 `summary.json` 则跳过，有 `checkpoint_last.pt` 则
`--resume auto`。不要并发启动第二份脚本。按旧运行速度估计总计约 5 小时，
但应以服务器 smoke/首个 50k 的实测速度更新 ETA。

监控时优先读取：

- `results/<run_id>/status.json`
- `results/<run_id>/events.jsonl`
- `results/<run_id>/train_metrics.jsonl`
- `nvidia-smi`

不要频繁重启，不要因为暂时无输出就判定失败。若用户要求持续盯实验，应提供
简短进度、当前 step、速度、ETA、GPU/显存、异常；遇到 SIGTERM/错误要先保留
artifact 并判断 checkpoint 是否有效。

## 7. 自动门禁

队列末尾运行 `scripts/analyze_v46_development.py`，输出：

```text
results/v46-development-selection.json
```

full `tangs-v46` 必须同时满足：

- versus local FixMatch：bACC >= +1.0 pp；
- versus local FixMatch：Tail >= +2.0 pp；
- versus local FixMatch：Head >= -2.0 pp；
- versus local FixMatch：GM >= 0.0 pp；
- dead class count 不高于 FixMatch；
- versus legacy v4.5：bACC >= +0.5 pp；
- versus unbounded classwise：bACC >= +0.25 pp。

只有全部 PASS 才设置 `allow_confirmatory_250k=true`。不得根据结果放宽阈值。

若最后一条失败：norm-aware correction budget 没有独立价值，必须删除/重构
norm-aware claim，不能把 unbounded classwise 偷换成 v4.6 full。

若 classwise 明显好而 rho=1 不好：可以先报告 STOP，再开启一个新版本的
有限 rho 诊断，不能事后篡改 v4.6 gate。

若所有 tail-row 版本都不超过 FixMatch：停止 250k，分析 representation bias、
pseudo-label coverage 和 base-method问题；不要靠多跑种子掩盖负结果。

## 8. PASS 后的唯一授权阶段

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_required_v46.sh
```

此脚本会先检查 machine-readable PASS，只串行运行：

1. 250k local FixMatch + observer；
2. 250k frozen `tangs-v46`, rho=1。

预计单卡总计约 10 小时。先审查这个核心配对，再决定 C10、STL、oracle、更多
ablations 和 cost experiments。文献 FixMatch 37.6 是上下文，本地 paired
difference 才是 v4.6 的主要因果比较。

## 9. 明确禁区

- 不运行或恢复旧 `scripts/run_required.sh`；它已默认 BLOCK。
- 不恢复旧 STL v4.5 checkpoint。
- 不把 development 或 smoke 数值写进论文主表。
- 不把论文 reported result 描述为本地复现。
- 不在看到 gate 结果后调整阈值。
- 不删除或覆盖失败/负结果 artifact。
- 不使用官方 C100 test set 调参。
- 不并行跑单卡队列。
- 不重置/清理当前 dirty worktree；仓库中存在大量用户原有修改和未跟踪文件。

## 10. 文档同步规则

任何新的测试、smoke、run 状态或 gate 结果，至少同步：

- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_TRACKER.md`
- `gradvax_experiments/reported_results_from_papers.md`（只更新证据状态，不能
  篡改文献数值）
- 本交接文件的“当前状态”部分

每个 DONE 必须包含 artifact 路径和 config hash。负结果同样记录。

## 11. 当前主要文件

- method/controller：`gradvax_experiments/tangs/surgery.py`
- trainer integration：`gradvax_experiments/tangs/trainer.py`
- protocol/config：`gradvax_experiments/tangs/config.py`
- C100 dev data：`gradvax_experiments/tangs/data.py`
- CLI：`gradvax_experiments/train.py`
- 50k queue：`gradvax_experiments/scripts/run_v46_development.sh`
- dev analyzer：`gradvax_experiments/scripts/analyze_v46_development.py`
- gated 250k pair：`gradvax_experiments/scripts/run_required_v46.sh`
- tests：`gradvax_experiments/tests/test_tailrow_v46.py`、
  `test_v46_development.py` 及其余 tests。

## 12. 给接手 AI 的首条建议

不要继续无依据地调 tau/beta/warm-up。先完成服务器环境审计与两个 5-step
v4.6 smoke；把原始路径、配置哈希和检查结论写回 tracker。通过后运行固定的
五项 50k 单卡队列并监控。第一个真正决定研究方向的信息是 paired C100
development gate，而不是继续讨论参数直觉。
