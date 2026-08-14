# TANGS 服务器交接方案（2026-08-14）

用途：将此文件和 `FIRST_TRY_2026-08-14.md` 直接交给下一位 AI。它说明如何安全地重新连接、核验或重新部署 TANGS；不应把它当作启动新实验的授权。

## 当前交接状态

- 本轮正式矩阵已由用户主动停止；正式监控 heartbeat 也已删除。
- C10 与 C100 已完成，STL-10 在 131,981 / 250,000 step 安全中断并保存可恢复 checkpoint；其余七个单元未启动。
- 完整 raw snapshot 已在本地保存，见
  `gradvax_experiments/results/first_try_2026-08-14/`。其 SHA-256 是
  `74bf4947378377b31f9441de5fafdd83d5dcc0412ab5609fd106e2a9d08372d0`。
- 详情和结果解释边界见 `refine-logs/FIRST_TRY_2026-08-14.md`。
- 本文件写作时，旧主机 `hn01-ssh.gpuhome.cc:30824` 有响应，但已拒绝专用公钥登录。
  这可能是实例重建、`authorized_keys` 变化或访问配置变化；它**不证明** persistent
  volume 中的数据已丢失。不要假设任何远端残留文件仍可用；先核验。

## SSH 连接方式

没有使用常驻本地连接、Python 控制脚本或 SSH config alias。每次检查都是按需执行：

```powershell
ssh.exe -i "C:\Users\林涛\.ssh\codex_tangs_gpuhome_ed25519" `
  -p 30824 root@hn01-ssh.gpuhome.cc
```

- 私钥位于上面的本地路径，已确认存在；`C:\Users\林涛\.ssh\config` 不存在。
- 私钥绝不能上传到 Git、服务器仓库或聊天记录。不要使用或索取旧密码。
- 若重新租用实例，主机名/端口通常会变化；需由用户在新实例上安装此密钥的**公钥**，
  然后替换命令中的 host/port。
- 连通性检查应使用 `BatchMode=yes`，例如：

```powershell
ssh.exe -i "C:\Users\林涛\.ssh\codex_tangs_gpuhome_ed25519" `
  -p <PORT> -o BatchMode=yes -o ConnectTimeout=15 root@<HOST> "hostname; nvidia-smi"
```

## 已知的旧服务器布局（仅供核验）

此前 server persistent volume 的基路径是：

```text
/root/rivermind-data/tangs/
├── venv/                         # Python 3.11 / PyTorch 环境
├── repo/                         # Git clone；实验在 repo/gradvax_experiments/
├── required.log                  # 正式矩阵共享日志
├── required.pid                  # 可能是过期 PID，不能据此判断仍在运行
├── development_runner.sh          # 旧开发阶段 launcher（可能仍在）
├── development.log / development.pid
├── bootstrap.log / ENV_READY      # 环境准备痕迹（可能仍在）
└── tangs_first_try_20260814.tar.gz # 约 319 MB 的远端备份（本地已有已校验副本）
```

先执行以下只读核验；若 volume 未保留，直接走“新部署”流程，**不要删除或覆盖**任何现存日志、结果、checkpoint 或 archive：

```bash
BASE=/root/rivermind-data/tangs
ps -ef | grep -E '[r]un_required|[d]evelopment_runner|[t]rain.py' || true
nvidia-smi
ls -lah "$BASE"
git -C "$BASE/repo" status --short
git -C "$BASE/repo" log -1 --oneline
find "$BASE/repo/gradvax_experiments/results" -maxdepth 1 -mindepth 1 -printf '%f\n' | sort
```

若看到旧的 `required.pid`，必须再用 `ps -p "$(cat required.pid)"` 和进程命令行确认，
不能把 PID 文件本身视作活跃训练。

## 本地 Git 状态与推荐发布方式

本地 Git 根目录是 `D:\Auto-claude-code-research-in-sleep`，当前分支为 `main`；其初始
提交为 `6ce41d29f4bcbfbce4ebb8701752a869a19a5c75`，并且 worktree 有大量旧文件删除、
新 TANGS 文件和文档改动。这些改动尚未形成干净的、可复现的部署提交。

因此下一位 AI 应：

1. 先阅读用户更新后的实验方案、代码 diff 与 `git status`；不要使用 `git add -A`、
   `git reset --hard` 或恢复用户已有的删除。
2. 创建一个新分支（建议 `codex/tangs-<version-or-date>`），只暂存本轮运行必需的
   `gradvax_experiments` 源码、脚本、测试、依赖清单和相关方案文档；不要提交数据、
   checkpoint、`results/` archive、私钥或环境目录。
3. 运行当前代码对应的最小测试/语法检查，提交后 push 到用户的 GitHub `origin`。
4. 在服务器上 clone/fetch 该**精确提交**，再记录 `git rev-parse HEAD` 到运行 manifest。

服务器上的 Git 应维护整个项目 clone：

```bash
BASE=/root/rivermind-data/tangs
mkdir -p "$BASE"
git clone <local-origin-url> "$BASE/repo"       # 新服务器首次部署
cd "$BASE/repo"
git checkout <published-branch-or-commit>
git rev-parse HEAD
REPO="$BASE/repo/gradvax_experiments"
```

若旧 clone 存在，先只读检查 `git status` 与 remote；确认没有用户文件后再 `git fetch`
并 checkout 已发布提交。不要直接把本地 dirty worktree 用 SCP 覆盖到远端。

## 数据、环境与启动前门禁

数据集不在 Git 中。此前的路径约定为：

```text
$REPO/data/cache/cifar-10-batches-py
$REPO/data/cache/cifar-100-python
$REPO/data/cache/stl10_binary
```

新服务器必须用官方下载或从本地缓存传输数据；传输后核验目录结构、文件数/校验和和
可读性。不要把数据集打入 Git 提交。此前已验证的数据包括 CIFAR-10、CIFAR-100 和
STL-10，但新实例仍需重新验证。

旧服务器依赖 `$BASE/venv/bin/python`；系统 `python` 不在 PATH，曾导致一次 launcher
启动失败。因此所有命令必须显式激活 venv：

```bash
source /root/rivermind-data/tangs/venv/bin/activate
cd /root/rivermind-data/tangs/repo/gradvax_experiments
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

在启动任何长实验前，下一位 AI 必须按**用户最新方案**执行当前仓库的 smoke/preflight，
检查：数据目录、冻结参数、Git commit、磁盘可用空间、GPU 可见性、FP32/AMP 设置、
checkpoint/resume 行为和结果输出路径。旧 v4.5 的 `run_required.sh` 命令不自动适用于
用户后续更新的方案，不能未经检查直接复用。

## 历史启动与安全停止参考

历史正式实验的 launcher 是在激活 venv 后用 `nohup` 启动 `bash scripts/run_required.sh`，
stdout/stderr 追加到 `$BASE/required.log`，PID 写到 `$BASE/required.pid`。启动前曾确认
`run_required.sh` 会跳过有 `summary.json` 的已完成单元，并对存在
`checkpoint_last.pt` 的单元传递 `--resume auto`。

旧训练器对主 Python 训练进程的 `SIGTERM` 注册了 handler：写入
`checkpoint_last.pt`，将 `status.json.state` 设为 `interrupted` 后退出。首轮停止时，
STL-10 的 131,981-step checkpoint 已由此机制保存。**在更新代码后，必须重新阅读当前
trainer 的 signal/checkpoint 逻辑，再使用相同停止方式。** 不能杀整个进程组，也不能把
日志中的预期 `KeyboardInterrupt: received signal 15` 误报为训练故障。

## 监控规则

- 当前没有活跃实验，也没有活跃 heartbeat automation。
- 只有在用户明确授权启动某个 run 后，才为该 run 建立监控；默认可用 30 分钟，用户曾
  在正式矩阵阶段要求降为两小时。
- 每轮只核验进程、GPU、step、日志、checkpoint、NaN/Inf/OOM 和磁盘；不要把单次指标
  波动或一次原始结果自行定性为论文结论。
- 若主机临时不可达，报告 `WAIT`，不要把它称为实验失败。
- 只有完整原始结果路径和 config hash 都存在时，才可在 tracker 标为 DONE。

## 可直接交给下一位 AI 的提示词

```text
请先完整阅读：
1. refine-logs/SERVER_HANDOFF_2026-08-14.md
2. refine-logs/FIRST_TRY_2026-08-14.md
3. 用户最新的 EXPERIMENT_PLAN.md、EXPERIMENT_TRACKER.md 与代码改动。

不要启动实验。先检查本地 Git diff，并用交接文档中的专用 SSH 密钥尝试只读连接新/旧服务器。
确认服务器是否保留 persistent volume、Git clone、数据、venv 和 first_try archive；若无，
按交接文档创建干净的新部署。向我报告：连接状态、远端 Git commit/status、数据与环境
就绪情况、以及把当前更新发布到 GitHub 再部署所需的精确文件范围。不要删除任何文件，
不要运行长实验，直到我明确授权。
```
