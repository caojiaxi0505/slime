# Hybrid step-GRPO 修复计划

日期：2026-07-15
适用分支：`feature/cc-ags-swe`
基线 HEAD：`b81a8963`
对应问题报告：[Hybrid run `qwen35_9b_cc_ags_2node_hybrid_c64_t45_bmr3zlex-RANK_0` 问题报告](../notes/2026-07-15-hybrid-run-bmr3zlex-bug-report.md)

## 0. 实施状态（2026-07-15）

已在 `feature/cc-ags-swe`、基线 `b81a8963` 上完成代码实现，尚未提交 commit，也未重启或新提交集群任务。

已完成：

- Stage-1 保存真实 `.harness/trajectory.jsonl`；空文件、坏 JSON、tool ID 或 snapshot 错位均输出 `WARNING` 并禁止 Stage-2；
- 同时安装 `PostToolUse` 与 `PostToolUseFailure` snapshot hook，使成功和失败工具调用都有同 ID 的 workspace 边界；
- `branch_step_t=-1` 表示初始状态；`initial.diff` 作为 baseline 指纹，不会在 `prepare_workspace` 后重复应用；
- workspace 与 transcript 都恢复到目标 edit 之前；prefix-reseed 若没有产生新的 adapter segment，会直接丢弃该 branch；
- `rollout_id` 继续负责 16 个外层调度单元，新增 `loss_group_id` / `loss_weight` 实现 episode 等权与显式 `λ`；
- 训练侧新增 `train/pg_loss_stage1`、`train/pg_loss_stage2`、`train/pg_loss_stage2_weighted`，并禁止 Hybrid 误用 per-token loss；
- 修正 W&B branch 去重、有效 episode/group/token、排队时间、退出码、bundle 路径和 W&B credential 注入；
- 新 launcher 会写不含 credential 的 `run_manifest.json`，并要求 Kubernetes Secret 提供 `WANDB_API_KEY`。

本地 Gate 结果：

- 相关 transcript、分叉、loss、W&B、launcher 测试：`87 passed`；
- CP/DP/microbatch 多进程不变量：`42 passed`；
- 全量 `tests/claudecode_ags`：`172 passed`，另有 2 个失败和 1 个收集错误，均位于本次未改动的普通 generate/SWE-eval 测试；
- Python compile、shell syntax、`git diff --check`：通过。

尚未执行：Gate 2 真实 AGS 单题 smoke 及其后的 rollout/训练 Gate。这些会使用外部 AGS/GPU 资源，应在确认新 W&B Secret、EXP_TAG 和共同 base checkpoint 后单独启动。

## 1. 目标

把当前 Hybrid 实现修回原本要验证的训练语义：

1. Stage-2 同时继承分叉点之前的 workspace 和对话历史；
2. 分叉发生在目标 edit 之前，让 branch 重新决定这次修改；
3. 每次独立尝试先按自己的有效 token 求均值，再用固定权重合并 Stage-1 与 Stage-2；
4. W&B、超时和样本数能够如实反映实际运行；
5. 修复后从共同 base checkpoint 开始一次干净重跑。

本计划不包含算法调参，也不尝试“修好代码后从当前 Hybrid checkpoint 接着训练”。当前 checkpoint 已经接受过错误目标的更新，只保留作问题复盘。

## 2. 修复后的目标行为

对每道题 `p`：

- `L_v(p)`：该题所有有效 Stage-1 episode 的等权平均；
- `L_g(p)`：某个 edit 分叉组内所有有效 branch episode 的等权平均；
- `L_b(p)`：该题所有有效 edit 分叉组的等权平均；
- 设 `ell(e)` 是一条独立 episode 在自身有效 token 上的平均损失，则完整公式是：

```text
L_v(p) = mean_{e in Stage-1(p)} ell(e)

L_g(p) = mean_{e in BranchGroup(p,g)} ell(e)

L_b(p) = mean_{g in ActiveBranchGroups(p)} L_g(p)

L = (1 / 16) × Σ_p [L_v(p) + λ × L_b(p)]
```

所以 `L_g(p)` 是中间量：先保证同一个 edit 内的 branch 等权，再由 `L_b(p)` 保证不同 edit group 等权。它不会作为第三项重复加进总损失。

建议把 `λ` 做成显式配置，默认值为 `1.0`，并记录到 W&B config。它表示 Stage-2 相对一份 Stage-1 损失的名义权重。正式长跑前再通过短跑确认是否保留 `1.0`；不能继续让 branch 数量或长度隐式决定权重。

具体权重为：

```text
每条有效 Stage-1 episode：1 / 有效 Stage-1 episode 数
分叉组 g 中每条有效 branch：λ / (有效分叉组数 × 组内有效 branch 数)
```

若某一阶段没有有效样本，该阶段贡献为 0，不改变一批 16 道题的总分母。

## 3. 已锁定的设计决定

### 3.1 保留现有 `rollout_id` 作为调度身份

当前系统用 `rollout_id` 表示一次外层 rollout 执行，并据此维持 RBS/GBS=16。compact rollout 的多个 segment 也依赖它通过调度和校验。

因此，不能简单地给每条 Stage-1 trial 和 Stage-2 branch 分配新的 `rollout_id`。这样可能修正损失分母，却会破坏动态批调度和 compact rollout 语义。

修复方案是新增两个独立字段：

- `loss_group_id`：一次独立 episode 的训练身份；同一 episode 的 compact segments 共享该值，不同 trial/branch 必须不同；
- `loss_weight`：该 episode 在最终目标中的显式权重。

未设置这两个字段时，自动退回当前行为：`loss_group_id=rollout_id`、`loss_weight=1.0`。这样朴素 GRPO 和其他任务不需要改配置，结果应与修复前逐数值一致。

### 3.2 transcript 无效时 fail closed

若 transcript 缺失、为空、JSON 损坏，或不能与 tool snapshot 对齐：

- 保留该次 Stage-1 episode 参与训练；
- 将它标记为 Stage-2 不可分叉；
- 不允许 fallback 为“新 agent 从原始 prompt 重新开始”。
- 至少输出一条 `WARNING`，包含 instance、trial、bundle 路径和失败原因；同时累计到 W&B 的 transcript 失败计数。

宁可少生成 branch，也不能悄悄改变 Stage-2 的含义。

### 3.3 用逻辑 `-1` 表示第一处 edit 之前

实际 bundle 的 snapshot 文件从 `step_0001.diff` 开始，并没有物理的 `snapshot -1`。当前代码还把 workspace 的 `step_t=-1` 解释为 `final.diff`，把 transcript 的负数位置解释为“保留完整对话”。所以现有 `-1` 不能直接使用，否则拿到的是终局状态，不是初始状态。

修复后重新定义一个明确的逻辑初始状态：

- `branch_step_t=-1` 表示 agent 尚未执行任何工具动作；
- workspace 由 `prepare_workspace` 恢复到 Stage-1 的初始状态，并用显式保存的 `initial.diff` 做严格比对；通常该 diff 为空。若非空，不会重复应用；重建后续累计状态时会先反向撤销该 baseline diff，再应用目标累计 diff；
- transcript prefix 截在第 0 个 assistant/tool 动作之前，只保留启动这次任务所需的初始 user message，不能返回完整 transcript；若 Claude Code 的输出流不回显这条初始 user message，则从 bundle 中的 `agent_prompt` 按固定格式重建；
- `final.diff` 只能通过显式的 final-state API 读取，不再借用 `-1` 这个值。

为方便校验，新 bundle 在 Stage-1 agent 启动前保存 `initial.diff` 或等价的 baseline fingerprint。旧 bundle 若 task metadata 完整，也可尝试用 `prepare_workspace` 重建 baseline，但必须通过 diff 校验后才能分叉。

这样目标 edit 为 `i=0` 时可以从逻辑 `-1` 分叉，不需要跳过第一处 edit。

### 3.4 正式实验必须从共同 base 重启

修复只保证未来更新正确，不能撤销当前 checkpoint 已发生的错误更新。新的 Hybrid 对照必须：

- 使用共同 base `Qwen3.5-9B_torch_dist`；
- 使用新的 EXP_TAG 和输出目录；
- 固定并记录代码 commit、数据版本、`λ` 和 launcher 配置；
- 不覆盖或续训当前 run。

## 4. 实施顺序

修复按以下依赖顺序推进。每一阶段通过 Gate 后，再进入下一阶段。

### 阶段 A：补齐 transcript 捕获与校验

涉及文件：

- `slime/agent/harness/common.py`
- `examples/claudecode_ags/step_reconstruct/session_capture.py`
- `examples/claudecode_ags/step_reconstruct/live_runners.py`
- `examples/claudecode_ags/agent_runtime.py`
- `examples/claudecode_ags/step_reconstruct/workspace_rebuild.py`

实施内容：

1. 在 AGS sandbox 关闭前，把 `<workdir>/.harness/trajectory.jsonl` 拉回对应 bundle。
2. 原子写入 `transcript.jsonl`；不再创建 0 字节占位文件。
3. 增加 bundle 校验器，至少检查：
   - 文件非空且每行可解析；
   - `tool_use` 与 `tool_result` 能按 ID 配对；
   - transcript 中的工具步与 snapshot 序号能对齐；
   - bundle 记录 transcript 字节数、事件数和最后一个已对齐工具 ID。
4. 用结构化 JSON 事件截取 prefix，不再依赖字符串计数。
5. prefix 必须通过 stdin 传给 `claude -p --input-format stream-json`；恢复模式下不再同时附加原始 positional prompt，避免题目被输入两次。
6. 保存 Stage-2 恢复调用的输出，便于审计它是否真的接续了 prefix。
7. transcript 校验失败时只取消 Stage-2 资格，并记录明确原因。

阶段 Gate：

- 一条真实 AGS Stage-1 smoke 生成非空、可解析的 transcript；
- prefix 中最后一个工具结果与选择的 pre-edit snapshot 一致；
- Stage-2 确实走 prefix resume 路径；
- 人为制造空文件、坏 JSON 和错位 tool ID 时，均不会启动 fresh agent fallback。

### 阶段 B：修正 pre-edit 分叉语义

涉及文件：

- `examples/claudecode_ags/step_reconstruct/edit_ppl.py`
- `examples/claudecode_ags/step_reconstruct/selection.py`
- `examples/claudecode_ags/step_reconstruct/workspace_rebuild.py`
- `examples/claudecode_ags/step_reconstruct/live_runners.py`

实施内容：

1. 明确区分两个序号：
   - `edit_step_i`：workspace diff 发生变化的目标 edit；
   - `branch_step_t=i-1`：真正用于重建的 edit 前状态。
2. 在 Stage-1 agent 启动前保存 `initial.diff` 或 baseline fingerprint，作为逻辑 `branch_step_t=-1` 的校验依据。
3. workspace 使用 `branch_step_t` 重建；conversation prefix 也停在同一位置。
4. 重新定义 `-1`：workspace 使用 baseline，transcript 截在第一个工具动作之前；不能再分别返回 `final.diff` 和完整 transcript。
5. final state 改用显式 API，避免与 initial state 共用 `-1`。
6. `step_group_key` 标识“要重试的目标 edit”，使用 `edit_step_i`，不能因为恢复点是 `i-1` 而把两个不同 edit 混为一组。
7. branch metadata 同时记录 `edit_step_i`、`branch_step_t`、`source_trial_idx` 和 `branch_uid`。

阶段 Gate：

- 构造一个 edit 前后内容明确不同的仓库，重建结果中不得包含目标 patch；
- transcript prefix 不得包含目标 edit 的 assistant/tool 事件；
- K 条 branch 从完全相同的 pre-edit workspace 和 prefix 开始；
- 第一处 edit 可以从已校验的逻辑 `-1` 状态分叉；该 workspace 是 baseline，prefix 只包含初始 user message；
- 多个 edit 点的 `step_group_key` 不碰撞。

### 阶段 C：分离调度身份与损失身份

这是风险最高的阶段，先完成纯数值测试，再接入真实 Hybrid。

涉及文件：

- `slime/utils/types.py`
- `slime/ray/rollout.py`
- `slime/backends/megatron_utils/cp_utils.py`
- `slime/backends/megatron_utils/loss.py`
- `examples/claudecode_ags/step_reconstruct/hybrid_generate.py`
- `examples/claudecode_ags/step_reconstruct/step_grpo_advantage.py`

实施内容：

1. 给 `Sample` 增加可选的 `loss_group_id` 和 `loss_weight`。
2. Hybrid 为每条独立 Stage-1 trial、每条独立 Stage-2 branch 生成唯一 `loss_group_id`；compact segments 自动继承同一个值。
3. 保留同一道题下共享的外层 `rollout_id`，DP schedule 仍只按它调度 16 个 rollout 单元。
4. rollout filter 完成后，再按实际留下的 episode 和 edit group 计算 `loss_weight`，避免被过滤样本仍占分母。
5. 在 train data 转换阶段：
   - 按 `loss_group_id` 汇总 episode 的有效 token 数；
   - 生成 `loss_group_mask_sums` 和 `loss_weights`；
   - 没有新字段时兼容旧的 `rollout_mask_sums` 行为。
6. 让字段完整经过 DP split、microbatch、tensor 化和 actor 训练路径。
7. 扩展 sample-mean reducer：先算每个 episode 的 token 均值，再乘 `loss_weight`。
8. 同步处理 policy gradient、TIS 重建、entropy/KL 等使用同一 sample-mean 口径的训练项和诊断项，避免“训练按新口径、指标仍按旧口径”。
9. 最终仍除以固定 GBS=16；不能再额外除以 episode 数。
10. 新增 `STEP_GRPO_BRANCH_LOSS_WEIGHT`，默认 `1.0`，并把有效值写入 W&B config 和 run manifest。

核心数值测试：

1. **旧行为兼容：**不提供新字段时，朴素 GRPO 和既有 compact rollout 测试逐数值不变。
2. **episode 等权：**episode 长度不同但权重相同时，对总损失的名义贡献相同。
3. **segment 不变性：**把一个 episode 拆成 1、2 或更多 compact segments，损失和梯度不变。
4. **branch 数量不变性：**K 从 2 改成 8，在分支均值相同且 `λ` 固定时，Stage-2 总权重不变。
5. **branch 长度不变性：**给某条 branch 增加 mask 外 token，或等价拆长，不改变阶段权重。
6. **分叉组等权：**一个 edit 有 2 条有效 branch、另一个有 8 条时，两个 edit group 仍各占 `L_b` 的一半。
7. **CP/DP 不变性：**改变 context parallel、data parallel 和 microbatch 切分后，损失与梯度在容差内一致。
8. **调度不变：**一个训练 step 仍恰好有 16 个外层 `rollout_id`，但 `loss_group_id` 数等于实际独立 episode 数。

阶段 Gate：

- 上述测试全部通过；
- 用同一份 rollout dump 离线重算公式，与训练侧数值一致；
- `λ=0` 时 Hybrid Stage-1 的 reducer 与朴素 GRPO reducer 数值一致；
- 修改 branch 数或平均长度不再改变名义 Stage-1/Stage-2 权重。

### 阶段 D：修正指标、目录和安全配置

涉及文件：

- `examples/claudecode_ags/wandb_metrics.py`
- `examples/claudecode_ags/step_reconstruct/_common.py`
- Hybrid/GRPO/eval 的 launcher 与 PyTorchJob template

实施内容：

1. W&B 的 branch episode key 优先使用 `branch_uid`；fallback key 必须包含 edit step 和 branch index。
2. 新增或明确记录：
   - Stage-1/Stage-2 实际 episode 数、有效 episode 数；
   - 候选 edit 数、选中数和各类跳过原因；
   - transcript 缺失、损坏、对齐失败数；
   - 有效 edit group 数和每组 branch 数；
   - `L_v`、`L_b`、`λ`、两阶段有效 token 数和名义 loss 权重；
   - `agent_exit_code`、不含排队的 `agent_elapsed_sec`、`agent_queue_wait_sec`。
   - transcript 缺失或校验失败时必须同时写 `WARNING`，不能只有 W&B 计数而没有本地日志。
3. 顶层 `outcome/resolved_rate` 继续明确表示 Stage-1；Stage-2 单独展示，避免混用。
4. 复核当前 worktree 已有的“先拿并发槽、再开 guard/计时”修复，并补回归测试；不重复改写已正确部分。
5. 若显式设置 `STEP_GRPO_BUNDLE_DIR`，直接使用该路径，不再追加第二层 `step_reconstruct_bundles`。读取旧 run 时可保留 legacy fallback，但新产物只写标准路径。
6. launcher 不再把 W&B key 放进命令行或模板明文：
   - 通过 Kubernetes Secret 注入 `WANDB_API_KEY`；
   - 删除 `--wandb-key ...`；
   - 启动日志只记录“credential 已配置”，不记录值。
7. 已出现在旧 `run.log` 的 W&B credential 需要由账号持有人轮换；这是外部操作，不由代码自动执行。

阶段 Gate：

- W&B Stage-2 episode 数与 rollout dump 中唯一 `branch_uid` 数完全一致；
- 每条产生 branch 的样本都有退出码、agent 用时和排队用时；
- 新 bundle 路径只出现一层目录；
- 扫描新日志、Job spec 和进程参数，均找不到 credential 明文。

### 阶段 E：补端到端测试

新增或扩展：

- `tests/claudecode_ags/test_step_reconstruct_live_runners.py`
- `tests/claudecode_ags/test_step_reconstruct_rebuild.py`
- `tests/claudecode_ags/test_step_reconstruct_hybrid_orchestration.py`
- 新增 loss-group/reducer 专项测试文件

端到端最小链路必须覆盖：

```text
真实 Stage-1
  → 捕获 transcript 与 snapshots
  → 选择高 edit-PPL edit
  → 重建 pre-edit workspace/prefix
  → 生成至少 2 条 branch
  → 计算 advantage 与显式 loss weight
  → 转为 train data
  → 核对最终损失
```

不能只靠手写 transcript 或 mock 掉 resume 调用。单元测试可以使用 mock，但至少有一条 AGS smoke 要验证真实文件和真实 Claude Code 输入格式。

阶段 Gate：

- 正常链路全通；
- 空 transcript、损坏 transcript、首个 edit 的逻辑 `-1`、branch 部分失败和 compact segment 五类情况均有覆盖；
- 任何异常都不会静默退回错误语义。

## 5. 验证与重跑阶梯

不要修完后直接提交 16 卡长跑。按以下顺序逐级放大：

### Gate 1：CPU/本地纯逻辑测试

- 跑新增测试和所有既有 step-reconstruct/compact rollout 测试；
- 重点验证损失、梯度和切分不变量；
- 不连接 AGS，不消耗训练 GPU。

### Gate 2：单题真实 AGS smoke

- 1 道题、1 条 Stage-1，分别覆盖第一处 edit 和一个非首 edit，每个位置生成 2 条 branch；
- 人工检查 transcript、snapshot、pre-edit diff 和 resume 输入；
- 验证 branch 的首个新动作发生在恢复 prefix 之后。

### Gate 3：小规模 rollout-only

- 2–4 道题，K=2；
- 不做参数更新；
- 核对 sample 身份、group、权重、W&B 计数和离线重算。

### Gate 4：Stage-1-only 训练消融

- 从共同 base 启动 Hybrid runner；
- 禁用 Stage-2 或设 `λ=0`；
- 运行 1–2 step；
- 验证其数学损失口径与朴素 GRPO 一致。

这里不要求两边随机轨迹逐条相同；验收的是相同输入样本上的 reducer 和梯度一致，以及运行指标无系统性异常。

### Gate 5：短 Hybrid 训练

- 先做 1-node 1–2 step，再做 2-node 2–3 step；
- 使用正式 K=8、RBS=16 和候选 `λ`；
- 同时记录 `L_v`、`L_b`、stage 权重、active episode/group 数和 grad norm；
- 2-node 短跑通过后才能恢复完整配置。

### Gate 6：正式 2-node 重跑

建议新 tag：

```text
qwen35_9b_cc_ags_2node_hybrid_c64_t45_fix_v1
```

固定以下公平条件：

- 16 GPU、RBS=16、每题 Stage-1 K=8；
- agent 45 分钟、真并发 64；
- 与朴素 GRPO 相同的数据版本、题目顺序规则和 base checkpoint；
- 新 EXP_TAG、新输出目录、新 W&B run；
- 记录 commit SHA、镜像、tool、ALB、seed 和全部 Hybrid 配置。

正式比较仍以 SWE-bench Verified 484 pass@4 为最终依据；训练期 resolved rate 只作健康指标，`grad_norm` 不作为算法胜负标准。

## 6. 全局验收标准

只有以下条件全部满足，才把新 run 标记为“可用于 Hybrid vs GRPO 算法比较”：

- 每条实际生成的 branch 都有非空、可解析且已对齐的 transcript；
- 不存在 fresh-agent fallback；
- workspace 和 conversation 都停在目标 edit 之前；
- 第一处 edit 能从已校验的逻辑 `-1` 状态分叉，且 `-1` 不会读取 `final.diff` 或完整末尾 transcript；
- 每个独立 trial/branch 有唯一 `loss_group_id`，只有其 compact segments 共享；
- 一个 step 仍只有 16 个调度 `rollout_id`；
- 训练损失与离线公式在数值容差内一致；
- 固定 `λ` 时，改变 branch 数量或长度不改变阶段名义权重；
- W&B branch episode 数等于唯一 `branch_uid` 数；
- 超时率只以 `exit_code=-1` 和不含排队的 agent 用时统计；
- 日志和 Job spec 不含 credential 明文；
- 无 NaN、无整步异常丢样本，既有朴素 GRPO/compact rollout 回归测试通过；
- 从共同 base checkpoint 开始，未续接旧 Hybrid checkpoint。

## 7. 必须停止放大的条件

任一项出现时，停止进入更大规模 Gate，先回到对应阶段修复：

- 发现任意 branch 使用空 transcript 或原始 prompt 重新开始；
- 重建 workspace 已包含目标 edit；
- 逻辑 `-1` 被解析为 final workspace 或完整末尾 transcript；
- loss group 数与有效独立 episode 数不一致；
- 同一数据在不同 segment/DP/CP 切分下损失明显变化；
- W&B 数量与 dump 对不上；
- Stage-2 名义权重随 branch token 占比漂移；
- 新日志再次出现 W&B credential；
- 训练出现 NaN、持续异常 sample rejection 或权重同步错误。

resolved rate 单步波动和 `grad_norm` 较小本身不是停止条件；先判断实现不变量和训练健康项是否异常。

## 8. 建议提交拆分

为便于审查和回滚，建议拆成五个独立提交：

1. `capture and validate hybrid transcripts`
2. `branch from aligned pre-edit state`
3. `separate rollout scheduling from loss episode weighting`
4. `fix hybrid metrics, bundle path, and timeout observability`
5. `remove plaintext wandb credentials from launchers`

每个提交都应带对应测试。第 3 个提交只要回归未完全通过，就不能与 launcher 改动一起提交长跑。

## 9. 交付物

修复完成时应具备：

1. 上述代码与测试；
2. 一份单题 AGS smoke 的 bundle 审计结果；
3. 一份 loss 公式的离线重算报告；
4. Stage-1-only 和短 Hybrid run 的 W&B 链接及 Gate 结论；
5. 新正式 run 的完整配置清单；
6. 旧 run 的明确标签：仅供问题复盘，不用于算法结论。

## 10. 执行优先级

优先级保持为：

```text
transcript 接续
  → pre-edit 语义
  → loss 身份与显式权重
  → 指标/目录/凭据
  → 小规模验证
  → 干净长跑
```

前 3 项决定“训练的到底是不是目标算法”；后 3 项决定“结果是否可核验、是否值得投入长跑资源”。
