# 失败状态驱动的 Turn-Level Teacher Imitation

本文只描述方法：为什么做、训练数据如何产生、模型学习什么。Claude Code resume、Adapter、token 对齐和 JSONL 落盘等内容属于工程实现，不在本文展开。

# 背景

想让 LLM 在 Claude Code 这类长程 Harness 中完成 SWE 任务，只依赖终局强化学习或普通 SFT 都存在明显限制：

1. **终局奖励的信用分配太粗**：一条 SWE 轨迹通常包含几十轮 Read、Grep、Edit 和 Bash，但 GRPO 最后只得到是否 resolved 的单个奖励。同一条轨迹内的 assistant token 共用同一个 advantage，无法区分真正决定成败的动作与大量中间探索；当同组样本奖励相同时，甚至没有可用的组内学习信号。

2. **普通 SFT 与学生实际遇到的状态不一致**：离线 SFT 通常学习强模型从干净起点生成的成功轨迹，但训练后的学生会走进自己的错误状态，例如读错文件、形成不完整 patch、执行错误测试，或在已经修改代码后不知道下一步做什么。专家成功轨迹很少覆盖这些状态，模型即使会模仿标准解法，也不一定会从自己的错误中恢复。

3. **让强模型完整重做每道题成本过高**：如果对每个学生失败状态都让强模型从该状态一直跑到任务结束，需要重复创建沙箱、执行大量工具并运行终局评测。训练成本会随分叉点数量快速增长，难以覆盖足够多的学生状态。

# 解决方案

针对上述问题，采用**失败状态驱动的 Turn-Level Teacher Imitation**：先让学生模型完成整题探索，再把强模型放到学生失败轨迹中的真实中间状态，让强模型给出短程 continuation；学生只模仿这些 continuation 中的 assistant 输出。

核心变化是：**监督数据不再只来自强模型自己的成功轨迹，而是来自学生实际访问过、但没有解决任务的状态。** 强模型负责回答“在这个具体状态下，接下来应该怎么做”。

### 核心思路

| | 朴素 GRPO | Hybrid Turn-Level GRPO | Turn-Level Teacher Imitation |
| :-: | :-: | :-: | :-: |
| 状态从哪里来 | 学生完整轨迹 | 学生失败轨迹的分叉点 | 学生失败轨迹的分叉点 |
| 后续行为由谁生成 | 学生 | 学生的多条 continuation | 强模型的短程 continuation |
| 学习信号 | 整题终局 reward | 同起点 continuation 的终局 reward 差异 | 强模型 assistant token 的交叉熵 |
| 信用粒度 | 整条 episode | 分叉点级 | 分叉点后的具体动作级 |
| 是否要求续跑到终局 | 是 | 是 | 否，当前最多 2 个 assistant turn |

该方法与 DAgger 的出发点相近：先用当前学生策略产生状态，再向专家查询这些状态下的正确行为。区别在于，这里只查询未解决轨迹中的可恢复工具决策点，并让 Teacher 真实执行工具，从而保留 SWE Harness 的环境反馈。

# 方法流程

对每一道 SWE 题执行以下步骤。

### 1. 学生完成整题探索

使用当前学生模型采样 $K$ 条完整 Claude Code 轨迹，当前设置为 $K=8$。每条轨迹都运行到 Agent 结束或达到时间上限，并用 SWE 评测得到 `resolved=True/False`。

这些轨迹有两个作用：

- 判断学生当前是否能够解决这道题；
- 保存学生真实访问过的模型上下文、代码状态和工具执行状态。

当前 `sft_only` 实验不直接训练这些学生轨迹。

### 2. 从未解决轨迹中提取学习状态

只处理 `resolved=False` 的学生轨迹。对每条失败轨迹，收集所有可以准确恢复的工具决策位置，包括 Read、Grep、Edit、Write、Bash 等。

一个候选位置表示：恢复到学生生成该次工具调用之前，让 Teacher 替代学生决定下一步。纯文本结束轮、缺少模型 checkpoint 或无法恢复代码状态的位置不产生训练数据。基础设施失败产生的占位轨迹同样不会被当作有效失败行为学习。

当前方法选择所有有效工具决策位置，而不是只选择 Edit，也不按 edit-PPL 排序截断。因此一条较长的失败轨迹可以产生多条局部监督数据。

### 3. Teacher 从同一状态短程续跑

对每个候选位置，恢复该位置之前的完整状态，然后让强模型从这里继续操作。当前 Teacher 最多生成 2 个 assistant turn：

1. Teacher 读取当前状态，输出文本、推理或工具调用；
2. Harness 实际执行工具；如果 Teacher 继续操作，它能够看到真实 tool result，再生成下一轮 assistant 输出。

因此监督数据不是脱离环境的“工具调用建议”，而是包含一次真实环境反馈的短程闭环：

```text
学生失败轨迹中的状态
        ↓
Teacher assistant：分析并调用工具
        ↓
Harness：执行工具并返回真实结果
        ↓
Teacher assistant：根据结果继续判断或修正
```

Teacher 不需要把任务重新跑到终局，也不对两轮短 continuation 单独计算 resolved。这里的目标是学习局部决策，而不是把每个分叉都变成一次完整 SWE 评测。

### 4. 学生模仿 Teacher continuation

每条训练样本由两部分组成：

```text
学生在分叉点保存的上下文         不计算 loss
Teacher assistant 输出             计算 CE loss
真实 tool result                   不计算 loss
Teacher 后续 assistant 输出        计算 CE loss
```

学生模型以 teacher forcing 方式对完整序列做 forward，但只有 Teacher 生成的 assistant token 参与交叉熵。原始学生前缀用于提供问题和历史状态，tool result 用于提供环境反馈；它们都是条件，不是模仿目标。

因此，该方法训练的是：在学生真实失败状态下，应该输出怎样的分析、工具选择、工具参数，以及看到工具结果后应该如何继续。

### 5. 等一批原始任务完成后统一更新

当前每次参数更新固定处理 16 道原始题，每题先采样 8 条学生轨迹。某道题的 8 条轨迹完成后，它的 Teacher continuation 可以立即开始生成；但必须等这一批 16 道题都处理完，才能进行一次参数更新。

不同题产生的 Teacher 样本数可能差异很大，因此不能直接对整批所有 token 求平均，否则失败轨迹更长、工具调用更多的题会获得更高训练权重。当前目标按三层归一化：

1. 一条 Teacher continuation 内，对可训练 token 求平均；
2. 同一道题内，对所有 Teacher continuation 求平均；
3. 对本批次中产生有效 Teacher 数据的题求平均。

这样每道有效题的总权重相同，不会因为某道题可分叉位置更多就主导梯度。

# 训练目标

设一道题为 $x$，学生模型为 $\pi_\theta$，强模型为 $\pi_T$。学生对该题生成 $K$ 条轨迹 $\tau_1,\ldots,\tau_K$，终局结果为 $r_i\in\{0,1\}$。

对于失败轨迹 $r_i=0$，令 $U_i$ 表示其中所有有效工具决策位置。Teacher 从位置 $u\in U_i$ 对应的状态续跑，得到 continuation $y_{i,u}$。令 $M_{i,u}$ 为其中需要训练的 Teacher assistant token 集合，则单条 continuation 的损失为：

$$
\ell_{i,u}
=
-\frac{1}{|M_{i,u}|}
\sum_{j\in M_{i,u}}
\log \pi_\theta\left(y_{i,u,j}\mid c_{i,u,j}\right),
$$

其中 $c_{i,u,j}$ 包含学生保存的分叉前缀、Teacher 已生成的前序 token，以及 Harness 返回的真实 tool result。

令一道题最终成功生成且通过数据检查的有效 relabel 集合为：

$$
A_x=\{(i,u)\mid r_i=0,\ u\in U_i,\ y_{i,u}\ \text{有效}\}.
$$

题目级损失为：

$$
L_x
=
\frac{1}{|A_x|}
\sum_{(i,u)\in A_x}
\ell_{i,u}.
$$

若一个 batch 中产生有效 Teacher 数据的题目集合为 $B^+$，最终训练目标为：

$$
L
=
\frac{1}{|B^+|}
\sum_{x\in B^+}
L_x.
$$

没有失败轨迹或没有有效恢复位置的题不产生梯度。如果整个 batch 都没有有效 Teacher 数据，则不执行这次参数更新。

# 为什么使用两轮 Teacher continuation

只采一个 assistant turn 时，Teacher 可能刚输出 Read、Edit 或 Bash，模型只能学到“发起什么动作”，无法学习如何解释工具结果。让 Teacher 完整跑到任务结束又会显著增加沙箱时间和评测成本。

两轮 continuation 提供了最短的工具闭环：第一轮采取动作，第二轮看到真实结果后继续决策。它不能证明局部动作最终一定解决整题，但能够监督长程 Agent 中最常见的“动作—观察—再决策”过程。

# 与 Hybrid Turn-Level GRPO 的关系

两种方法使用相同类型的学生中间状态，但训练信号不同：

- Hybrid Turn-Level GRPO 从同一状态采样多条**学生 continuation**，跑到终局后比较 reward；
- Turn-Level Teacher Imitation 从该状态采样**Teacher continuation**，直接模仿 Teacher token；
- 前者保留终局任务目标，但仍需要完整续跑和评测；后者提供更直接的局部监督，但依赖 Teacher 在该状态下的决策质量。

当前实验采用纯模仿口径：学生完整轨迹只负责产生失败状态，不同时计算 Stage-1 GRPO loss，也不运行学生的分叉 GRPO。

# 当前实验设置

| 项目 | 设置 |
| :-: | :-: |
| 学生模型 | Qwen3.5-9B |
| Teacher | DeepSeek-V4-Flash-0731 |
| 每次更新的原始题数 | 16 |
| 每题学生完整轨迹数 | 8 |
| 选点范围 | 失败轨迹的全部有效工具决策位置 |
| 每条 Teacher continuation | 最多 2 个 assistant turn |
| 工具执行 | 使用真实 SWE 沙箱执行 |
| Teacher continuation 终局评测 | 不执行 |
| 训练模式 | `sft_only` |
| 学习目标 | Teacher assistant token 的交叉熵 |
| 样本权重 | 有效题等权，题内 continuation 等权，continuation 内 token 等权 |

# 方法能够学习什么

- 学生在错误或不完整代码状态下应该先检查什么；
- 应该选择哪个工具以及如何填写工具参数；
- 如何根据真实命令输出、文件内容或报错调整下一步；
- 如何从学生已经形成的错误探索路径中恢复，而不只会复现标准成功轨迹。

# 方法边界

- Teacher 只运行短 continuation，局部监督不等价于整题必然 resolved；
- 方法会继承 Teacher 的错误判断和行为偏好；
- 长失败轨迹会产生较多 Teacher 请求，计算成本仍然较高；
- 当前不训练学生成功轨迹，也不把终局 reward 加入 imitation loss；
- 方法成立的前提是分叉状态能够准确恢复，Teacher 推理和学生训练必须基于同一逻辑上下文。

# 下一步设计

当前方法会在失败轨迹的全部有效工具决策位置调用 Teacher。这样能够最大限度覆盖学生访问过的错误状态，但长轨迹会产生大量 continuation，其中一部分行为已经被学生掌握，继续请求 Teacher 和执行 SFT 的边际收益较低。

下一步可以把“是否值得向 Teacher 查询”建模为 turn 级学习价值预测。对候选位置 $u$，Teacher continuation 为 $y_u^T$，其中参与训练的 assistant token 集合为 $M_u$。当前学生在这条 continuation 上的 token 平均交叉熵定义为：

$$
v_\theta(u)
=
-\frac{1}{|M_u|}
\sum_{j\in M_u}
\log \pi_\theta\left(y^T_{u,j}\mid c_{u,j}\right).
$$

$v_\theta(u)$ 较高，表示当前学生给 Teacher 行为分配的概率较低，从该位置获得的监督信号通常更强；$v_\theta(u)$ 较低，表示学生已经较接近 Teacher，重复训练的收益可能较小。这里的分数应称为**学习价值**，而不是 turn 对最终 resolved 的因果重要性：高交叉熵也可能来自罕见表达、复杂工具参数或异常上下文，并不保证该 turn 决定任务成败。

真实的 $v_\theta(u)$ 只有在 Teacher 已经生成 continuation 后才能计算，因此不能直接用它提前减少 Teacher 请求。下一步在学生模型主干上增加一个标量 **Teacher Query Value Head**，用候选 turn 开始前的 hidden state 预测该位置尚未观测到的 Teacher CE。

设 $h_u$ 为候选 turn 开始前的学生 hidden state，预测值为：

$$
\hat v_u
=
g_\phi\left(\operatorname{sg}(h_u)\right),
$$

其中 $g_\phi$ 是新增的标量头，$\operatorname{sg}$ 表示先阻断该辅助目标对语言模型主干的梯度。Teacher continuation 生成后，现有 SFT forward 计算出的 $v_\theta(u)$ 作为监督标签，使用 Huber loss 训练该头：

$$
L_{\mathrm{head}}
=
\operatorname{Huber}
\left(
\hat v_u,
\operatorname{stopgrad}\left(v_\theta(u)\right)
\right).
$$

它在结构上类似 PPO 的 Value Head，但预测目标和用途不同：

| | PPO Value Head | Teacher Query Value Head |
| :-: | :-: | :-: |
| 输入 | 当前状态的 hidden state | 候选 turn 开始前的 hidden state |
| 预测目标 | 从当前状态出发的未来期望 reward | 当前学生在 Teacher continuation 上的预期 CE |
| 监督标签 | Monte Carlo return 或 GAE | 实际 Teacher continuation 的 token 平均 CE |
| 用途 | 计算 advantage | 决定是否值得请求 Teacher |

完整流程分为两个阶段：

1. **收集监督标签**：利用当前“全部有效 turn 都查询 Teacher”的训练数据，为每个候选位置保存 $h_u$ 和实际 $v_\theta(u)$，作为 Value Head 的冷启动数据；
2. **预测并选择**：后续学生生成轨迹时同步输出每个候选位置的 $\hat v_u$，在每道题或每条失败轨迹中选择预测值最高的 Top-$B$ 个 turn，只在这些位置运行 Teacher continuation 和 SFT。

为避免 Value Head 只看到自己已经偏好的位置，选点时仍需保留少量均匀随机候选。随机候选产生的真实 CE 用于持续校准预测，也能发现“预测值低但实际学习价值高”的新状态。被选中的 Teacher 样本继续沿用当前的 continuation 内、题内和 batch 内三层归一化，不再将原始 CE 直接乘入 SFT loss，避免极端高损失样本获得不受控的训练权重。

该设计只有在 $\hat v_u$ 能随学生原始 rollout 同步产生时才能真正节省成本。如果先完成 rollout，再对全部长上下文额外执行一次模型 forward 计算 Value Head，额外计算可能抵消减少 Teacher 请求带来的收益。因此工程上需要让训练、checkpoint 和 SGLang rollout 同时支持该标量头，并在候选 turn 边界直接返回预测值。

该设计能否减少成本并保持效果，需要同时观察以下指标：

| 指标 | 作用 |
| :-: | :- |
| Teacher continuation 数量 / task | 衡量查询和沙箱成本是否下降 |
| 入选 turn 的实际 CE | 检查 Teacher Query Value Head 是否选中了学生尚未掌握的行为 |
| 随机候选与 Top-$B$ 候选的实际 CE 差异 | 衡量排序是否有效 |
| Value Head 预测误差与排序相关性 | 判断预测分数是否可以用于稳定选点 |
| Value Head 计算开销 | 检查选点本身是否抵消 Teacher 成本收益 |
| 固定训练步数或固定 Teacher 预算下的 resolved | 判断节省查询后是否保留或提高实际学习效果 |

因此，该方案在训练目标上可行。正确形式不是“直接按当前 turn 的 CE 选点”，而是“用已经查询过的 Teacher continuation CE 训练一个共享学生主干的标量头，再预测尚未查询 turn 的预期学习价值”。当前全 turn 训练产生的数据正好可以作为该 Value Head 的冷启动数据。
