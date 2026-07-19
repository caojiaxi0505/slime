# 【已废弃】精确状态分叉控制变量策略梯度

英文名：Exact-State Branch Control-Variate Policy Gradient  
日期：2026-07-19  
状态：研究方案，尚未实现

## 1. 一句话说明

先按朴素 GRPO 对同一道题生成一组完整轨迹，并按朴素 GRPO 的现有 reward、advantage 和 loss
逻辑计算 reward 上升方向。再从一个中间状态生成多条新分支，用这些分支重新估计该位置的局部
上升方向。新估计
与原估计的差值经过采样概率校正后，作为一个期望为零的修正项：

$$
\widehat g=g_0+\alpha C.
$$

- $g_0$ 是朴素 GRPO 根据完整轨迹算出的 reward 上升方向；
- $C$ 是分叉数据构造出的零均值修正；
- $\alpha$ 控制修正强度。

这里的朴素 GRPO 就是当前对照路线：同一道题生成多条完整 episodes，用包含当前样本自身的
组均值和组标准差计算 advantage，整条 episode 进入 GRPO loss，不做中途分叉。

分叉数据不作为第二份 loss 直接相加。它旨在降低 $g_0$ 的噪声，同时保持朴素 GRPO
原始梯度估计的期望不变。

## 2. 为什么需要分叉

一条 coding-agent 轨迹通常包含很多轮交互，但最终只有一个任务 reward。较早动作得到的 reward
同时受后面许多动作影响，因此同一个早期动作可能因为后续采样不同而得到完全不同的结果。

从同一个中间状态重新采样多个下一步动作，并把每条分支继续运行到任务结束，可以更稳定地回答：

> 已经到达这个状态后，下一步动作对最终结果有什么影响？

但是，不能把这些分支直接当作额外训练样本。分叉位置通常是有选择的，直接增加其 loss 会使该
位置得到额外权重。本文使用“新局部估计减去原局部估计”来消除这份额外权重。

## 3. 训练对象和符号

一次 assistant turn 视为一个动作。它可以包含 reasoning、文本和多个工具调用，不是单个 token，
也不专指一次 Edit。

| 符号 | 定义 |
|---|---|
| $\theta$ | 模型参数 |
| $\theta_0$ | 生成当前数据时冻结的模型参数快照 |
| $\pi_\theta$ | 参数为 $\theta$ 的策略；$\pi_{\theta_0}$ 是本批 rollout policy |
| $\mathcal B=\{\tau_i\}_{i=1}^{N}$ | 同一道题生成的 $N$ 条完整根轨迹 |
| $\mathcal I(\mathcal B)$ | 根轨迹中所有符合预先规定资格的 assistant turns |
| $e(v)$ | turn $v$ 所属的根轨迹编号 |
| $s_v$ | turn $v$ 生成前的完整状态 |
| $a_v$ | 在 $s_v$ 生成的一整个 assistant turn |
| $R_i$ | 根轨迹 $\tau_i$ 的最终 reward |
| $g_0$ | 朴素 GRPO 根据 $\mathcal B$ 计算的 reward 上升方向 |
| $z_v$ | 动作 $a_v$ 的 score，即训练 token 的 log-prob 梯度之和 |
| $\bar R$ | $N$ 条根轨迹的 reward 均值，包含当前样本自身 |
| $h_v$ | 原轨迹对 turn $v$ 给出的辅助局部梯度向量，不是 reward 或 advantage |
| $U$ | 被抽中进行分叉的 turn |
| $q_v$ | 抽中 turn $v$ 的已知概率 |
| $K$ | 从相同状态新生成的分支数 |
| $\bar h_v$ | 假如在位置 $v$ 分叉，多条新分支给出的局部梯度估计 |
| $\bar h_U$ | 实际被选中位置 $U$ 对应的 $\bar h_v$ |
| $C$ | 零均值修正项 |
| $\alpha$ | 方差控制系数 |
| $\widehat g$ | 加入控制变量后的最终 reward 上升方向 |

turn 索引 $v$ 是二元索引 $(i,t)$ 的简写，其中 $i$ 是根轨迹编号，$t$ 是该轨迹中的 turn
编号。一个 turn 是否属于 $\mathcal I(\mathcal B)$，必须在该 turn 的动作生成前即可确定；
不能看完动作或终局 reward 后再把它加入或移出候选集合。

全文的 $g_0$、$z_v$、$h_v$、$\bar h_v$、$C$ 和 $\widehat g$ 都表示最大化 reward 的上升
方向。若代码通过梯度下降最小化 loss，则实现中使用这些方向的相反数；两种符号不能混用。

完整状态 $s_v$ 至少包括：

- 模型实际使用的 prompt token；
- 已经返回给模型的消息和工具结果；
- Claude Code 会话状态；
- workspace 的文件、权限、symlink、mtime 和 Git 状态；
- 会影响后续执行的 runtime 状态。

只恢复文本 transcript 或只恢复代码文件，都不等于恢复了同一个状态。

## 4. 方法

### 4.1 生成完整根轨迹

用 $\pi_{\theta_0}$ 对同一道题生成 $N\ge2$ 条完整轨迹：

$$
\mathcal B=\{\tau_1,\ldots,\tau_N\}.
$$

朴素 GRPO 照常根据这批完整轨迹计算 reward 上升方向：

$$
g_0\equiv g_{\mathrm{GRPO}}(\mathcal B).
$$

$g_0$ 保留当前朴素 GRPO 的 reward、advantage、clipping 和 loss 归一化逻辑。本文不修改它。
朴素 GRPO 的组均值包含当前样本：

$$
\bar R=\frac{1}{N}\sum_{j=1}^{N}R_j.
$$

当前朴素 GRPO 开启 std normalization 时，使用 population standard deviation：

$$
\sigma_R
=
\sqrt{\frac{1}{N}\sum_{j=1}^{N}(R_j-\bar R)^2},
\qquad
A_i^{\mathrm{GRPO}}
=
\frac{R_i-\bar R}{\sigma_R+10^{-6}}.
$$

同时，为每个 turn 计算一个只用于构造 $C$ 的辅助局部估计。先把 GRPO 的组中心化 reward
乘以确定性的尺度校正：

$$
\widetilde A_i
=
\frac{N}{N-1}(R_i-\bar R)
=
R_i-\frac{1}{N-1}\sum_{j\ne i}R_j.
$$

第一个写法明确使用了包含自身的 GRPO 组均值。第二个写法只是代数恒等式，用来说明：
经过尺度校正后，被减去的比较均值不再包含当前样本自己的 reward；$\widetilde A_i$ 本身仍然
包含 $R_i$。

turn $v$ 的 score 为：

$$
z_v=
\left.
\sum_{\ell\in a_v}
\nabla_\theta
\log\pi_\theta(a_{v,\ell}\mid s_v,a_{v,<\ell})
\right|_{\theta=\theta_0}.
$$

其中 $\ell$ 是 turn 内的 token 下标。局部估计为：

$$
h_v=z_v\widetilde A_{e(v)}.
$$

$h_v$ 是一个梯度向量，与模型参数同维度。$z_v$ 给出“怎样调整参数会提高动作 $a_v$ 的概率”，
$\widetilde A_{e(v)}$ 决定这个方向应被加强还是减弱。因此，$h_v$ 表示：只看这一条完整轨迹的
最终结果时，动作 $a_v$ 得到的局部训练信号。

$h_v$ 不是朴素 GRPO advantage，也不是 $g_0$ 的强制拆分结果。$g_0$ 仍使用原来的 GRPO
advantage；$\widetilde A_i$ 只用于构造零均值修正，并且不除以本组 reward 的随机 group
standard deviation。

### 4.2 选择一个分叉位置

从所有候选 turns 中抽取位置 $U$。若
$\mathcal I(\mathcal B)=\varnothing$，本批直接令 $C=0$，只使用 $g_0$；以下假设候选集合非空：

$$
U\sim q(\cdot\mid\mathcal B).
$$

每个 turn 都必须有非零概率：

$$
q_v>0,\qquad \forall v\in\mathcal I(\mathcal B).
$$

可以让高 edit-PPL 位置更容易被选中。先给每个 turn 定义一个实数分数 $x_v$，例如 edit turn
使用 edit-PPL，其他 turn 使用 $0$。再将分数归一化：

$$
p_{\mathrm{focus},v}
=
\frac{\exp(\beta x_v)}
{\sum_{u\in\mathcal I(\mathcal B)}\exp(\beta x_u)},
\qquad \beta\ge0.
$$

$\beta$ 控制分布集中程度：$\beta=0$ 时为均匀分布，$\beta$ 越大，概率越集中到高分位置。
实际计算应使用数值稳定的 softmax。此时 $\sum_v p_{\mathrm{focus},v}=1$。再与均匀分布混合：

$$
q_v=
\epsilon\frac{1}{|\mathcal I(\mathcal B)|}
+(1-\epsilon)p_{\mathrm{focus},v},
\qquad 0<\epsilon\le1.
$$

因此：

$$
\sum_{v\in\mathcal I(\mathcal B)}q_v
=
\epsilon+(1-\epsilon)
=1.
$$

均匀项不负责归一化；$p_{\mathrm{focus}}$ 本身已经归一化。它的作用是给每个位置设置概率下限：

$$
q_v\ge\frac{\epsilon}{|\mathcal I(\mathcal B)|},
\qquad
\frac{1}{q_v}\le\frac{|\mathcal I(\mathcal B)|}{\epsilon}.
$$

例如共有 $100$ 个 turns、$\epsilon=0.1$，可以理解为 $90\%$ 的选择概率按 edit-PPL 分配，
$10\%$ 均匀分配。每个 turn 的概率至少为 $0.001$，所以 $1/q_v$ 最大为 $1000$。

若只使用有限分数的精确 softmax，理论上每个概率也都大于零；但有些概率可能极小，甚至在实际
计算中下溢为零。均匀项用于限制这种长尾。后面的 $1/q_U$ 负责消除非均匀选择带来的额外权重。

### 4.3 从相同状态生成新分支

恢复原动作 $a_U$ 生成前的精确状态 $s_U$，再从同一个 policy 生成 $K\ge2$ 条独立分支：

$$
a_U^{(k)}\sim\pi_{\theta_0}(\cdot\mid s_U),
\qquad k=1,\ldots,K.
$$

每个新动作都继续运行到任务结束，得到最终 reward $R_U^{(k)}$。完整续跑只用于评价恢复后的
第一个 assistant turn；续跑部分的动作不进入这个局部梯度。

分支组的 reward 均值同样包含每条分支自身：

$$
\bar R_U^{\mathrm{branch}}
=
\frac{1}{K}
\sum_{r=1}^{K}R_U^{(r)}.
$$

只对分支的第一个 assistant turn 计算 score：

$$
z_U^{(k)}
=
\left.
\sum_{\ell\in a_U^{(k)}}
\nabla_\theta
\log\pi_\theta
\left(a_{U,\ell}^{(k)}\mid s_U,a_{U,<\ell}^{(k)}\right)
\right|_{\theta=\theta_0}.
$$

这里使用与根轨迹 $z_v$ 完全相同的训练 token 范围和预先确定的 token mask。

多分支局部估计为：

$$
\bar h_U
=
\frac{1}{K}
\sum_{k=1}^{K}
z_U^{(k)}
\frac{K}{K-1}
\left(R_U^{(k)}-\bar R_U^{\mathrm{branch}}\right).
$$

$N/(N-1)$ 和 $K/(K-1)$ 分别消除不同组大小带来的确定性尺度差。即使 $N\ne K$，
$h_U$ 和 $\bar h_U$ 仍估计同一个条件均值。

### 4.4 构造修正项

原轨迹在位置 $U$ 已经提供了一个单样本估计 $h_U$，新分支使用 $K$ 个条件样本得到
$\bar h_U$。两者相减，再除以该位置的抽样概率：

$$
C=
\frac{1}{q_U}
\left(\bar h_U-h_U\right).
$$

最终使用：

$$
\boxed{\widehat g=g_0+\alpha C}.
$$

若每批有放回地独立抽取 $M\ge1$ 个位置，使用：

$$
C_M=
\frac{1}{M}
\sum_{m=1}^{M}
\frac{\bar h_{U_m}-h_{U_m}}{q_{U_m}}.
$$

若无放回地抽到集合 $S$，令
$\rho_v=\Pr(v\in S\mid\mathcal B)>0$ 为位置 $v$ 的包含概率，则使用：

$$
C_S=
\sum_{v\in S}
\frac{\bar h_v-h_v}{\rho_v}.
$$

无放回公式通常不能再机械除以 $M$。

## 5. 为什么 $C$ 的期望为零

对任意候选位置 $v$，$\bar h_v$ 表示“假如在 $v$ 分叉”得到的多分支估计；实际只计算
$\bar h_U$。给定状态 $s_v$ 后，原轨迹的当前动作及后缀与 $K$ 条新分支必须使用相同的 policy
和环境执行规则，并且条件独立、同分布。记该状态下局部梯度的真实条件均值为：

$$
\mu(s_v)
=
\mathbb E[h_v\mid s_v]
=
\mathbb E[\bar h_v\mid s_v].
$$

同组的其他根轨迹还必须与被评分的根轨迹独立；同组的其他分支也必须与当前分支条件独立。
score 覆盖完整随机动作，或者 token mask 在相应 token 采样前已经确定。此时：

$$
\mathbb E[z_v\mid s_v]=0.
$$

第 4.1 节和第 4.3 节的尺度校正分别满足：

$$
\frac{N}{N-1}(R_i-\bar R)
=
R_i-\frac{1}{N-1}\sum_{j\ne i}R_j,
$$

$$
\frac{K}{K-1}
\left(R_U^{(k)}-\bar R_U^{\mathrm{branch}}\right)
=
R_U^{(k)}
-\frac{1}{K-1}\sum_{r\ne k}R_U^{(r)}.
$$

两个被减去的均值都与当前被评分动作独立，因此其 score 期望为零。于是 $h_v$ 和
$\bar h_v$ 的条件均值都等于：

$$
\mu(s_v)
=
\mathbb E\!\left[z_vR_{e(v)}\mid s_v\right].
$$

给定已经生成的根轨迹，先对分叉位置和新分支取期望：

$$
\begin{aligned}
\mathbb E_{U,\mathrm{branch}}[C\mid\mathcal B]
&=
\sum_{v\in\mathcal I(\mathcal B)}
q_v\frac{1}{q_v}
\left(\mu(s_v)-h_v\right)\\
&=
\sum_{v\in\mathcal I(\mathcal B)}
\left(\mu(s_v)-h_v\right).
\end{aligned}
$$

$q_v$ 与 $1/q_v$ 相消，所以 $q$ 可以依赖完整根轨迹，包括 edit-PPL 和 reward，但不能把任何
候选位置的概率设为零。

由于每条轨迹的 turn 数是随机的，还需要以下条件之一：

- 每条轨迹有固定的最大 turn 数；或者
- 候选 turn 的总修正绝对可积：

$$
\mathbb E\!\left[
\sum_{v\in\mathcal I(\mathcal B)}
\left\lVert\mu(s_v)-h_v\right\rVert_2
\right]
<\infty.
$$

同时，turn 是否存在以及是否有候选资格，必须在该 turn 动作采样前即可确定。满足这些条件后，
对根轨迹使用条件期望和停止和论证可得：

$$
\mathbb E[C]=0.
$$

因此，只要 $\alpha$ 在当前数据生成前已经确定：

$$
\mathbb E[\widehat g]=\mathbb E[g_0].
$$

$C$ 不需要在每个 batch 内等于零。它只需要在重复采样后的期望上等于零。

## 6. $\alpha$ 怎么确定

$\alpha$ 不是第二份 loss 的权重。它决定加入多少零均值修正。假设：

$$
\mathbb E\!\left[\lVert g_0\rVert_2^2\right]<\infty,
\qquad
0<
\mathbb E\!\left[\lVert C\rVert_2^2\right]
<\infty.
$$

使
$\mathbb E[\lVert\widehat g-\mathbb E[g_0]\rVert_2^2]$
最小的标量为：

$$
\alpha^*
=-
\frac{
\mathbb E\!\left[
\left\langle g_0-\mathbb E[g_0],C\right\rangle
\right]
}{
\mathbb E\!\left[\lVert C\rVert_2^2\right]
}
=-
\frac{
\mathbb E\!\left[\langle g_0,C\rangle\right]
}{
\mathbb E\!\left[\lVert C\rVert_2^2\right]
}.
$$

实际训练中，$\alpha$ 必须在当前根轨迹、分叉位置和新分支采样前，由历史 batches 或独立 pilot
确定。若 $\mathbb E[\lVert C\rVert_2^2]=0$，令 $\alpha=0$。若相关二阶矩不有限，最优公式
没有定义，必须先修改采样分布或修正项。若历史估计显示修正没有降低方差，也令 $\alpha=0$，
训练就退回朴素 GRPO 方向 $g_0$。

## 7. 必须满足的条件

1. **状态相同。**所有新分支从同一个模型上下文、会话、workspace 和 runtime 状态开始。
2. **分布相同。**给定相同状态，原动作及后缀和各条新分支使用相同 policy、采样配置、工具
   与环境执行规则，并且条件独立、同分布。
3. **局部 score 合法。**不能根据动作生成后的内容再决定保留哪些 token score。
4. **局部估计口径一致。**$h_U$ 和 $\bar h_U$ 使用相同 reward 和动作范围。
5. **组均值包含自身并校正尺度。**根轨迹使用 $N/(N-1)$，分支使用 $K/(K-1)$，保证组大小
   不同时两侧仍估计同一个量。
6. **局部估计不做随机标准差归一化。**朴素 GRPO 的 $g_0$ 可以保留 std normalization；
   $h_U$ 和 $\bar h_U$ 不能分别除以各自组内 reward 的随机 group standard deviation。
7. **选择概率全覆盖。**所有候选 turns 的 $q_v$ 都大于零并准确记录，且
   $\mathbb E[\lVert C\rVert_2^2]<\infty$。
8. **候选资格不能事后决定。**turn 是否存在、是否可被分叉，必须在其动作生成前即可确定；
   不能只保留 edit、失败或最终高 reward 的 turns。轨迹长度还必须有固定上限，或满足第 5 节
   的绝对可积条件。
9. **$\alpha$ 提前确定。**不能看完当前 batch 后再拟合 $\alpha$ 并回乘到同一批数据。
10. **不能按结果删轨迹或分支。**根轨迹组不能按 resolved、reward 或动作内容事后删样本。
    动作自身导致的合法超时、非法调用和未解决，必须按预先规定且两侧一致的终止与 reward 规则
    计入。模型服务、网络或评测基础设施造成的缺失不能简单当作负 reward，也不能只保留成功返回
    的分支。失败后令 $C=0$ 只有在失败在动作生成前可确定，或给定状态后与
    $h_v,\bar h_v$ 条件独立时才保持零均值。

严格零均值结论针对 rollout policy $\pi_{\theta_0}$ 处计算的原始修正向量 $C$。参数更新后重复
使用同一个 $C$，不能再声称每一步都保持
$\mathbb E[\widehat g]=\mathbb E[g_0]$。此外，在
$\widehat g$ 之后使用依赖当前样本的 global norm clipping、Adam 等非线性变换，也不能由
$\mathbb E[C]=0$ 推出实际参数更新的期望不变。

## 8. 每种数据实际负责什么

| 数据 | 用途 | 如何进入最终梯度 |
|---|---|---|
| 完整根轨迹 | 计算朴素 GRPO 梯度 $g_0$ | 按朴素 GRPO 现有逻辑计算 |
| 根轨迹中的原动作 $a_U$ | 计算局部估计 $h_U$ | 随完整根轨迹进入 $g_0$，并额外用于 $C$ |
| 新分支的第一个动作 $a_U^{(k)}$ | 计算多分支估计 $\bar h_U$ | 只用于 $C$ |
| 新分支第一个动作之后的续跑 | 得到 $R_U^{(k)}$，评价第一个动作 | 不计算局部修正梯度 |

最重要的限制是：

> 不能把新分支单独产生的 reward 上升方向直接加到 $g_0$；它必须与原动作的 $-h_U$ 及
> $1/q_U$ 一起构成有正有负的修正方向 $C$。

## 9. 与当前 Hybrid 实现的对应关系

| 本文概念 | 当前实现中的对应部分 | 需要怎样处理 |
|---|---|---|
| 完整根轨迹 $\mathcal B$ | Stage-1 的完整 attempts | 全部 episode 有效且 std 配置相同时，当前 advantage 数值与朴素 GRPO 等价；异常或部分 episode 为零 mask、随后只按 surviving episodes 归一化和重加权时可能不等价。std=0 或全组 mask=0 的整组过滤本身仍是零梯度。正式实现应直接复用朴素 GRPO 的 post-process 与 reducer |
| 原轨迹局部估计 $h_v$ | Stage-1 的 token、reward、log-prob 和 checkpoint | 需要新增稳定的 turn id 与 token span，再按包含自身的组均值及 $N/(N-1)$ 计算；现有数据不能直接视为已经得到全部 $h_v$ |
| 分叉位置 $U$ 与概率 $q_U$ | 当前只从失败 attempts 的 patch steps 中确定性选择 global top-k edit-PPL | 现逻辑中大量位置 $q_v=0$，不能用于本方法；必须先按 assistant turn/checkpoint 去重，再对预先定义候选集合做全覆盖概率采样，edit-PPL 只能改变概率大小 |
| 精确恢复状态 $s_U$ | token-exact checkpoint、原生 session 和 workspace snapshot | 当前只支持满足约束的 main-chain、非 wipe、可映射且不含 Task/Agent 的 tool turn；要覆盖候选集合，必须扩展 text-only、非 patch 等 turn 的 session/workspace 映射，或在动作生成前定义可恢复资格 |
| $K$ 条完整新分支 | 当前为每个 top-k 位置生成 $K$ 条 Stage-2 branch rollouts，失败 branch 会被丢弃 | 保留完整续跑用于终局 reward，但必须遵守已记录的抽样设计，并按第 7 节规则处理所有结果与基础设施缺失 |
| 多分支估计 $\bar h_U$ | 当前没有直接对应；Stage-2 会训练整个 continuation | 必须新增 resumed first-turn 的稳定 token span 和局部 mask，只用第一个 turn 构造 $\bar h_U$ |
| 当前 Stage-2 梯度贡献 | Stage-2 整个 continuation 通过独立 loss 产生梯度 | 删除该独立梯度贡献，改为同时包含 $\bar h_U$ 和 $-h_U$ 的修正方向 $C$ |
| 修正项 $C$ | 当前 sample-level scalar advantage 无法表达同一根轨迹上的局部 $-h_U$ | 新增 per-token/per-span auxiliary advantage，或单独的局部 loss 路径 |
| 最终更新 $\widehat g$ | 当前由 Stage-1 loss 与 Stage-2 loss 共同产生的更新方向 | 改为 $g_0+\alpha C$ |

按旧 setting 的 $K=8$，当前逻辑最多选择 $8$ 个位置、每个位置生成 $8$ 条 branches；这对应
最多 $8$ 个分叉组、每组 $8$ 条分支，不是“只抽一个位置”。

换句话说：

- 当前 Stage-1 对应本文的“完整根轨迹”；
- 当前 Stage-2 对应本文的“同状态新分支”；
- 真正新增的算法部分不是分叉本身，而是用
  $C=(\bar h_U-h_U)/q_U$ 代替直接加入的 Stage-2 梯度贡献。

## 10. 最小验证

先从同一个 checkpoint 做短程 pilot，至少记录：

- $\alpha$、$\lVert g_0\rVert$、$\lVert C\rVert$ 和 $\lVert\widehat g\rVert$；
- $\langle g_0,C\rangle$；
- $q_v$ 和 $1/q_v$ 的分布；
- 跨 batch 的梯度方差；
- resolved/128、每步 wall time 和最终 SWE484。

出现以下任一情况，不应扩大实验：

1. $\alpha$ 长期接近零；
2. $\widehat g$ 的实测方差不低于 $g_0$；
3. $1/q_v$ 长尾使 $C$ 比 $g_0$ 更不稳定；
4. 在相同 wall time 或 rollout 预算下，最终评测没有改善。

零均值只说明修正项不改变朴素 GRPO 梯度的期望，不保证它一定提高最终能力。
