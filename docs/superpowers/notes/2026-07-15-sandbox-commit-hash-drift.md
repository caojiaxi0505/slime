# Rollout 沙箱初始 commit hash 漂移

日期：2026-07-15

> 2026-07-30 更新：SWE-bench Classic rollout 已不再创建无父 `slime initial bug state`
> commit，而是保留真实 base 历史并创建普通 `SWE-bench` 子提交。本文仍是当时问题和
> SWE-Smith/Scale-SWE orphan 路径的有效记录。后续修复见
> `2026-07-30-swebench-git-workspace-official-alignment.md`。

适用分支：`feature/cc-ags-swe`

状态：已修复并通过针对性测试；尚未提交 commit，未重启任何训练任务

关联实现：`examples/claudecode_ags/workspace_init.py`

## 1. 结论

修复前的 rollout workspace 初始化不是完全可重复的。

同一份代码树在不同时间创建沙箱时，`__slime_buggy` 分支上的初始 commit hash 曾经可能不同。Claude Code 会把该 hash 放进 system prompt 的 `Recent commits`，因此即使不使用 transcript resume，两次运行的完整模型输入也可能从 commit hash 处开始不同。

现在已经固定合成 commit 的时间字段：后续新建的相同代码树会得到相同 HEAD。该修复只处理 commit 漂移，不处理 resume 消息重建。

这是一项独立于 transcript resume 的 prompt 可复现性问题。它不能解释 resume 路径中的 tool result 丢失和消息重排，但会使 token-exact 比较更早失败。

## 2. 修复前证据

2026-07-15 的 token-exact resume 诊断中，两份内容等价、分别初始化的 workspace 产生了不同的初始提交：

```text
Stage-1: 80748ea slime initial bug state
Stage-2: ec3eb88 slime initial bug state
```

Claude Code 将它们写入 system prompt：

```text
Recent commits:
80748ea slime initial bug state
```

和：

```text
Recent commits:
ec3eb88 slime initial bug state
```

因此两边 `prompt_ids` 的首个差异出现在 0-based 下标 `23118`，即人类计数的第 `23119` 个 token。前面的 23,118 个 token 相同。

离线分量分析显示，更换 Stage-1/Stage-2 system prompt 会造成 1 token 的总长度差。这里的“1 token”只是长度净差；两个 hash 的实际 token 值从该位置起已经不同。

## 3. 根因

rollout 侧的 Git scrub 会删除已有 refs，然后为当前代码树创建一个新的无父提交：

```bash
git checkout --orphan __slime_buggy
git add -A
git commit --allow-empty -m 'slime initial bug state'
```

该提交固定了：

- 代码树；
- commit message；
- author/committer 名称和邮箱。

但没有固定：

- author timestamp；
- committer timestamp。

Git commit hash 是整个 commit object 的摘要，其中包含 author/committer 时间。只要两个沙箱的 commit 创建时间不同，即使代码树和 message 完全相同，commit hash 也可能不同。

## 4. 影响范围

直接影响：

- 同一任务在不同时间重新初始化 rollout 沙箱时，Claude Code system prompt 可能不同；
- Stage-1 与 Stage-2 即使 workspace 文件内容一致，也可能看到不同的 `Recent commits`；
- 任何要求完整 `prompt_ids` 相同的 token-exact 检查都会失败。

不由该问题解释的差异：

- `stream-json` resume 将真实 tool result 替换成 `[Tool result missing due to internal error]`；
- skills reminder、current-date reminder、原始任务和历史消息被重新排列；
- 历史 assistant reasoning 在 chat template 中的呈现差异。

这些仍是单独的 resume/消息重建问题。修复 commit hash 后，不能据此宣称 token-exact resume 已经实现。

## 5. 已实施的修复

`workspace_init.py` 现在为合成的 `slime initial bug state` commit 固定：

- `GIT_AUTHOR_DATE=2000-01-01T00:00:00+0000`；
- `GIT_COMMITTER_DATE=2000-01-01T00:00:00+0000`。

同时显式设置 `commit.gpgSign=false`，避免环境中的 Git 签名配置向这个合成 commit 注入不可重复的签名内容。

实际命令的关键部分为：

```bash
GIT_AUTHOR_DATE='2000-01-01T00:00:00+0000' \
GIT_COMMITTER_DATE='2000-01-01T00:00:00+0000' \
git -c user.email=slime@local \
    -c user.name=slime \
    -c commit.gpgSign=false \
    commit --allow-empty -q -m 'slime initial bug state'
```

固定时间只用于这个人为创建、用于隐藏原始历史的无父提交。它不表示真实项目时间，也不应应用到评测侧保留原 HEAD 的轻量 scrub。

当代码树、message、author、committer 和两个时间字段都相同时，生成的 commit object 与 hash 应稳定相同。不同代码树仍会得到不同 hash，这是正确行为。

## 6. 验证结果

新增测试：`tests/claudecode_ags/test_workspace_git_scrub.py`

该测试为两个相同代码树分别注入不同的外部 Git 时间，再执行真实 rollout scrub。修复前测试失败：

```text
first HEAD:  dff6646631a778272dfa341145570f2e8775bfb7
second HEAD: 808c94f3bfe6c53aa69fd41c2e27a40db40f059f
```

修复后测试通过：

```text
first HEAD:    a40bd90399341896b1e282e07ae3b09873d39b29
second HEAD:   a40bd90399341896b1e282e07ae3b09873d39b29
different HEAD: 56ebba2ba95ec6a9bba6155154eff3ced54767d2
author time:   946684800
committer time: 946684800
parents:       none
```

这里的 hash 来自测试用的 `tracked.txt`，不是某一道 SWE 任务的固定 hash。它验证的是：

- 相同 tree、不同外部时间：HEAD 相同；
- 不同 tree：HEAD 不同；
- 初始提交仍然没有 parent；
- author/committer 时间都固定为 2000-01-01 00:00:00 UTC。

测试结果：

```text
tests/claudecode_ags/test_workspace_git_scrub.py: 1 passed
tests/claudecode_ags/test_workspace_init.py
  + test_workspace_git_scrub.py: 21 passed
```

测试环境说明：本机默认 Python 3.9 低于项目要求，因此测试使用本机已有的 Python 3.13。新的真实 Git 测试不需要任何业务依赖替身；合并运行旧的 `test_workspace_init.py` 时，仅为该解释器缺失的 PyTorch 依赖提供了未参与 workspace/Git 路径的 `SingletonMeta` 导入替身，没有 mock 本次修改的命令生成或 Git 执行。

另外通过：

- Python 3.13 `py_compile`；
- `git diff --check`。

## 7. 修复记录

2026-07-15：

- 在 `workspace_init.py` 增加固定的合成 commit 时间；
- 对合成 commit 显式关闭 GPG signing；
- 增加跨目录、跨外部时间、不同 tree 和无 parent 的真实 Git 回归测试；
- 验证 workspace 初始化相关测试全部通过。

## 8. 剩余边界

该修复只消除 rollout 合成 commit 因创建时间不同而发生的 hash 漂移。

它不会：

- 修改已经创建的 workspace 或已经运行中的任务；
- 固定跨日期变化的 Claude Code `currentDate` reminder；
- 修复 `stream-json` resume 的 tool result 丢失；
- 修复 reminder、原始任务和历史消息的重排；
- 单独实现 token-exact resume。

因此，后续新建的相同代码树应得到相同的合成 HEAD；完整 Stage-1/Stage-2 prompt 是否一致，仍需分别检查 resume 消息状态。
