# SWE-bench Git workspace 官方逻辑对齐

日期：2026-07-30

适用分支：`feature/cc-ags-swe`

状态：已实现并通过定向测试；尚未提交 commit，尚未用于新训练或评测任务

关联实现：`examples/claudecode_ags/workspace_init.py`

## 1. 结论

SWE-bench Classic 的 rollout workspace 初始化已从“删除真实历史并创建无父合成 commit”改为：

1. 将工作区恢复到真实 `base_commit`；
2. 保留 `base_commit` 及其祖先历史；
3. 删除 remote、额外分支和可能暴露未来提交的 refs；
4. 删除指向 `base_commit` 之后提交的 tag；
5. 清理 reflog 和不可达对象，并检查未来提交已不可见；
6. 在真实 `base_commit` 上创建普通子提交 `SWE-bench`。

该拓扑与官方 SWE-bench 的 repository setup 语义一致。为保持 rollout prompt 可复现，本实现
额外固定了 `SWE-bench` setup commit 的时间；这是确定性增强，不改变代码树、父子关系或可见
历史。

本次修改只作用于 SWE-bench Classic，包括 SWE-Gym 和 SWE-bench Verified。SWE-Smith 和
Scale-SWE 仍沿用原来的无父合成 commit 逻辑。

## 2. 修复前的问题

修复前，SWE-bench rollout 先执行：

```text
git reset --hard <base_commit>
```

随后执行重度 Git scrub：

```text
删除 remotes、branches 和 tags
git checkout --orphan __slime_buggy
git add -A
git commit -m "slime initial bug state"
```

最终 workspace 的 HEAD 是一个没有父提交的合成 commit：

```text
slime initial bug state
```

这会产生以下差异：

| 项目 | 修复前 | 官方 SWE-bench |
|---|---|---|
| HEAD | 无父合成 commit | `base_commit` 上的普通 setup commit |
| `base_commit` | 不再是 HEAD 的祖先 | 是 HEAD 的直接父提交 |
| 过去提交历史 | 被全部删除 | 保留 |
| 过去的版本 tag | 被全部删除 | 保留 |
| 未来提交 | 不可见 | 不可见 |
| 未跟踪文件 | `git add -A` 可能加入合成 commit | setup commit 只提交 tracked changes |

因此，模型看到的仓库不是官方 SWE-bench 给 agent 准备的仓库。具体影响包括：

- Claude Code system prompt 中的 `Recent commits` 不同；
- agent 无法正常使用 `git log`、`git blame` 和历史 diff 理解代码；
- 依赖历史或版本 tag 的仓库行为可能不同；
- 使用官方 Git 状态的评测结果与我们的结果不再是严格同口径。

问题不在于“未来提交是否泄露”：旧逻辑确实阻止了未来提交。问题在于它同时删除了本应保留
的 `base_commit` 和过去历史，清理范围过大。

## 3. 官方逻辑

官方参考实现位于：

`swebench/harness/test_spec/python.py::make_repo_script_list_py`

源码：

<https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/test_spec/python.py#L245-L295>

官方流程的关键步骤是：

| 顺序 | 官方操作 | 目的 |
|---:|---|---|
| 1 | `git clone --single-branch` | 只获取目标分支 |
| 2 | `git reset --hard <base_commit>` | 恢复题目规定的初始代码 |
| 3 | `git remote remove origin` | 防止 agent 获取远端未来历史 |
| 4 | 删除指向目标时间之后提交的 tag | 防止通过 tag 查看未来提交 |
| 5 | 清理 reflog 并执行 `git gc` | 删除其他入口和不可达对象 |
| 6 | 检查 `git log --all` 中没有未来提交 | 将清理结果作为 correctness gate |
| 7 | 创建 `SWE-bench` setup commit | 将环境安装造成的 tracked changes 作为干净基线 |

官方逻辑的核心不是“删除全部 Git 历史”，而是：

> 保留真实 base 和过去历史，只删除能够访问未来提交的入口。

## 4. 本次如何修复

### 4.1 为 SWE-bench 增加独立初始化路径

新增：

```text
build_swebench_git_setup_command(workdir, base_commit)
apply_swebench_git_setup(sb, workdir, base_commit)
```

SWE-bench rollout 不再调用通用的 orphan scrub。SWE-Smith 和 Scale-SWE 继续调用原逻辑，避免
改变其他数据集的既有语义。

### 4.2 恢复并校验真实 base

初始化仍先执行：

```text
git reset --hard <base_commit>
```

随后重新解析 `base_commit`，并要求当前 HEAD 必须等于该 commit。reset 或校验失败时，
workspace 初始化直接失败，不会继续训练，也不会静默退回 orphan commit。

### 4.3 模拟官方 single-branch clone

AGS runtime 使用预构建镜像，不会像官方脚本一样现场执行 `git clone --single-branch`。因此
本实现显式删除：

- 所有 Git remotes；
- 所有 remote-tracking refs；
- 当前分支之外的本地分支；
- stash、notes、replace 等其他 refs。

当前分支和可安全保留的历史 tag 不删除。若 workspace 处于 detached HEAD，则不会人为创建
新的本地分支。

### 4.4 删除未来 tag 并执行 correctness gate

对每个 tag，解析它最终指向的 commit。若该 commit 的时间晚于 `base_commit`，删除该 tag。
之后执行：

```text
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

最后检查所有仍可访问的 refs。只要还有时间晚于 `base_commit` 的 commit，初始化立即失败。
因此“未来历史不可见”不再只依赖清理命令是否看起来执行过，而是有明确的结果检查。

### 4.5 创建官方风格的 setup commit

清理完成后，在真实 `base_commit` 上创建普通子提交：

```text
message:         SWE-bench
author name:     SWE-bench
author email:    setup@swebench.config
committer name:  SWE-bench
committer email: setup@swebench.config
parent:          base_commit
```

同时：

- 显式设置 `commit.gpgSign=false`；
- 使用 `--no-gpg-sign`，避免环境配置注入签名；
- 使用 `git commit -am`，不把 `.harness/` 等 untracked runtime 文件加入 commit；
- 创建后再次检查 `HEAD^ == base_commit`。

## 5. 确定性增强

官方脚本使用执行时的当前时间创建 `SWE-bench` commit。同一题在不同时间初始化时，setup
commit hash 因此可能不同；Claude Code 又会把 commit hash 写入 system prompt。

为避免重新引入 token-exact resume 已修复过的 hash 漂移，本实现使用 `base_commit` 的
committer 时间，同时固定 author 和 committer 时间：

```text
GIT_AUTHOR_DATE=<base_commit committer date>
GIT_COMMITTER_DATE=<base_commit committer date>
```

结果是：

- 相同 `base_commit`、相同 tree 和相同 message 产生相同 setup HEAD；
- 不同 base 或不同 tree 仍产生不同 HEAD；
- setup commit 仍是 `base_commit` 的普通子提交；
- 真实项目已有 commit 的作者、时间和签名不会被改写。

这是相对官方脚本唯一有意保留的可复现性增强。

## 6. 修复前后对比

| 项目 | 修复前 | 修复后 |
|---|---|---|
| SWE-bench rollout HEAD | `slime initial bug state` | `SWE-bench` |
| HEAD parent | 无 | 真实 `base_commit` |
| base 祖先历史 | 删除 | 保留 |
| 过去 tag | 全部删除 | 保留 |
| 未来 tag | 删除 | 删除 |
| remote / remote refs | 删除 | 删除 |
| 额外本地分支 | 删除 | 删除 |
| 未来提交检查 | 无结果校验 | 仍可访问即失败 |
| untracked 文件 | 可能被 `git add -A` 提交 | 不提交 |
| setup commit hash | 固定 orphan 时间后可重复 | 使用 base 时间，可重复 |
| 初始化失败 | scrub 错误可能被忽略 | 直接返回失败 |

## 7. 影响范围

| 路径 | 当前行为 |
|---|---|
| SWE-bench Classic rollout | 使用本次官方对齐逻辑 |
| SWE-Gym rollout | 属于 SWE-bench Classic，使用本次逻辑 |
| SWE-bench Verified rollout | 属于 SWE-bench Classic，使用本次逻辑 |
| SWE-bench 独立 eval sandbox | reset 到 base；不向 agent 暴露，保留镜像 tags/graph |
| SWE-Smith | 继续使用 `__slime_buggy` orphan commit |
| Scale-SWE | 继续使用 `__slime_buggy` orphan commit |
| ReBench / Generic | 不执行上述 Git scrub |

该修复只影响修改后新建的 workspace。已完成的 rollout、训练 checkpoint 和评测结果不会被
追溯修改。要使用新逻辑，必须重新提交任务。

## 8. 验证

真实 Git 回归测试构造了：

```text
ancestor -> base_commit -> future_commit
```

并额外创建：

- 指向过去提交的 `v1.0`、`v1.1`；
- 指向未来提交的 `v2.0`；
- 指向未来提交的额外本地分支；
- 指向未来提交的 remote-tracking ref；
- 一个名为 `origin` 的 remote。

执行新初始化后验证：

- `HEAD^ == base_commit`；
- `ancestor` 仍是 HEAD 的祖先；
- setup tree 与 base tree 一致；
- `v1.0` 和 `v1.1` 保留；
- `v2.0`、未来分支和 remote ref 删除；
- remote 列表为空；
- `future_commit` 不再出现在 `git rev-list --all`；
- Git identity、GPG 配置和 commit 时间正确；
- 两个外部环境时间不同的相同仓库得到相同 setup HEAD。

测试结果：

```text
tests/claudecode_ags/test_workspace_git_scrub.py: 3 passed
tests/claudecode_ags/test_workspace_init.py:     20 passed
合计：                                             23 passed
```

另外通过：

- Python 3.11 `py_compile`；
- `git diff --check`。

## 9. 代码与测试位置

- 实现：`examples/claudecode_ags/workspace_init.py`
- Git 拓扑测试：`tests/claudecode_ags/test_workspace_git_scrub.py`
- workspace 路由测试：`tests/claudecode_ags/test_workspace_init.py`

## 10. 与旧文档的关系

`2026-07-15-sandbox-commit-hash-drift.md` 记录的是旧 orphan commit 的确定性修复。其历史问题、
证据和 token 差异仍然有效，但从本次修改开始：

- SWE-bench Classic 不再创建该 orphan commit；
- 旧的固定 orphan 时间仍用于 SWE-Smith 和 Scale-SWE；
- SWE-bench 改为固定普通 `SWE-bench` setup commit 的时间。

