# 架构决策记录

状态：`Proposed`

ADR 用于记录难以逆转、跨模块或影响公共契约的决定。小型实现细节不需要 ADR。

## 需要优先决策

1. SDK 是单包多 extra，还是 runtime adapter 分包；
2. 模型 worker 的进程模型与 IPC；
3. manifest 格式、schema 版本与信任边界；
4. 本地元数据使用 SQLite 的事务边界；
5. Web 前端技术栈；
6. 仓库许可证和第三方模型许可证展示方式。

## 命名

```text
doc/adr/0001-short-decision-title.md
```

编号不复用。被替代的 ADR 保留并标为 `Superseded by ADR-NNNN`。

## 模板

```markdown
# ADR-NNNN：标题

状态：Proposed | Accepted | Rejected | Superseded
日期：YYYY-MM-DD

## 背景

触发决定的约束和问题。

## 决定

选择了什么。

## 备选方案

考虑过什么，以及未选择的原因。

## 后果

正面、负面、迁移和验证影响。
```
