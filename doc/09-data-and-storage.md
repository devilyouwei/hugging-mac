# 数据与存储

状态：`Proposed`

## 数据类别

| 类别 | 示例 | 默认是否提交 |
|---|---|---|
| 配置 | manifest、recipe、schema | 是 |
| 小型 fixture | 测试图片、短文本 | 审核后 |
| 模型资源 | 权重、转换产物 | 否 |
| 数据集 | 原始/处理后数据 | 否 |
| 运行产物 | 预测、报告、checkpoint | 否 |
| 元数据 | job、lease、索引 | 否 |

## 路径抽象

所有路径通过配置解析，默认位于项目目录，未来支持用户级缓存。SDK 返回 artifact/resource reference，
不把机器绝对路径写入可分享的结果。

## 元数据存储

首阶段优先 SQLite + 文件系统：

- SQLite 保存 job、实例元数据、资源索引、lease 与实验索引；
- 大对象保存在内容寻址文件系统；
- 数据库只保存 digest 和相对/逻辑引用。

repository interface 隔离存储实现，业务层不直接散落 SQL。

## 数据集

dataset snapshot 记录来源、版本、split、过滤和转换摘要。敏感或受限数据集不复制进 artifact；
运行记录只保留不可逆摘要和用户可理解的来源标识。

## 原子性与恢复

- 下载、转换、checkpoint 使用 staging → 校验 → commit；
- job 状态和最终 artifact 引用在同一提交边界更新；
- 启动时扫描孤儿 staging 和过期 lease；
- 清理有 dry-run、配额和审计记录。
