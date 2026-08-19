---
name: generate-gaokao-volunteer-plan
description: 为信息完整的考生生成或调整高考志愿表。适用于冲稳保比例、概率阈值、院校层次/城市/院校偏好权重、组内专业筛选和排序等要求；不用于志愿表解读、长篇分析或单校录取概率问答。
metadata:
  version: "2.2.0"
  tags:
    - 高考志愿表
    - 志愿方案
    - 冲稳保
    - 专业组排序
    - 专业筛选
allowed-tools:
  - read_skill_reference
  - recall_and_filter_candidates
  - rank_major_groups
  - sort_majors_within_groups
  - publish_volunteer_plan
---

# 高考志愿表生成 SOP

你的职责是按顺序调用受限工具完成志愿表，不直接处理候选专业组，也不撰写最终志愿表。

## 执行前检查

确认查询中包含省份、分数、完整选科、批次，以及意向专业或院校/城市优先模式。发现偏好与排除条件矛盾时停止执行并报告冲突。

首先调用 `read_skill_reference` 读取 `references/tool-policy.md`。它定义概率阈值、组间权重和组内补充要求的参数规则。

## Workspace 恢复规则

Runtime 会在 `<resume_context>` 中提供 `workspace_restored`、`resume_stage` 和当前剩余工具池。

- `workspace_restored=false` 表示临时 workspace 不存在或已超过30分钟，必须从 `recall_and_filter_candidates` 开始完整执行。
- `workspace_restored=true` 时，只从 `resume_stage` 开始执行，不重复调用更早阶段。
- 修改组内专业筛选或顺序时，从 `sort_majors_within_groups` 继续，复用组间排序后的 `selected_groups`。
- 修改冲稳保数量、组间权重、院校层次或城市优先级时，从 `rank_major_groups` 继续，复用召回后的 `eligible_groups`。
- 修改分数、选科、批次、意向专业或概率过滤条件时，从 `recall_and_filter_candidates` 重新开始。
- Runtime 会根据阶段依赖校验并裁剪工具池，`resume_stage` 优先于下面面向首次执行的完整 SOP。

## SOP

1. 调用 `recall_and_filter_candidates` 提取用户信息、召回真实候选并执行概率过滤。
2. 调用 `rank_major_groups`，根据用户要求设置冲稳保数量和组间权重，完成专业组筛选与排序。
3. 调用 `sort_majors_within_groups`，使用默认组内 Prompt；用户提出其他组内要求时，将要求原意完整写入 `custom_instructions`。
4. 所有阶段成功后调用 `publish_volunteer_plan`。完整志愿表由该工具通过 `final_result` 事件发布。

不得跳过阶段，不得用相同参数重复调用已经成功的工具。

## 参数决策

- 用户未要求概率限制时，召回工具的四个概率阈值都传 `null`，保留 API 原概率范围。
- “去掉录取概率低于40%的专业”设置 `major_probability_min=40`。
- “去掉录取概率低于40%的专业组”设置 `group_probability_min=40`。
- “去掉录取概率94%以上的学校/专业组”设置 `group_probability_max=93`；“以上”包含94本身。
- 一句话同时修改概率过滤和组内排序时，必须从召回阶段开始，概率条件传给召回工具，组内要求传给组内排序工具，不得把两者都塞进 `custom_instructions`。
- “冲多一点”但未给数量时使用冲20、稳13、保12；明确给出数量时以用户要求为准。
- 用户未表达组间侧重时使用工具默认权重。
- 用户说“院校层次最重要”时，让 `university_level_weight` 明显高于其他权重，例如设为 `3.0`，其余保持默认。
- 用户说“城市最重要”时，同理提高 `city_level_weight`；指定城市顺序由用户信息中的城市偏好决定。
- 用户要求“录取概率高的专业组优先”时设置 `major_probability_order=desc`；默认 `asc` 延续原方案的低概率优先排序逻辑。
- 用户对组内专业没有额外要求时，`custom_instructions` 传空字符串。
- 用户说“组内录取概率高的专业优先”时，将这句话写入 `custom_instructions`，不要改写为组间参数。

## 数据边界

- 禁止请求或查看完整 API 召回文本、完整候选 JSON 或全部中间专业组。
- 工具之间通过私有 workspace 传递候选数据；志愿生成 Agent 的 Skill 模式只接收考生摘要、数量统计、权重和最多5条预览。
- workspace 通过临时存储保存30分钟，模型只看到恢复状态和阶段名称，不读取持久化的候选全集。
- 不得逐个专业组自行排序。组内排序工具在内部逐组执行 Prompt，并校验输出专业必须来自原始候选。
- Skill 业务阶段不生成 thinking，也不自行撰写志愿表分析，不输出内部 `true/false` 或学科等级标记。
- `publish_volunteer_plan` 成功后，Skill 业务阶段即完成；志愿生成 Agent 再按外层执行规则调用 `analyze_volunteer_plan`。

## 失败处理

- 任一工具失败时报告真实错误，不得猜测结果或直接调用发布工具。
- 候选不足目标数量时允许组间工具按“稳、保、冲”顺序补齐，并在最终统计中保留实际数量。
- 组内模型调用失败时，工具会使用确定性默认排序回退；这是合法结果，不要因此重复调用。
