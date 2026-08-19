# 志愿 Skill 原子工具参数规范

## 跨工具数据流

四个业务工具共享本次 Skill 执行专属的 `VolunteerPlanWorkspace`。上一步的业务数据不由 SubAgent 复制到下一步参数：

```text
recall_and_filter_candidates -> workspace.user_info / eligible_groups
rank_major_groups            -> workspace.selected_groups
sort_majors_within_groups    -> workspace.final_groups
publish_volunteer_plan       -> workspace.plan_text / summary
```

因此，SubAgent 只传递本阶段新增的策略参数。不要请求、回填或转述 `user_info`、候选组、排序后专业组和完整志愿表。

## 1. recall_and_filter_candidates

`query` 必须是完整用户信息。四个概率阈值均可为空，空值表示不做对应过滤。

| 参数 | 含义 |
| --- | --- |
| `group_probability_min/max` | 专业组录取概率上下限 |
| `major_probability_min/max` | 组内专业录取概率上下限 |

上下限参数是闭区间。因而“去掉94%以上的学校”表示保留专业组概率 `<94%`，应传 `group_probability_max=93`；“去掉高于94%的学校”才传 `group_probability_max=94`。

不要把省份、分数、选科或专业名称拆成额外参数；工具会从完整 query 提取并校验。

## 2. rank_major_groups

默认冲稳保为15/15/15，总数最多45。权重都是非负浮点数，值越大表示越重要，不要求总和等于1，工具会自动归一化。

| 权重 | 默认值 | 作用 |
| --- | ---: | --- |
| `university_level_weight` | 0.9 | C9、985、211、双一流、省重点层次 |
| `city_level_weight` | 0.4 | 一线、新一线、省会等城市层次 |
| `university_preference_weight` | 1.2 | 用户明确指定的院校及顺序 |
| `city_preference_weight` | 0.8 | 用户明确指定的城市及顺序 |
| `major_probability_weight` | 1.0 | 意向专业平均录取概率 |
| `subject_grade_weight` | 0.7 | 专业相关学科评估等级 |
| `major_relevance_weight` | 1.0 | 组内意向专业占比 |

`major_probability_order=asc` 表示概率较低的组在同一冲稳保区间中优先，`desc` 表示概率较高的优先。

用户说某因素“最重要”时，应把对应权重调整到高于其他所有权重；没有提到的权重保留默认值，不要全部重算。

## 3. sort_majors_within_groups

默认规则来自旧版志愿生成 Prompt：专业优先模式下意向专业在前、相关专业居中、低相关专业在后；城市/院校优先模式下按低、中、高录取概率排列。

`custom_instructions` 只放用户明确提出的组内要求，并在与默认规则冲突时优先。例如：

```text
把软件工程放在计算机科学与技术之前；同等相关度时录取概率高的优先。
```

不得在这里放冲稳保数量、专业组概率阈值或院校层次权重。

## 4. publish_volunteer_plan

无参数。前三个业务工具都成功后才能调用。该工具发布完整表格并只向 SubAgent 返回有界统计摘要。
