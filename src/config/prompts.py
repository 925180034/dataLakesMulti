from typing import Dict

# Prompt模板集合
PROMPT_TEMPLATES: Dict[str, str] = {
    
    # 规划器智能体Prompt
    "planner_strategy_decision": """
你是一个专业的任务规划器。分析用户请求，判断应该使用哪种策略来处理数据发现任务。

用户请求: "{user_query}"

查询信息:
- 查询表数量: {table_count}
- 指定列数量: {column_count}

策略选择:
1. **Bottom-Up (自下而上)**: 适用于寻找可连接的表、精确连接、或当用户明确要求"匹配列"时使用
2. **Top-Down (自上而下)**: 适用于寻找可合并的表、相似主题、或进行一般性探索时使用

请仅返回策略编号 (1 或 2):
""",

    "planner_final_report": """
基于以下结果生成最终报告:

策略: {strategy}
处理步骤: {steps}
匹配结果: {results}

请生成一份简洁、用户友好的报告，包括:
1. 发现的相关表
2. 匹配的具体原因
3. 建议的后续操作
""",

    # 列发现智能体Prompt
    "column_discovery": """
你是一个列匹配专家。给定一个查询列，在数据湖中找到最相关的匹配列。

查询列信息:
- 表名: {table_name}
- 列名: {column_name}
- 数据类型: {data_type}
- 样本值: {sample_values}

候选列信息:
{candidates}

请分析每个候选列与查询列的相似度，考虑:
1. 语义相似性 (列名含义)
2. 数据类型匹配
3. 值分布相似性

为每个匹配提供置信度评分 (0-1) 和匹配原因。

输出格式:
```json
{{
  "matches": [
    {{
      "target_column": "table.column",
      "confidence": 0.95,
      "reason": "列名语义相似且数据类型匹配"
    }}
  ]
}}
```
""",

    # 表聚合智能体Prompt
    "table_aggregation": """
你是一个表评分专家。基于列匹配结果，评估每个目标表的相关性。

列匹配结果:
{column_matches}

请为每个目标表计算综合评分 (0-100)，考虑因素:
1. 匹配列的数量
2. 匹配列的平均置信度  
3. 是否包含关键列 (如ID、主键)
4. 匹配列的重要性

输出格式:
```json
{{
  "ranked_tables": [
    {{
      "table_name": "table_A",
      "score": 95.0,
      "evidence_columns": ["col1", "col2"],
      "reason": "包含2个高置信度匹配列，其中包括关键ID列"
    }}
  ]
}}
```
""",

    # 表发现智能体Prompt
    "table_discovery": """
你是一个表语义发现专家。基于查询表的特征，找到数据湖中语义相似的表。

查询表信息:
- 表名: {table_name}
- 列结构: {columns}
- 样本数据: {sample_data}

请分析查询表的主题和用途，然后在候选表中找到最相关的表。

候选表信息:
{candidates}

考虑因素:
1. 表的主题领域相似性
2. 列结构的整体相似性
3. 数据内容的语义相关性

输出前10个最相关的表名。
""",

    # 表匹配智能体Prompt  
    "table_matching": """
你是一个表间匹配专家。详细比较两个表，找出所有匹配的列对。

源表: {source_table}
目标表: {target_table}

源表列信息:
{source_columns}

目标表列信息:
{target_columns}

请逐一比较每对列的相似性，包括:
1. 列名语义相似性
2. 数据类型兼容性
3. 值分布相似性

输出格式:
```json
{{
  "table_pair": "{source_table} -> {target_table}",
  "column_matches": [
    {{
      "source_column": "col1",
      "target_column": "col2", 
      "confidence": 0.9,
      "match_type": "semantic",
      "reason": "列名语义高度相似"
    }}
  ],
  "overall_similarity": 0.85
}}
```
"""
}


def get_prompt_template(template_name: str) -> str:
    """获取Prompt模板"""
    if template_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template: {template_name}")
    return PROMPT_TEMPLATES[template_name]


def format_prompt(template_name: str, **kwargs) -> str:
    """格式化Prompt模板"""
    template = get_prompt_template(template_name)
    return template.format(**kwargs)