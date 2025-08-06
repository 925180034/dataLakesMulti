from typing import Dict

# English Prompt Templates Collection
PROMPT_TEMPLATES: Dict[str, str] = {
    
    # Planner Agent Prompt
    "planner_strategy_decision": """
You are a professional task planner specializing in data discovery. Analyze the user request and determine the optimal strategy for processing the data discovery task.

User Request: "{user_query}"

Query Information:
- Number of query tables: {table_count}
- Number of specified columns: {column_count}

Strategy Selection:
1. **Bottom-Up Strategy**: Suitable for finding joinable tables, precise joins, or when the user explicitly requests "column matching"
2. **Top-Down Strategy**: Suitable for finding union-compatible tables, similar topics, or general exploratory queries

Please return only the strategy number (1 or 2):
""",

    "planner_final_report": """
Generate a comprehensive final report based on the following results:

Strategy: {strategy}
Processing Steps: {steps}
Matching Results: {results}

Please generate a concise, user-friendly report including:
1. Discovered relevant tables
2. Specific reasons for matches
3. Recommended follow-up actions
""",

    # Column Discovery Agent Prompt
    "column_discovery": """
You are a column matching expert. Given a query column, identify the most relevant matching columns in the data lake.

Query Column Information:
- Table Name: {table_name}
- Column Name: {column_name}
- Data Type: {data_type}
- Sample Values: {sample_values}

Candidate Column Information:
{candidates}

Please analyze the similarity between each candidate column and the query column, considering:
1. Semantic similarity (column name semantics)
2. Data type compatibility
3. Value distribution similarity

Provide a confidence score (0-1) and matching rationale for each match.

Output Format:
```json
{{
  "matches": [
    {{
      "target_column": "table.column",
      "confidence": 0.95,
      "reason": "High semantic similarity in column names with compatible data types"
    }}
  ]
}}
```
""",

    # Table Aggregation Agent Prompt
    "table_aggregation": """
You are a table scoring expert. Based on column matching results, evaluate the relevance of each target table.

Column Matching Results:
{column_matches}

Please calculate a comprehensive score (0-100) for each target table, considering:
1. Number of matched columns
2. Average confidence of matched columns
3. Presence of key columns (e.g., ID, primary key)
4. Importance of matched columns

Output Format:
```json
{{
  "ranked_tables": [
    {{
      "table_name": "table_A",
      "score": 95.0,
      "evidence_columns": ["col1", "col2"],
      "reason": "Contains 2 high-confidence matched columns, including critical ID column"
    }}
  ]
}}
```
""",

    # Table Discovery Agent Prompt
    "table_discovery": """
You are a table semantic discovery expert. Based on the query table characteristics, identify semantically similar tables in the data lake.

Query Table Information:
- Table Name: {table_name}
- Column Structure: {columns}
- Sample Data: {sample_data}

Please analyze the query table's domain and purpose, then identify the most relevant tables from the candidates.

Candidate Table Information:
{candidates}

Considerations:
1. Domain similarity
2. Overall column structure similarity
3. Semantic relevance of data content

Return the top 10 most relevant table names.
""",

    # Table Matching Agent Prompt  
    "table_matching": """
You are an inter-table matching expert. Perform detailed comparison between two tables to identify all matching column pairs.

Source Table: {source_table}
Target Table: {target_table}

Source Table Column Information:
{source_columns}

Target Table Column Information:
{target_columns}

Please compare each column pair for similarity, including:
1. Column name semantic similarity
2. Data type compatibility
3. Value distribution similarity

Output Format:
```json
{{
  "table_pair": "{source_table} -> {target_table}",
  "column_matches": [
    {{
      "source_column": "col1",
      "target_column": "col2", 
      "confidence": 0.9,
      "match_type": "semantic",
      "reason": "High semantic similarity in column names"
    }}
  ],
  "overall_similarity": 0.85
}}
```
"""
}


def get_prompt_template(template_name: str) -> str:
    """Retrieve prompt template by name"""
    if template_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template: {template_name}")
    return PROMPT_TEMPLATES[template_name]


def format_prompt(template_name: str, **kwargs) -> str:
    """Format prompt template with provided parameters"""
    template = get_prompt_template(template_name)
    return template.format(**kwargs)