"""
Dynamic Prompt Management for Multi-Agent System
支持动态参数的提示词管理
"""

def get_dynamic_matcher_prompt(dynamic_params=None):
    """
    Generate MatcherAgent prompt with dynamic parameters
    
    Args:
        dynamic_params: Dictionary containing dynamic thresholds and parameters
        
    Returns:
        System prompt with dynamic values
    """
    params = dynamic_params or {}
    
    # 获取动态参数
    llm_confidence_threshold = params.get('llm_confidence_threshold', 0.7)
    rule_high_threshold = params.get('rule_high_threshold', 0.9)
    rule_medium_threshold = params.get('rule_medium_threshold', 0.7)
    rule_low_threshold = params.get('rule_low_threshold', 0.5)
    
    # 根据任务类型调整描述
    task_type = params.get('task_type', 'join')
    
    if task_type == 'union':
        confidence_description = f"""
Matching confidence thresholds (UNION - Optimized for Recall):
- High confidence: scores > {rule_high_threshold}
- Medium confidence: scores > {rule_medium_threshold}
- Low confidence: scores > {rule_low_threshold}
- Minimum threshold: {llm_confidence_threshold} (very low to maximize recall)

For UNION operations, we prioritize recall over precision:
- Accept more candidates to avoid missing potential matches
- Use relaxed thresholds to capture semantic similarities
- Consider partial schema overlap as acceptable
"""
    else:
        confidence_description = f"""
Matching confidence thresholds (JOIN - Balanced):
- High confidence: scores > {rule_high_threshold}
- Medium confidence: scores > {rule_medium_threshold}
- Low confidence: scores > {rule_low_threshold}
- Minimum threshold: {llm_confidence_threshold}

For JOIN operations, we balance precision and recall:
- Require at least one strong join key match
- Consider both exact and fuzzy column matches
- Validate cardinality compatibility
"""
    
    system_prompt = f"""You are an expert data analyst specializing in table matching for data lakes.
Your task is to evaluate whether two tables can be joined or unioned based on their schemas.

For JOIN operations:
- Tables must have at least one column that can serve as a join key
- Look for columns with similar names, types, and value patterns
- Consider foreign key relationships
- Check cardinality compatibility (one-to-one, one-to-many, many-to-many)

For UNION operations:
- Tables should have similar overall structure
- Column names don't need to match exactly, but types should be compatible
- Consider semantic similarity of the data
- Row-level compatibility is important

{confidence_description}

Dynamic matching factors:
- Exact column name match: Add 0.3 to score
- Similar data types: Add 0.2 to score
- Value overlap (for JOINs): Add 0.3 to score
- Schema similarity (for UNIONs): Add 0.2 to score

Current task: {task_type.upper()}
Current confidence threshold: {llm_confidence_threshold}

Provide your response in JSON format:
{{
    "match_score": float (0.0 to 1.0),
    "can_match": boolean,
    "match_type": "join" | "union" | "both" | "none",
    "matching_columns": [
        {{"source": "col1", "target": "col2", "confidence": 0.9, "reason": "exact name and type match"}}
    ],
    "cardinality": "one-to-one" | "one-to-many" | "many-to-many" | "unknown",
    "reasoning": "detailed explanation of your decision",
    "risks": ["risk1", "risk2"],
    "recommendations": ["recommendation1", "recommendation2"]
}}

IMPORTANT: For the current {task_type.upper()} task with threshold {llm_confidence_threshold}, 
be {"more lenient and accept partial matches" if task_type == "union" else "balanced in your evaluation"}."""
    
    return system_prompt


def get_dynamic_planner_prompt(dynamic_params=None):
    """
    Generate PlannerAgent prompt with dynamic parameters
    """
    params = dynamic_params or {}
    
    # 获取动态参数
    task_type = params.get('task_type', 'join')
    vector_top_k = params.get('vector_top_k', 500)
    confidence_threshold = params.get('llm_confidence_threshold', 0.7)
    
    # 根据任务类型调整策略建议
    if task_type == 'union':
        strategy_recommendation = f"""
For UNION tasks (current):
- Use top-down strategy focusing on table-level similarity
- Set lower confidence thresholds (current: {confidence_threshold})
- Recommend {vector_top_k} candidates (adjusted for recall)
- Prioritize semantic similarity over exact matches
"""
    else:
        strategy_recommendation = f"""
For JOIN tasks (current):
- Use bottom-up strategy focusing on column-level matching
- Set balanced confidence thresholds (current: {confidence_threshold})
- Recommend {vector_top_k} candidates
- Prioritize key column matches
"""
    
    system_prompt = f"""You are an intelligent strategy planner for a data lake discovery system.
Your role is to analyze the query task and select the optimal execution strategy.

Strategy Options:
1. Bottom-Up: Start from column-level matching, aggregate to tables
   - Best for JOIN operations
   - High precision but slower
   - Use when column matching is critical

2. Top-Down: Start from table-level similarity, then verify details
   - Best for UNION operations  
   - Faster but may miss subtle matches
   - Use when overall structure matters more

3. Hybrid: Combine both approaches adaptively
   - Best for complex or unknown queries
   - Balanced performance and accuracy

{strategy_recommendation}

Current system parameters:
- Vector search top_k: {vector_top_k}
- Confidence threshold: {confidence_threshold}
- Task type: {task_type}

Provide your response in JSON format:
{{
    "strategy_name": "bottom-up" | "top-down" | "hybrid",
    "reasoning": "explanation of your strategy choice",
    "top_k": {vector_top_k},
    "confidence_threshold": {confidence_threshold},
    "optimization_tips": ["tip1", "tip2", "tip3"]
}}"""
    
    return system_prompt


def get_dynamic_searcher_prompt(dynamic_params=None):
    """
    Generate SearcherAgent prompt with dynamic parameters
    """
    params = dynamic_params or {}
    
    # L1层参数
    metadata_min_overlap = params.get('metadata_min_column_overlap', 0.2)
    metadata_max_candidates = params.get('metadata_max_candidates', 500)
    
    # L2层参数
    vector_top_k = params.get('vector_top_k', 500)
    vector_threshold = params.get('vector_similarity_threshold', 0.4)
    
    # L3层参数
    llm_max_candidates = params.get('llm_max_candidates', 100)
    llm_confidence_threshold = params.get('llm_confidence_threshold', 0.15)
    
    system_prompt = f"""You are a search strategy expert for finding candidate tables in a data lake.
Your role is to determine the optimal search approach using a three-layer strategy.

Layer 1 - Metadata Filtering (Current Parameters):
- Minimum column overlap: {metadata_min_overlap}
- Maximum candidates to pass: {metadata_max_candidates}
- Use for quick elimination of non-matches
- Very fast (<10ms) but low precision

Layer 2 - Vector Search (Current Parameters):
- Top-K results: {vector_top_k}
- Similarity threshold: {vector_threshold}
- Good for finding conceptually similar tables
- Moderate speed (10-50ms) with good recall

Layer 3 - Smart Verification (Current Parameters):
- Maximum candidates to verify: {llm_max_candidates}
- Confidence threshold: {llm_confidence_threshold}
- High precision but slower (1-3s per table)

Provide your response in JSON format:
{{
    "search_strategy": {{
        "use_metadata": true,
        "use_vector": true,
        "use_smart": true
    }},
    "metadata_filters": {{
        "min_column_overlap": {metadata_min_overlap},
        "max_candidates": {metadata_max_candidates}
    }},
    "vector_config": {{
        "similarity_threshold": {vector_threshold},
        "max_candidates": {vector_top_k}
    }},
    "llm_config": {{
        "confidence_threshold": {llm_confidence_threshold},
        "max_candidates": {llm_max_candidates}
    }},
    "optimization": "speed" | "accuracy" | "balanced",
    "reasoning": "explanation of search strategy"
}}"""
    
    return system_prompt


def format_prompt_with_params(template: str, dynamic_params: dict = None, **kwargs) -> str:
    """
    Format any prompt template with dynamic parameters
    
    Args:
        template: Prompt template string
        dynamic_params: Dictionary of dynamic parameters
        **kwargs: Additional template variables
        
    Returns:
        Formatted prompt string
    """
    params = dynamic_params or {}
    
    # 合并动态参数和其他参数
    all_params = {**params, **kwargs}
    
    # 提供默认值
    defaults = {
        'llm_confidence_threshold': 0.15,
        'rule_high_threshold': 0.7,
        'rule_medium_threshold': 0.5,
        'rule_low_threshold': 0.3,
        'vector_top_k': 500,
        'metadata_max_candidates': 500,
        'llm_max_candidates': 100,
        'task_type': 'join'
    }
    
    # 合并默认值
    format_args = {**defaults, **all_params}
    
    try:
        return template.format(**format_args)
    except KeyError as e:
        import logging
        logging.warning(f"Missing key in prompt template: {e}")
        # 返回部分格式化的模板
        return template


# 兼容性函数
def format_prompt(prompt_name: str, **kwargs) -> str:
    """
    兼容性函数，保持与原有接口一致
    """
    # 如果有dynamic_params，使用动态版本
    if 'dynamic_params' in kwargs:
        dynamic_params = kwargs.pop('dynamic_params')
        
        if prompt_name == 'matcher':
            return get_dynamic_matcher_prompt(dynamic_params)
        elif prompt_name == 'planner':
            return get_dynamic_planner_prompt(dynamic_params)
        elif prompt_name == 'searcher':
            return get_dynamic_searcher_prompt(dynamic_params)
    
    # 否则使用原有的静态prompts
    from src.config.prompts import get_agent_prompt, format_user_prompt
    
    # 尝试匹配agent名称
    agent_map = {
        'matcher': 'MatcherAgent',
        'planner': 'PlannerAgent',
        'searcher': 'SearcherAgent',
        'analyzer': 'AnalyzerAgent',
        'optimizer': 'OptimizerAgent',
        'aggregator': 'AggregatorAgent'
    }
    
    agent_name = agent_map.get(prompt_name, prompt_name)
    
    if 'template_type' in kwargs:
        template_type = kwargs.pop('template_type')
        return get_agent_prompt(agent_name, template_type)
    else:
        return format_user_prompt(agent_name, **kwargs)