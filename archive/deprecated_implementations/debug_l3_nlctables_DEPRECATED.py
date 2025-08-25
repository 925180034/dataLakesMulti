#!/usr/bin/env python
"""
调试L3层NLCTables处理 - 分析多智能体系统的候选表处理
"""
import json

def analyze_l3_problem():
    """分析L3层为什么返回错误的表类型"""
    
    # 加载数据
    with open('examples/nlctables/join_subset/queries.json', 'r') as f:
        queries = json.load(f)
    with open('examples/nlctables/join_subset/tables.json', 'r') as f:
        tables = json.load(f)
    with open('examples/nlctables/join_subset/ground_truth.json', 'r') as f:
        ground_truth = json.load(f)
    
    # 分析第一个查询
    query = queries[0]
    query_id = query['query_id']
    seed_table = query['seed_table']
    
    print("=== L3层NLCTables问题分析 ===")
    print(f"Query ID: {query_id}")
    print(f"Seed table: {seed_table}")
    print(f"Query text: {query['query_text']}")
    print()
    
    # 获取正确答案
    import re
    match = re.search(r'nlc_join_(\\d+)', query_id)
    if match:
        numeric_key = match.group(1)
        gt_tables = []
        if numeric_key in ground_truth:
            gt_entries = ground_truth[numeric_key]
            gt_tables = [
                entry['table_id'] 
                for entry in gt_entries 
                if entry.get('relevance_score', 0) > 0
            ]
        print(f"正确答案 (Ground Truth): {gt_tables}")
    
    # 分析表的分布
    table_names = [t.get('name', t.get('table_name', '')) for t in tables]
    q_tables = [name for name in table_names if name.startswith('q_table_')]
    dl_tables = [name for name in table_names if name.startswith('dl_table_')]
    
    print(f"\\n表分布:")
    print(f"  q_table_*  (seed tables): {len(q_tables)}")
    print(f"  dl_table_* (target tables): {len(dl_tables)}")
    
    # 找相似的seed表
    seed_pattern = seed_table.replace('q_table_', '')
    similar_seeds = []
    for q_name in q_tables:
        if seed_pattern in q_name or q_name != seed_table:
            # 计算名字相似度
            q_pattern = q_name.replace('q_table_', '')
            if q_pattern.split('_')[0] == seed_pattern.split('_')[0]:  # 同一个数字前缀
                similar_seeds.append(q_name)
    
    print(f"\\n相似的seed表 (LLM可能选择的错误答案):")
    print(f"  {similar_seeds[:5]}")
    
    # 找正确的target表
    correct_targets = []
    for dl_name in dl_tables:
        if seed_pattern in dl_name:
            correct_targets.append(dl_name)
    
    print(f"\\n正确的target表 (应该选择的答案):")
    print(f"  {correct_targets[:5]}")
    
    print(f"\\n=== 问题诊断 ===")
    print("1. 多智能体系统把所有表都当作候选")
    print("2. LLM认为与q_table_118_j1_3最相似的是其他q_table_*")
    print("3. 系统没有被告知只在dl_table_*中搜索答案")
    print("4. 缺少'只返回data lake表'的约束")
    
    print(f"\\n=== 解决方案 ===")
    print("1. 修改多智能体系统，明确指定候选表范围")
    print("2. 在prompt中强调'只从dl_table开头的表中选择'")
    print("3. 过滤掉所有q_table_*表，只传递dl_table_*给LLM")
    print("4. 或者直接禁用L3层，因为L2层的name-based matching已经足够准确")

if __name__ == "__main__":
    analyze_l3_problem()