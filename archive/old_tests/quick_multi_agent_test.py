#!/usr/bin/env python
"""
快速多智能体系统测试 - 验证核心功能
Quick Multi-Agent System Test - Verify Core Functions
"""

import json
import time
import numpy as np
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_multi_agent_simple():
    """简单的多智能体测试"""
    
    print("\n" + "="*70)
    print("🚀 MULTI-AGENT SYSTEM - QUICK TEST")
    print("="*70)
    
    # 加载数据
    tables_file = 'examples/final_subset_tables.json'
    with open(tables_file, 'r') as f:
        tables = json.load(f)
    
    print(f"📊 Loaded {len(tables)} tables")
    
    # 加载查询
    queries_file = 'examples/separated_datasets/join_subset/queries_filtered.json'
    ground_truth_file = 'examples/separated_datasets/join_subset/ground_truth_transformed.json'
    
    with open(queries_file, 'r') as f:
        queries = json.load(f)[:10]  # 只测试10个查询
    
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    print(f"📋 Testing {len(queries)} JOIN queries")
    print()
    
    # 简单的多Agent处理流程
    results = []
    times = []
    
    for i, query in enumerate(queries):
        start_time = time.time()
        
        query_table_name = query['query_table']
        query_column = query.get('query_column', '')
        
        # 查找查询表
        query_table = None
        for table in tables:
            if table['table_name'] == query_table_name:
                query_table = table
                break
        
        if not query_table:
            logger.warning(f"Query table {query_table_name} not found")
            continue
        
        # ===== 多Agent协同处理 =====
        
        # 1. PlannerAgent - 制定策略
        strategy = plan_strategy(query['task_type'])
        
        # 2. AnalyzerAgent - 分析表结构
        analysis = analyze_table(query_table)
        
        # 3. SearcherAgent - 搜索候选
        candidates = search_candidates(query_table, tables, strategy, analysis)
        
        # 4. MatcherAgent - 精确匹配
        matches = match_tables(query_table, candidates, strategy)
        
        # 5. AggregatorAgent - 聚合结果
        final_results = aggregate_results(matches)
        
        # 记录时间
        query_time = time.time() - start_time
        times.append(query_time)
        
        # 获取ground truth
        query_key = f"{query_table_name}:{query_column}" if query_column else query_table_name
        expected = ground_truth.get(query_key, [])
        if isinstance(expected, str):
            expected = [expected]
        
        # 评估结果
        predicted = [m['table'] for m in final_results]
        hit = any(p in expected for p in predicted[:10]) if expected else False
        
        results.append({
            'query': query_table_name,
            'predicted': predicted[:5],
            'expected': expected[:5],
            'hit': hit,
            'time': query_time
        })
        
        print(f"Query {i+1}: {query_table_name}")
        print(f"  ⏱️  Time: {query_time:.3f}s")
        print(f"  ✅ Hit: {hit}")
        print(f"  🎯 Predicted: {predicted[:3]}")
        print()
    
    # 计算指标
    hits = sum(1 for r in results if r['hit'])
    avg_time = np.mean(times) if times else 0
    
    print("="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    print(f"  Total Queries: {len(results)}")
    print(f"  Hits: {hits}/{len(results)} ({hits/max(len(results),1)*100:.1f}%)")
    print(f"  Avg Time: {avg_time:.3f}s")
    print(f"  Total Time: {sum(times):.2f}s")
    print()
    
    # 保存结果
    output_file = f"experiment_results/quick_test_{int(time.time())}.json"
    Path("experiment_results").mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'metrics': {
                'hit_rate': hits/max(len(results),1),
                'avg_time': avg_time,
                'total_queries': len(results)
            }
        }, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    print("\n✅ TEST COMPLETED!")
    print("="*70)

# ===== Agent实现 =====

def plan_strategy(task_type):
    """PlannerAgent - 制定策略"""
    if task_type == 'join':
        return {
            'name': 'join_strategy',
            'focus': 'foreign_keys',
            'use_column_match': True,
            'top_k': 100
        }
    else:
        return {
            'name': 'union_strategy',
            'focus': 'schema_similarity',
            'use_column_match': True,
            'top_k': 100
        }

def analyze_table(table):
    """AnalyzerAgent - 分析表结构"""
    analysis = {
        'column_count': len(table['columns']),
        'column_names': [col.get('column_name', col.get('name', '')).lower() for col in table['columns']],
        'column_types': [col.get('data_type', col.get('type', '')) for col in table['columns']],
        'key_columns': []
    }
    
    # 识别关键列
    for col in table['columns']:
        col_name = col.get('column_name', col.get('name', '')).lower()
        if any(key in col_name for key in ['_id', '_key', '_code', '_fk']):
            analysis['key_columns'].append(col_name)
    
    return analysis

def search_candidates(query_table, all_tables, strategy, analysis):
    """SearcherAgent - 搜索候选表"""
    candidates = []
    query_col_count = analysis['column_count']
    query_col_names = set(analysis['column_names'])
    
    for table in all_tables:
        if table['table_name'] == query_table['table_name']:
            continue
        
        # 计算相似度
        score = 0.0
        
        # 列数相似度
        col_count = len(table['columns'])
        if abs(col_count - query_col_count) <= 2:
            score += 0.3
        
        # 列名重叠
        table_col_names = {col.get('column_name', col.get('name', '')).lower() for col in table['columns']}
        overlap = len(query_col_names & table_col_names)
        if overlap > 0:
            score += 0.7 * (overlap / max(len(query_col_names), len(table_col_names)))
        
        if score > 0.2:
            candidates.append({
                'table': table['table_name'],
                'score': score,
                'columns': len(table['columns'])
            })
    
    # 排序并返回Top-K
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:strategy['top_k']]

def match_tables(query_table, candidates, strategy):
    """MatcherAgent - 精确匹配验证"""
    matches = []
    
    for candidate in candidates[:30]:  # 最多处理30个
        # 简单的规则验证
        if candidate['score'] > 0.8:
            match_score = candidate['score']
        elif candidate['score'] > 0.5:
            match_score = candidate['score'] * 0.9
        else:
            match_score = candidate['score'] * 0.8
        
        matches.append({
            'table': candidate['table'],
            'score': match_score,
            'method': 'rule_based'
        })
    
    return matches

def aggregate_results(matches):
    """AggregatorAgent - 聚合结果"""
    # 排序
    sorted_matches = sorted(matches, key=lambda x: x['score'], reverse=True)
    
    # 去重
    seen = set()
    unique_matches = []
    for match in sorted_matches:
        if match['table'] not in seen:
            seen.add(match['table'])
            unique_matches.append(match)
    
    return unique_matches[:10]

if __name__ == "__main__":
    test_multi_agent_simple()