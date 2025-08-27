#!/usr/bin/env python3
"""
NLCTables Prompt优化器
利用NLCTables的自然语言查询特性，优化LLM匹配prompt
"""

import json
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class NLCTablesPromptOptimizer:
    """NLCTables专门的Prompt优化器"""
    
    def __init__(self):
        self.query_patterns = {
            'join': [
                r'join.*with', r'combine.*tables', r'merge.*data',
                r'link.*to', r'connect.*between', r'relate.*to'
            ],
            'union': [
                r'similar.*to', r'like.*this', r'same.*as',
                r'equivalent.*to', r'comparable.*with', r'union.*with'
            ]
        }
        
        self.entity_extractors = {
            'table_references': r'(?:table|data|dataset|source)\s+(?:about|for|containing|with)\s+(\w+)',
            'column_mentions': r'(?:column|field|attribute|property)\s+(?:named|called|like)\s+(\w+)',
            'data_types': r'(?:containing|with|having)\s+(\w+)\s+(?:data|values|information)',
            'relationships': r'(?:related|connected|linked|associated)\s+(?:to|with)\s+(\w+)'
        }
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """分析查询意图"""
        query_lower = query.lower()
        
        # 检测任务类型
        task_type = 'unknown'
        for task, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    task_type = task
                    break
            if task_type != 'unknown':
                break
        
        # 提取实体
        entities = {
            'tables': [],
            'columns': [],
            'data_types': [],
            'relationships': []
        }
        
        for entity_type, pattern in self.entity_extractors.items():
            matches = re.findall(pattern, query_lower)
            if matches:
                key = entity_type.split('_')[0] + 's'  # 简化key名
                entities[key] = list(set(matches))
        
        # 分析查询复杂度
        complexity = self._assess_complexity(query)
        
        # 识别关键需求
        requirements = self._extract_requirements(query_lower)
        
        return {
            'task_type': task_type,
            'entities': entities,
            'complexity': complexity,
            'requirements': requirements,
            'original_query': query
        }
    
    def _assess_complexity(self, query: str) -> str:
        """评估查询复杂度"""
        words = query.split()
        if len(words) < 10:
            return 'simple'
        elif len(words) < 20:
            return 'medium'
        else:
            return 'complex'
    
    def _extract_requirements(self, query: str) -> List[str]:
        """提取关键需求"""
        requirements = []
        
        # 检查特定需求
        if 'exact' in query or 'precise' in query:
            requirements.append('exact_match')
        if 'similar' in query or 'like' in query:
            requirements.append('similarity_match')
        if 'all' in query:
            requirements.append('comprehensive')
        if 'best' in query or 'top' in query:
            requirements.append('ranking_important')
        if 'related' in query:
            requirements.append('relationship_focus')
        if 'same type' in query or 'same kind' in query:
            requirements.append('type_consistency')
        
        return requirements
    
    def generate_optimized_prompt(self, query_table: Dict, candidate_table: Dict,
                                 query_intent: Dict, task_type: str) -> str:
        """生成优化的prompt"""
        
        # 基础prompt模板
        if task_type == 'join':
            base_prompt = self._generate_join_prompt(query_table, candidate_table, query_intent)
        else:  # union
            base_prompt = self._generate_union_prompt(query_table, candidate_table, query_intent)
        
        # 添加自然语言查询上下文
        if query_intent.get('original_query'):
            context_prompt = f"""
用户查询意图：{query_intent['original_query']}

请特别注意：
- 查询关注的实体：{', '.join(query_intent['entities'].get('tables', []))}
- 提到的列：{', '.join(query_intent['entities'].get('columns', []))}
- 数据类型：{', '.join(query_intent['entities'].get('data_types', []))}
- 关系：{', '.join(query_intent['entities'].get('relationships', []))}

基于以上用户意图，请评估这两个表的匹配程度。
"""
            base_prompt = context_prompt + "\n" + base_prompt
        
        # 根据需求添加特定指导
        if 'exact_match' in query_intent.get('requirements', []):
            base_prompt += "\n注意：用户需要精确匹配，请严格评估。"
        elif 'similarity_match' in query_intent.get('requirements', []):
            base_prompt += "\n注意：用户寻找相似的表，语义相似性很重要。"
        
        if 'relationship_focus' in query_intent.get('requirements', []):
            base_prompt += "\n重点：评估表之间的关系和连接可能性。"
        
        return base_prompt
    
    def _generate_join_prompt(self, query_table: Dict, candidate_table: Dict,
                             query_intent: Dict) -> str:
        """生成JOIN任务的优化prompt"""
        
        # 提取关键信息
        query_cols = {col['name']: col.get('type', 'unknown') 
                     for col in query_table.get('columns', [])}
        candidate_cols = {col['name']: col.get('type', 'unknown')
                         for col in candidate_table.get('columns', [])}
        
        # 找出潜在的join keys
        common_patterns = ['id', 'key', 'code', 'number', 'name']
        potential_join_keys = []
        
        for q_col in query_cols:
            for c_col in candidate_cols:
                # 检查是否可能是join key
                for pattern in common_patterns:
                    if pattern in q_col.lower() and pattern in c_col.lower():
                        potential_join_keys.append((q_col, c_col))
                        break
        
        prompt = f"""评估表连接可能性：

查询表：{query_table.get('table_name', 'unknown')}
候选表：{candidate_table.get('table_name', 'unknown')}

潜在连接键：
{chr(10).join([f'- {q} <-> {c}' for q, c in potential_join_keys[:5]]) if potential_join_keys else '无明显连接键'}

列匹配情况：
- 查询表列数：{len(query_cols)}
- 候选表列数：{len(candidate_cols)}
- 相似列名：{len(set(query_cols.keys()) & set(candidate_cols.keys()))}

请基于以下标准评分（0.0-1.0）：
1. 存在可连接的键（外键关系）
2. 数据类型兼容性
3. 业务逻辑相关性
4. 值域重叠程度

仅返回一个0到1之间的浮点数作为匹配分数。
"""
        return prompt
    
    def _generate_union_prompt(self, query_table: Dict, candidate_table: Dict,
                              query_intent: Dict) -> str:
        """生成UNION任务的优化prompt"""
        
        # 分析schema相似性
        query_cols = set(col['name'].lower() for col in query_table.get('columns', []))
        candidate_cols = set(col['name'].lower() for col in candidate_table.get('columns', []))
        
        common_cols = query_cols & candidate_cols
        similarity_ratio = len(common_cols) / max(len(query_cols), len(candidate_cols)) if query_cols or candidate_cols else 0
        
        # 分析值的语义相似性
        query_samples = []
        candidate_samples = []
        
        for col in query_table.get('columns', [])[:3]:  # 取前3列的样本
            if 'sample_values' in col:
                query_samples.extend(col['sample_values'][:2])
        
        for col in candidate_table.get('columns', [])[:3]:
            if 'sample_values' in col:
                candidate_samples.extend(col['sample_values'][:2])
        
        prompt = f"""评估表的语义相似性（UNION兼容性）：

查询表：{query_table.get('table_name', 'unknown')}
候选表：{candidate_table.get('table_name', 'unknown')}

Schema相似度：{similarity_ratio:.1%}
共同列：{', '.join(list(common_cols)[:5]) if common_cols else '无'}

查询表样本值：{', '.join(query_samples[:5])}
候选表样本值：{', '.join(candidate_samples[:5])}

请基于以下标准评分（0.0-1.0）：
1. 表示相同类型的实体或概念
2. 列结构可以对齐（允许子集）
3. 数据粒度和范围相似
4. 业务含义相关

考虑UNION操作的语义：这两个表是否代表相似的数据，可以合并？

仅返回一个0到1之间的浮点数作为匹配分数。
"""
        return prompt
    
    def enhance_llm_matching(self, query_table: Dict, candidate_table: Dict,
                            natural_language_query: str = None,
                            task_type: str = 'join') -> Tuple[str, Dict]:
        """增强LLM匹配 - 主要接口"""
        
        # 分析查询意图
        query_intent = {}
        if natural_language_query:
            query_intent = self.analyze_query_intent(natural_language_query)
            
            # 如果从查询中检测到任务类型，使用它
            if query_intent.get('task_type') != 'unknown':
                task_type = query_intent['task_type']
        
        # 生成优化的prompt
        optimized_prompt = self.generate_optimized_prompt(
            query_table, candidate_table, query_intent, task_type
        )
        
        # 返回prompt和元数据
        metadata = {
            'task_type': task_type,
            'query_intent': query_intent,
            'has_nlq': natural_language_query is not None,
            'prompt_type': 'nlc_optimized'
        }
        
        return optimized_prompt, metadata
    
    def batch_enhance(self, query_table: Dict, candidate_tables: List[Dict],
                     natural_language_query: str = None,
                     task_type: str = 'join') -> List[Tuple[str, Dict]]:
        """批量增强多个候选表的匹配"""
        results = []
        
        # 一次性分析查询意图
        query_intent = {}
        if natural_language_query:
            query_intent = self.analyze_query_intent(natural_language_query)
            if query_intent.get('task_type') != 'unknown':
                task_type = query_intent['task_type']
        
        # 为每个候选表生成优化的prompt
        for candidate in candidate_tables:
            prompt = self.generate_optimized_prompt(
                query_table, candidate, query_intent, task_type
            )
            metadata = {
                'task_type': task_type,
                'query_intent': query_intent,
                'candidate_table': candidate.get('table_name', 'unknown')
            }
            results.append((prompt, metadata))
        
        return results


# 集成到LLM Matcher的辅助函数
def integrate_with_llm_matcher(llm_matcher_instance, dataset_type: str):
    """将NLCTables优化集成到现有的LLM Matcher"""
    
    if dataset_type != 'nlctables':
        return llm_matcher_instance  # 不修改非NLCTables的matcher
    
    # 创建优化器
    optimizer = NLCTablesPromptOptimizer()
    
    # 保存原始的prompt生成方法
    original_generate = llm_matcher_instance.generate_prompt
    
    # 创建增强版本
    def enhanced_generate_prompt(query_table, candidate_table, task_type='join'):
        # 检查是否有自然语言查询
        nlq = query_table.get('natural_language_query') or query_table.get('query_text')
        
        if nlq:
            # 使用优化器
            prompt, metadata = optimizer.enhance_llm_matching(
                query_table, candidate_table, nlq, task_type
            )
            logger.debug(f"使用NLCTables优化prompt，意图：{metadata.get('query_intent', {}).get('task_type')}")
            return prompt
        else:
            # 回退到原始方法
            return original_generate(query_table, candidate_table, task_type)
    
    # 替换方法
    llm_matcher_instance.generate_prompt = enhanced_generate_prompt
    
    return llm_matcher_instance


if __name__ == '__main__':
    # 测试代码
    optimizer = NLCTablesPromptOptimizer()
    
    # 测试查询意图分析
    test_queries = [
        "Find tables that can be joined with the customer table",
        "Show me tables similar to product inventory",
        "I need tables containing user data that are related to orders",
        "Get all tables with exact same columns as sales_report"
    ]
    
    for query in test_queries:
        intent = optimizer.analyze_query_intent(query)
        print(f"\n查询：{query}")
        print(f"意图：{intent}")
    
    # 测试prompt生成
    test_query_table = {
        'table_name': 'customers',
        'columns': [
            {'name': 'customer_id', 'type': 'int'},
            {'name': 'name', 'type': 'string'},
            {'name': 'email', 'type': 'string'}
        ]
    }
    
    test_candidate_table = {
        'table_name': 'orders',
        'columns': [
            {'name': 'order_id', 'type': 'int'},
            {'name': 'customer_id', 'type': 'int'},
            {'name': 'total', 'type': 'float'}
        ]
    }
    
    nlq = "Find tables that can be joined with the customer table"
    prompt, metadata = optimizer.enhance_llm_matching(
        test_query_table, test_candidate_table, nlq, 'join'
    )
    
    print(f"\n生成的优化Prompt：")
    print(prompt)
    print(f"\n元数据：")
    print(metadata)