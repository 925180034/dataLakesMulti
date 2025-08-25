#!/usr/bin/env python
"""
NLCTables数据适配器 - 使其能在主系统上运行
基于three_layer_ablation_optimized.py的架构
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
import sys

logger = logging.getLogger(__name__)

class NLCTablesAdapter:
    """将NLCTables数据格式适配到主系统格式"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_nlctables_dataset(self, task_type: str, subset_type: str = 'subset') -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        加载NLCTables数据集并转换为主系统格式
        
        Args:
            task_type: 'join' 或 'union'
            subset_type: 'subset' 或 'complete'
        
        Returns:
            (tables, queries, ground_truth) - 主系统格式
        """
        # NLCTables数据路径
        base_dir = Path(f'examples/nlctables/{task_type}_{subset_type}')
        
        if not base_dir.exists():
            raise FileNotFoundError(f"NLCTables数据集不存在: {base_dir}")
        
        # 加载NLCTables数据
        with open(base_dir / 'tables.json', 'r') as f:
            tables = json.load(f)
        
        with open(base_dir / 'queries.json', 'r') as f:
            nlc_queries = json.load(f)
        
        with open(base_dir / 'ground_truth.json', 'r') as f:
            nlc_ground_truth = json.load(f)
        
        # 转换数据格式
        tables = self._convert_tables(tables, task_type)
        queries = self._convert_queries(nlc_queries, task_type)
        ground_truth = self._convert_ground_truth(nlc_ground_truth, task_type, subset_type)
        
        self.logger.info(f"✅ 加载NLCTables数据: {len(tables)} 表, {len(queries)} 查询, {len(ground_truth)} ground truth条目")
        
        return tables, queries, ground_truth
    
    def _convert_tables(self, nlc_tables: List[Dict], task_type: str = 'join') -> List[Dict]:
        """转换表格式，并修复空列的seed表"""
        converted_tables = []
        
        # 尝试加载原始seed表数据（如果存在）
        original_seed_tables = {}
        original_base = Path('/root/autodl-tmp/datalakes/nlcTables')
        if original_base.exists():
            task_dir = f'nlcTables-J' if task_type == 'join' else 'nlcTables-U'
            
            # 方法1：从query-test目录加载单个seed表文件
            query_dir = original_base / task_dir / 'query-test'
            if query_dir.exists():
                try:
                    for seed_file in query_dir.glob('q_table_*.json'):
                        # Skip empty files
                        if seed_file.stat().st_size == 0:
                            self.logger.debug(f"跳过空文件: {seed_file.name}")
                            continue
                            
                        with open(seed_file, 'r') as f:
                            seed_data = json.load(f)
                            table_name = seed_file.stem  # 使用文件名作为表名
                            
                            # 转换seed表格式
                            table = {
                                'name': table_name,
                                'table_name': table_name,
                                'columns': []
                            }
                            
                            # 从"title"字段提取列名
                            if 'title' in seed_data and seed_data['title']:
                                for col_name in seed_data['title']:
                                    column = {
                                        'name': col_name,
                                        'column_name': col_name,
                                        'type': 'string',  # 默认类型
                                        'data_type': 'string',
                                        'sample_values': []
                                    }
                                    table['columns'].append(column)
                            
                            # 提取样本数据（如果有）
                            if 'data' in seed_data:
                                for row in seed_data['data'][:5]:  # 只取前5行作为样本
                                    for i, value in enumerate(row):
                                        if i < len(table['columns']):
                                            table['columns'][i]['sample_values'].append(str(value))
                            
                            original_seed_tables[table_name] = table
                    
                    if original_seed_tables:
                        self.logger.info(f"✅ 从query-test加载了 {len(original_seed_tables)} 个原始seed表")
                except Exception as e:
                    self.logger.warning(f"无法从query-test加载seed表: {e}")
            
            # 方法2：从origin-seed-table.json加载（备用）
            if not original_seed_tables:
                seed_path = original_base / task_dir / 'origin-seed-table.json'
                if seed_path.exists():
                    try:
                        with open(seed_path, 'r') as f:
                            seed_data = json.load(f)
                            for table in seed_data:
                                table_name = table.get('name', table.get('table_name', ''))
                                original_seed_tables[table_name] = table
                        self.logger.info(f"✅ 从origin-seed-table加载了 {len(original_seed_tables)} 个原始seed表")
                    except Exception as e:
                        self.logger.warning(f"无法从origin-seed-table加载seed表: {e}")
        
        for table in nlc_tables:
            # 确保所有表有标准的name字段
            table_name = table.get('name', table.get('table_name', ''))
            if 'name' not in table and 'table_name' in table:
                table['name'] = table['table_name']
            elif 'table_name' not in table and 'name' in table:
                table['table_name'] = table['name']
            
            # 如果是seed表且列为空，尝试从原始数据加载
            if table_name.startswith('q_table_') and not table.get('columns'):
                if table_name in original_seed_tables:
                    original_table = original_seed_tables[table_name]
                    table['columns'] = original_table.get('columns', [])
                    if 'sample_data' not in table and 'sample_data' in original_table:
                        table['sample_data'] = original_table['sample_data']
                    self.logger.debug(f"修复了seed表 {table_name} 的列信息")
            
            # 确保列格式正确
            if 'columns' in table:
                converted_columns = []
                for col in table['columns']:
                    # 标准化列格式
                    converted_col = {
                        'name': col.get('name', col.get('column_name', '')),
                        'column_name': col.get('column_name', col.get('name', '')),
                        'type': col.get('type', col.get('data_type', 'string')),
                        'data_type': col.get('data_type', col.get('type', 'string')),
                        'sample_values': col.get('sample_values', [])
                    }
                    converted_columns.append(converted_col)
                table['columns'] = converted_columns
            
            converted_tables.append(table)
        
        return converted_tables
    
    def _convert_queries(self, nlc_queries: List[Dict], task_type: str) -> List[Dict]:
        """转换查询格式"""
        converted_queries = []
        
        for query in nlc_queries:
            # NLCTables查询格式：{query_id, seed_table, expected_tables}
            # 主系统查询格式：{query_table, task_type}
            
            converted_query = {
                'query_table': query.get('seed_table', query.get('query_table', '')),
                'task_type': task_type,
                'query_id': query.get('query_id', '')
            }
            
            # 保留原始信息以便调试
            if 'expected_tables' in query:
                converted_query['expected_tables'] = query['expected_tables']
            
            converted_queries.append(converted_query)
        
        return converted_queries
    
    def _convert_ground_truth(self, nlc_ground_truth: Any, task_type: str, subset_type: str) -> List[Dict]:
        """转换ground truth格式
        
        NLCTables的ground truth是一个字典：
        {
            "1": [{"table_id": "dl_table_xxx", "relevance_score": 2}, ...],
            "3": [{"table_id": "dl_table_yyy", "relevance_score": 2}, ...],
            ...
        }
        
        需要转换为主系统格式：
        [
            {"query_table": "q_table_xxx", "ground_truth": ["dl_table_xxx", ...]},
            ...
        ]
        """
        converted_gt = []
        
        # 处理字典格式的ground truth
        if isinstance(nlc_ground_truth, dict):
            # 需要查询表名的映射
            # 从queries中获取query_id到seed_table的映射
            queries_path = Path(f'examples/nlctables/{task_type}_{subset_type}/queries.json')
            
            query_mapping = {}
            if queries_path.exists():
                with open(queries_path, 'r') as f:
                    queries = json.load(f)
                    for q in queries:
                        # 提取query_id中的数字部分
                        query_id = q.get('query_id', '')
                        if '_' in query_id:
                            # 例如 'nlc_join_1' -> '1'
                            id_num = query_id.split('_')[-1]
                            query_mapping[id_num] = q.get('seed_table', '')
            
            # 转换ground truth
            for query_id, expected_tables in nlc_ground_truth.items():
                # 获取对应的seed表名
                seed_table = query_mapping.get(query_id, f'q_table_{query_id}')
                
                # 提取表名列表
                table_names = []
                if isinstance(expected_tables, list):
                    for table_info in expected_tables:
                        if isinstance(table_info, dict):
                            table_id = table_info.get('table_id', '')
                            if table_id:
                                table_names.append(table_id)
                        elif isinstance(table_info, str):
                            table_names.append(table_info)
                
                converted_item = {
                    'query_table': seed_table,
                    'ground_truth': table_names
                }
                converted_gt.append(converted_item)
        
        # 处理列表格式的ground truth（备用）
        elif isinstance(nlc_ground_truth, list):
            for gt_item in nlc_ground_truth:
                if isinstance(gt_item, dict):
                    # 已经是字典格式
                    if 'query_table' in gt_item and 'ground_truth' in gt_item:
                        converted_gt.append(gt_item)
                    elif 'seed_table' in gt_item and 'expected_tables' in gt_item:
                        converted_item = {
                            'query_table': gt_item['seed_table'],
                            'ground_truth': gt_item['expected_tables']
                        }
                        converted_gt.append(converted_item)
                    else:
                        converted_gt.append(gt_item)
        
        return converted_gt
    
    def prepare_for_main_system(self, task_type: str, subset_type: str = 'subset') -> Dict[str, Any]:
        """
        准备数据供主系统使用
        
        Returns:
            包含所有必要数据的字典
        """
        tables, queries, ground_truth = self.load_nlctables_dataset(task_type, subset_type)
        
        return {
            'tables': tables,
            'queries': queries,
            'ground_truth': ground_truth,
            'dataset_name': f'nlctables_{task_type}_{subset_type}',
            'task_type': task_type
        }
    
    def convert_seed_table_to_query(self, seed_table_name: str, tables: List[Dict], task_type: str = 'join') -> Dict:
        """
        将NLCTables的seed表转换为主系统的查询格式
        
        Args:
            seed_table_name: seed表名（如q_table_xxx）
            tables: 所有表的列表
            task_type: 任务类型
        
        Returns:
            主系统格式的查询
        """
        return {
            'query_table': seed_table_name,
            'task_type': task_type
        }
    
    @staticmethod
    def filter_candidates_for_nlctables(tables: List[Dict]) -> List[Dict]:
        """
        过滤NLCTables的候选表（排除seed表）
        
        Args:
            tables: 所有表
        
        Returns:
            候选表列表（dl_table_开头的表）
        """
        candidates = []
        for table in tables:
            table_name = table.get('name', table.get('table_name', ''))
            # NLCTables的候选表以dl_table_开头，seed表以q_table_开头
            if table_name.startswith('dl_table_'):
                candidates.append(table)
        
        return candidates


def integrate_with_main_system():
    """
    将NLCTables集成到主系统的示例代码
    
    这个函数展示如何修改three_layer_ablation_optimized.py来支持NLCTables
    """
    code_snippet = '''
    # 在three_layer_ablation_optimized.py的load_dataset函数中添加:
    
    def load_dataset(task_type: str, dataset_type: str = 'subset') -> tuple:
        """加载数据集（支持NLCTables）"""
        
        # 检查是否是NLCTables数据集
        if 'nlctables' in dataset_type.lower():
            from nlctables_adapter import NLCTablesAdapter
            adapter = NLCTablesAdapter()
            
            # 解析subset类型
            if 'complete' in dataset_type:
                subset_type = 'complete'
            else:
                subset_type = 'subset'
            
            # 使用适配器加载数据
            tables, queries, ground_truth = adapter.load_nlctables_dataset(task_type, subset_type)
            
            logger.info(f"📊 Loaded NLCTables dataset: {len(tables)} tables, {len(queries)} queries")
            return tables, queries, ground_truth
        
        # 原有的数据加载逻辑...
        # [existing code]
    '''
    
    return code_snippet


if __name__ == "__main__":
    # 测试适配器
    import argparse
    
    parser = argparse.ArgumentParser(description='NLCTables适配器测试')
    parser.add_argument('--task', choices=['join', 'union'], default='join')
    parser.add_argument('--subset', choices=['subset', 'complete'], default='subset')
    
    args = parser.parse_args()
    
    # 创建适配器
    adapter = NLCTablesAdapter()
    
    # 加载数据
    try:
        data = adapter.prepare_for_main_system(args.task, args.subset)
        
        print(f"✅ 成功加载NLCTables数据集")
        print(f"   表数量: {len(data['tables'])}")
        print(f"   查询数量: {len(data['queries'])}")
        print(f"   Ground Truth数量: {len(data['ground_truth'])}")
        print(f"   数据集名称: {data['dataset_name']}")
        print(f"   任务类型: {data['task_type']}")
        
        # 显示第一个查询示例
        if data['queries']:
            print(f"\n📋 第一个查询示例:")
            print(f"   {data['queries'][0]}")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()