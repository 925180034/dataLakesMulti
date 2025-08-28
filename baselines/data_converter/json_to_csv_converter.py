#!/usr/bin/env python3
"""
数据格式转换器: JSON → CSV
将多智能体系统的JSON表格数据转换为baseline方法期望的CSV文件格式

支持的数据集:
- NLCTables (JOIN/UNION)
- WebTable (JOIN/UNION)  
- OpenData (JOIN/UNION)

输出格式:
- LSH: 独立CSV文件 + 查询配置
- Aurum: 独立CSV文件 + MinHash索引
- Starmie: 独立CSV文件 + 预训练格式
- Santos: 独立CSV文件 + 知识库格式
- D3L: 独立CSV文件 + 索引配置
"""

import json
import pandas as pd
import os
from pathlib import Path
import argparse
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaselineDataConverter:
    """将JSON表格数据转换为各种baseline方法的格式"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_json_tables(self, json_file: str) -> List[Dict]:
        """加载JSON格式的表格数据"""
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def json_table_to_csv(self, table_data: Dict) -> pd.DataFrame:
        """将单个JSON表格转换为pandas DataFrame"""
        table_name = table_data.get('name', table_data.get('table_name', 'unknown_table'))
        columns = table_data['columns']
        
        # 构建DataFrame数据
        data = {}
        max_samples = 0
        
        # 找出最长的sample_values长度
        for col in columns:
            sample_values = col.get('sample_values', [])
            max_samples = max(max_samples, len(sample_values))
        
        # 填充数据，短列用None/空字符串补齐
        for col in columns:
            col_name = col['name']
            sample_values = col.get('sample_values', [])
            
            # 扩展到统一长度
            while len(sample_values) < max_samples:
                sample_values.append('')
                
            data[col_name] = sample_values[:max_samples]
        
        return pd.DataFrame(data)
    
    def convert_for_lsh(self, dataset_name: str, task_type: str):
        """转换为LSH Ensemble格式"""
        input_file = self.input_dir / f"{dataset_name}/{task_type}_subset/tables.json"
        output_dir = self.output_dir / "lsh" / dataset_name / task_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Converting {dataset_name}-{task_type} for LSH...")
        
        tables = self.load_json_tables(input_file)
        
        # 转换每个表格为单独的CSV文件
        for i, table in enumerate(tables):
            df = self.json_table_to_csv(table)
            csv_filename = table.get('name', table.get('table_name', f'table_{i}'))
            if not csv_filename.endswith('.csv'):
                csv_filename += '.csv'
            
            csv_path = output_dir / csv_filename
            df.to_csv(csv_path, index=False)
        
        logging.info(f"LSH: 转换了 {len(tables)} 个表格到 {output_dir}")
        return len(tables)
    
    def convert_for_aurum(self, dataset_name: str, task_type: str):
        """转换为Aurum格式"""
        input_file = self.input_dir / f"{dataset_name}/{task_type}_subset/tables.json"
        output_dir = self.output_dir / "aurum" / dataset_name / task_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Converting {dataset_name}-{task_type} for Aurum...")
        
        tables = self.load_json_tables(input_file)
        
        # 转换每个表格为单独的CSV文件
        for i, table in enumerate(tables):
            df = self.json_table_to_csv(table)
            csv_filename = table.get('name', table.get('table_name', f'table_{i}'))
            if not csv_filename.endswith('.csv'):
                csv_filename += '.csv'
            
            csv_path = output_dir / csv_filename
            df.to_csv(csv_path, index=False)
        
        logging.info(f"Aurum: 转换了 {len(tables)} 个表格到 {output_dir}")
        return len(tables)
    
    def convert_for_starmie(self, dataset_name: str, task_type: str):
        """转换为Starmie格式"""
        input_file = self.input_dir / f"{dataset_name}/{task_type}_subset/tables.json"
        output_dir = self.output_dir / "starmie" / dataset_name / task_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Converting {dataset_name}-{task_type} for Starmie...")
        
        tables = self.load_json_tables(input_file)
        
        # Starmie需要特定的目录结构
        datalake_dir = output_dir / "datalake"
        datalake_dir.mkdir(exist_ok=True)
        
        for i, table in enumerate(tables):
            df = self.json_table_to_csv(table)
            csv_filename = table.get('name', table.get('table_name', f'table_{i}'))
            if not csv_filename.endswith('.csv'):
                csv_filename += '.csv'
            
            csv_path = datalake_dir / csv_filename
            df.to_csv(csv_path, index=False)
        
        logging.info(f"Starmie: 转换了 {len(tables)} 个表格到 {datalake_dir}")
        return len(tables)
    
    def convert_for_santos(self, dataset_name: str, task_type: str):
        """转换为Santos格式"""
        input_file = self.input_dir / f"{dataset_name}/{task_type}_subset/tables.json"
        output_dir = self.output_dir / "santos" / dataset_name / task_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Converting {dataset_name}-{task_type} for Santos...")
        
        tables = self.load_json_tables(input_file)
        
        for i, table in enumerate(tables):
            df = self.json_table_to_csv(table)
            csv_filename = table.get('name', table.get('table_name', f'table_{i}'))
            if not csv_filename.endswith('.csv'):
                csv_filename += '.csv'
            
            csv_path = output_dir / csv_filename
            df.to_csv(csv_path, index=False)
        
        logging.info(f"Santos: 转换了 {len(tables)} 个表格到 {output_dir}")
        return len(tables)
    
    def convert_for_d3l(self, dataset_name: str, task_type: str):
        """转换为D3L格式"""
        input_file = self.input_dir / f"{dataset_name}/{task_type}_subset/tables.json"
        output_dir = self.output_dir / "d3l" / dataset_name / task_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Converting {dataset_name}-{task_type} for D3L...")
        
        tables = self.load_json_tables(input_file)
        
        for i, table in enumerate(tables):
            df = self.json_table_to_csv(table)
            csv_filename = table.get('name', table.get('table_name', f'table_{i}'))
            if not csv_filename.endswith('.csv'):
                csv_filename += '.csv'
            
            csv_path = output_dir / csv_filename
            df.to_csv(csv_path, index=False)
        
        logging.info(f"D3L: 转换了 {len(tables)} 个表格到 {output_dir}")
        return len(tables)
    
    def convert_queries(self, dataset_name: str, task_type: str):
        """转换查询数据"""
        queries_file = self.input_dir / f"{dataset_name}/{task_type}_subset/queries.json"
        
        if not queries_file.exists():
            logging.warning(f"查询文件不存在: {queries_file}")
            return
        
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        # 为每个baseline保存查询配置
        for method in ['lsh', 'aurum', 'starmie', 'santos', 'd3l']:
            query_dir = self.output_dir / method / dataset_name / task_type
            query_dir.mkdir(parents=True, exist_ok=True)
            
            query_file = query_dir / "queries.json"
            with open(query_file, 'w', encoding='utf-8') as f:
                json.dump(queries, f, indent=2, ensure_ascii=False)
    
    def convert_ground_truth(self, dataset_name: str, task_type: str):
        """转换ground truth数据"""
        gt_file = self.input_dir / f"{dataset_name}/{task_type}_subset/ground_truth.json"
        
        if not gt_file.exists():
            logging.warning(f"Ground truth文件不存在: {gt_file}")
            return
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        # 为每个baseline保存ground truth
        for method in ['lsh', 'aurum', 'starmie', 'santos', 'd3l']:
            gt_dir = self.output_dir / method / dataset_name / task_type
            gt_dir.mkdir(parents=True, exist_ok=True)
            
            gt_file_out = gt_dir / "ground_truth.json"
            with open(gt_file_out, 'w', encoding='utf-8') as f:
                json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    
    def convert_all(self):
        """转换所有数据集和任务"""
        datasets = ['nlctables', 'webtable', 'opendata']
        tasks = ['join', 'union']
        methods = ['lsh', 'aurum', 'starmie', 'santos', 'd3l']
        
        results = {}
        
        for dataset in datasets:
            for task in tasks:
                # 检查输入文件是否存在
                input_file = self.input_dir / f"{dataset}/{task}_subset/tables.json"
                if not input_file.exists():
                    logging.warning(f"跳过不存在的数据集: {dataset}-{task}")
                    continue
                
                logging.info(f"转换数据集: {dataset}-{task}")
                
                # 转换表格数据
                for method in methods:
                    if method == 'lsh':
                        count = self.convert_for_lsh(dataset, task)
                    elif method == 'aurum':
                        count = self.convert_for_aurum(dataset, task)
                    elif method == 'starmie':
                        count = self.convert_for_starmie(dataset, task)
                    elif method == 'santos':
                        count = self.convert_for_santos(dataset, task)
                    elif method == 'd3l':
                        count = self.convert_for_d3l(dataset, task)
                    
                    results[f"{dataset}-{task}-{method}"] = count
                
                # 转换查询和ground truth
                self.convert_queries(dataset, task)
                self.convert_ground_truth(dataset, task)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='转换JSON表格数据为baseline格式')
    parser.add_argument('--input', '-i', type=str, 
                       default='/root/dataLakesMulti/examples',
                       help='输入JSON数据目录')
    parser.add_argument('--output', '-o', type=str,
                       default='/root/dataLakesMulti/baselines/data',
                       help='输出数据目录')
    parser.add_argument('--dataset', '-d', type=str, choices=['nlctables', 'webtable', 'opendata', 'all'],
                       default='all', help='要转换的数据集')
    parser.add_argument('--task', '-t', type=str, choices=['join', 'union', 'all'],
                       default='all', help='要转换的任务类型')
    parser.add_argument('--method', '-m', type=str, 
                       choices=['lsh', 'aurum', 'starmie', 'santos', 'd3l', 'all'],
                       default='all', help='目标baseline方法')
    
    args = parser.parse_args()
    
    converter = BaselineDataConverter(args.input, args.output)
    
    if args.dataset == 'all' and args.task == 'all' and args.method == 'all':
        # 转换所有数据
        results = converter.convert_all()
        print("\n=== 转换完成统计 ===")
        for key, count in results.items():
            print(f"{key}: {count} 个表格")
    else:
        print("单独转换功能待实现...")
    
    print(f"\n转换完成！输出目录: {args.output}")

if __name__ == "__main__":
    main()