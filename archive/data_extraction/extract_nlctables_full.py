#!/usr/bin/env python3
"""
完整提取NLCTables数据集，支持subset和complete两个版本
类似于WebTable数据集的结构
"""

import json
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Set
import unicodedata
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_text_for_json(text: str) -> str:
    """清理文本中的无效Unicode字符"""
    if not text:
        return ""
    # 移除不可见字符和控制字符
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    # 替换高位Unicode字符
    text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    # 移除零宽字符
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    return text.strip()


def convert_nlctable_format(table_data: Dict) -> Dict:
    """将NLCTables原始格式转换为系统标准格式"""
    columns = []
    title = table_data.get('title', [])
    data = table_data.get('data', [])
    numeric_cols = set(table_data.get('numericColumns', []))
    date_cols = set(table_data.get('dateColumns', []))
    
    # 构建列信息
    for i, col_name in enumerate(title):
        col_type = 'string'
        if i in numeric_cols:
            col_type = 'numeric'
        elif i in date_cols:
            col_type = 'date'
        
        # 提取样本值
        sample_values = []
        for row in data[:5]:  # 取前5行作为样本
            if i < len(row):
                value = clean_text_for_json(str(row[i]))
                if value:
                    sample_values.append(value)
        
        columns.append({
            'name': clean_text_for_json(col_name),
            'type': col_type,
            'sample_values': sample_values
        })
    
    return {
        'columns': columns,
        'row_count': table_data.get('numDataRows', len(data)),
        'caption': clean_text_for_json(table_data.get('caption', '')),
        'nl_description': clean_text_for_json(table_data.get('caption', '')),
        'metadata': {
            'page_title': clean_text_for_json(table_data.get('pgTitle', '')),
            'second_title': clean_text_for_json(table_data.get('secondTitle', '')),
            'has_header': table_data.get('numHeaderRows', 1) > 0,
            'numeric_columns': list(numeric_cols),
            'date_columns': list(date_cols)
        }
    }


def extract_nlctables_dataset(
    source_dir: str, 
    output_base_dir: str, 
    task_type: str = 'union',
    subset_size: int = 100,
    complete_size: int = None
):
    """
    提取NLCTables数据集，生成subset和complete两个版本
    
    Args:
        source_dir: 原始NLCTables数据集路径
        output_base_dir: 输出基础路径
        task_type: 'union' 或 'join'
        subset_size: subset版本的查询数量（默认100）
        complete_size: complete版本的候选表数量（None表示全部）
    """
    if task_type == 'union':
        source_path = Path(source_dir) / 'nlcTables-U'
    else:
        source_path = Path(source_dir) / 'nlcTables-J'
    
    logger.info(f"从 {source_path} 提取 {task_type} 数据集...")
    
    # 1. 提取查询信息
    queries = []
    queries_file = source_path / 'queries-test.txt'
    if queries_file.exists():
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    query_id = parts[0]
                    nl_condition = clean_text_for_json(parts[1])
                    seed_table = parts[2]
                    
                    queries.append({
                        'query_table': seed_table,  # 保持与原系统兼容
                        'query_id': f'nlc_{task_type}_{query_id}',
                        'query_text': nl_condition,
                        'seed_table': seed_table,
                        'task_type': task_type,
                        'nl_condition': nl_condition  # 额外保存NL条件
                    })
    
    logger.info(f"提取了 {len(queries)} 个查询")
    
    # 2. 提取ground truth
    ground_truth = {}
    qtrels_file = source_path / 'qtrels-test.txt'
    if qtrels_file.exists():
        with open(qtrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    query_id = parts[0]
                    table_id = parts[2]
                    relevance = int(parts[3])
                    
                    if query_id not in ground_truth:
                        ground_truth[query_id] = []
                    
                    if relevance > 0:  # 只保留相关的表
                        ground_truth[query_id].append({
                            'table_id': table_id,
                            'relevance_score': relevance
                        })
    
    logger.info(f"提取了 {len(ground_truth)} 个查询的ground truth")
    
    # 3. 收集所有ground truth中提到的表ID
    gt_table_ids = set()
    for query_id, tables in ground_truth.items():
        for table_info in tables:
            gt_table_ids.add(table_info['table_id'])
    
    logger.info(f"Ground truth中包含 {len(gt_table_ids)} 个唯一表")
    
    # 4. 提取查询表（从zip文件）
    query_tables = {}
    query_zip_files = list(source_path.glob('query-test-*.zip'))
    
    for zip_file in query_zip_files:
        logger.info(f"提取查询表从 {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('.json'):
                    try:
                        content = zf.read(name)
                        table_data = json.loads(content)
                        table_id = Path(name).stem
                        
                        # 转换格式
                        formatted_table = convert_nlctable_format(table_data)
                        formatted_table['name'] = table_id
                        query_tables[table_id] = formatted_table
                        
                    except Exception as e:
                        logger.warning(f"处理 {name} 时出错: {e}")
    
    logger.info(f"提取了 {len(query_tables)} 个查询表")
    
    # 5. 提取候选表（从datalake-test文件夹）
    candidate_tables = {}
    datalake_dir = source_path / 'datalake-test'
    
    if datalake_dir.exists():
        json_files = list(datalake_dir.glob('*.json'))
        logger.info(f"发现 {len(json_files)} 个候选表文件...")
        
        # 首先加载所有ground truth表
        loaded_count = 0
        for table_id in gt_table_ids:
            json_file = datalake_dir / f"{table_id}.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        table_data = json.load(f)
                        formatted_table = convert_nlctable_format(table_data)
                        formatted_table['name'] = table_id
                        candidate_tables[table_id] = formatted_table
                        loaded_count += 1
                except Exception as e:
                    logger.warning(f"处理 {json_file} 时出错: {e}")
        
        logger.info(f"成功加载 {loaded_count}/{len(gt_table_ids)} 个ground truth表")
        
        # 然后随机选择额外的候选表
        remaining_files = [f for f in json_files if f.stem not in candidate_tables]
        random.shuffle(remaining_files)  # 随机打乱
        
        # 为complete版本加载更多表
        if complete_size is None:
            complete_size = min(5000, len(json_files))  # 默认最多5000个表
        
        for json_file in remaining_files:
            if len(candidate_tables) >= complete_size:
                break
            
            table_id = json_file.stem
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    table_data = json.load(f)
                    formatted_table = convert_nlctable_format(table_data)
                    formatted_table['name'] = table_id
                    candidate_tables[table_id] = formatted_table
                    
            except Exception as e:
                logger.warning(f"处理 {json_file} 时出错: {e}")
    
    logger.info(f"总共提取了 {len(candidate_tables)} 个候选表")
    
    # 6. 生成subset版本（少量查询，少量表）
    subset_queries = queries[:subset_size]
    subset_gt = {k: v for k, v in ground_truth.items() if k in [q['query_id'].split('_')[-1] for q in subset_queries]}
    
    # 收集subset需要的表
    subset_table_ids = set()
    # 添加查询表
    for q in subset_queries:
        if q['seed_table'] in query_tables:
            subset_table_ids.add(q['seed_table'])
    # 添加ground truth表
    for query_id, tables in subset_gt.items():
        for table_info in tables:
            subset_table_ids.add(table_info['table_id'])
    
    # 再添加一些随机表（保证有足够的负样本）
    random_tables = list(set(candidate_tables.keys()) - subset_table_ids)
    random.shuffle(random_tables)
    subset_table_ids.update(random_tables[:500])  # 添加500个随机表
    
    # 构建subset表集合
    subset_tables = []
    for table_id in subset_table_ids:
        if table_id in query_tables:
            subset_tables.append(query_tables[table_id])
        elif table_id in candidate_tables:
            subset_tables.append(candidate_tables[table_id])
    
    # 保存subset版本
    subset_dir = Path(output_base_dir) / f'{task_type}_subset'
    subset_dir.mkdir(parents=True, exist_ok=True)
    
    with open(subset_dir / 'queries.json', 'w', encoding='utf-8') as f:
        json.dump(subset_queries, f, ensure_ascii=False, indent=2)
    
    with open(subset_dir / 'ground_truth.json', 'w', encoding='utf-8') as f:
        json.dump(subset_gt, f, ensure_ascii=False, indent=2)
    
    with open(subset_dir / 'tables.json', 'w', encoding='utf-8') as f:
        json.dump(subset_tables, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Subset版本保存至: {subset_dir}")
    logger.info(f"   - 查询数: {len(subset_queries)}")
    logger.info(f"   - 表数: {len(subset_tables)}")
    
    # 7. 生成complete版本（所有查询，更多表）
    all_tables = []
    # 添加所有查询表
    all_tables.extend(list(query_tables.values()))
    # 添加所有候选表
    all_tables.extend(list(candidate_tables.values()))
    
    complete_dir = Path(output_base_dir) / f'{task_type}_complete'
    complete_dir.mkdir(parents=True, exist_ok=True)
    
    with open(complete_dir / 'queries.json', 'w', encoding='utf-8') as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    
    with open(complete_dir / 'ground_truth.json', 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    
    with open(complete_dir / 'tables.json', 'w', encoding='utf-8') as f:
        json.dump(all_tables, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Complete版本保存至: {complete_dir}")
    logger.info(f"   - 查询数: {len(queries)}")
    logger.info(f"   - 表数: {len(all_tables)}")
    
    return {
        'subset': {
            'queries': len(subset_queries),
            'tables': len(subset_tables),
            'path': str(subset_dir)
        },
        'complete': {
            'queries': len(queries),
            'tables': len(all_tables),
            'path': str(complete_dir)
        }
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='提取NLCTables数据集')
    parser.add_argument('--source-dir', type=str, 
                       default='/root/autodl-tmp/datalakes/nlcTables',
                       help='原始NLCTables数据集路径')
    parser.add_argument('--output-dir', type=str,
                       default='/root/dataLakesMulti/examples/nlctables',
                       help='输出目录')
    parser.add_argument('--task', choices=['union', 'join', 'both'], default='both',
                       help='任务类型')
    parser.add_argument('--subset-size', type=int, default=100,
                       help='subset版本的查询数量')
    parser.add_argument('--complete-size', type=int, default=3000,
                       help='complete版本的候选表数量')
    
    args = parser.parse_args()
    
    tasks = ['union', 'join'] if args.task == 'both' else [args.task]
    
    for task in tasks:
        logger.info(f"\n{'='*80}")
        logger.info(f"开始提取 {task.upper()} 数据集")
        logger.info(f"{'='*80}")
        
        results = extract_nlctables_dataset(
            source_dir=args.source_dir,
            output_base_dir=args.output_dir,
            task_type=task,
            subset_size=args.subset_size,
            complete_size=args.complete_size
        )
        
        logger.info(f"\n提取完成！")
        logger.info(f"Subset: {results['subset']}")
        logger.info(f"Complete: {results['complete']}")