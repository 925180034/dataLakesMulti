#!/usr/bin/env python3
"""
准备原始NLCTables数据用于baseline评估
确保使用与主系统一致的queries和ground truth
"""

import json
import pandas as pd
import os
from pathlib import Path
import shutil

class OriginalDataPreparer:
    def __init__(self):
        # 原始数据路径
        self.original_base = Path("/root/autodl-tmp/datalakes")
        # 提取后的数据路径（有正确的queries和ground truth）
        self.extracted_base = Path("/root/dataLakesMulti/examples")
        # Baseline输出路径
        self.output_base = Path("/root/dataLakesMulti/baselines/data")
        
    def convert_nlctables_table_to_csv(self, table_json_path: Path) -> pd.DataFrame:
        """将NLCTables原始JSON格式转换为CSV DataFrame"""
        try:
            with open(table_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # NLCTables使用'table_array'存储表格数据
            if 'table_array' in data and data['table_array']:
                # 第一行通常是列名
                table_array = data['table_array']
                if len(table_array) > 0:
                    columns = table_array[0] if len(table_array) > 0 else []
                    rows = table_array[1:] if len(table_array) > 1 else []
                    
                    # 清理列名（处理非字符串或特殊字符）
                    clean_columns = []
                    for i, col in enumerate(columns):
                        if col is None or col == '':
                            clean_columns.append(f'column_{i}')
                        else:
                            # 转换为字符串并清理特殊字符
                            col_str = str(col).strip()
                            # 替换特殊字符
                            col_str = col_str.replace(' ', '_').replace(':', '_').replace('-', '_')
                            col_str = col_str.replace('(', '_').replace(')', '_').replace('.', '_')
                            clean_columns.append(col_str if col_str else f'column_{i}')
                    
                    # 创建DataFrame
                    if rows:
                        # 确保行数据长度与列数匹配
                        aligned_rows = []
                        for row in rows:
                            if len(row) < len(clean_columns):
                                # 填充缺失值
                                row = list(row) + [None] * (len(clean_columns) - len(row))
                            elif len(row) > len(clean_columns):
                                # 截断多余值
                                row = row[:len(clean_columns)]
                            aligned_rows.append(row)
                        
                        df = pd.DataFrame(aligned_rows, columns=clean_columns)
                    else:
                        # 只有列名，没有数据
                        df = pd.DataFrame(columns=clean_columns)
                    return df
            elif 'data' in data and data['data']:
                # 备用格式
                return pd.DataFrame(data['data'])
            else:
                # 空表
                return pd.DataFrame()
        except Exception as e:
            # 返回空DataFrame而不是抛出异常
            return pd.DataFrame()
    
    def prepare_nlctables_data(self, task='join'):
        """准备NLCTables数据，使用原始表格但与主系统一致的queries和ground truth"""
        print(f"\n📊 准备NLCTables {task.upper()} 数据...")
        
        # 1. 加载提取后的queries和ground truth（确保一致性）
        extracted_path = self.extracted_base / 'nlctables' / f'{task}_subset'
        queries_file = extracted_path / 'queries.json'
        gt_file = extracted_path / 'ground_truth.json'
        
        if not queries_file.exists() or not gt_file.exists():
            print(f"❌ 找不到提取后的queries或ground truth: {extracted_path}")
            return
        
        with open(queries_file, 'r') as f:
            queries = json.load(f)
        with open(gt_file, 'r') as f:
            ground_truth = json.load(f)
        
        print(f"✅ 加载了 {len(queries)} 个查询和 {len(ground_truth)} 个ground truth")
        
        # 2. 转换原始数据表格
        if task == 'join':
            original_dir = self.original_base / 'nlcTables' / 'nlcTables-J'
        else:
            original_dir = self.original_base / 'nlcTables' / 'nlcTables-U'
        
        datalake_dir = original_dir / 'datalake-test'
        query_dir = original_dir / 'query-test'
        
        # 创建输出目录
        output_dir = self.output_base / 'aurum_original' / 'nlctables' / task
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计转换
        converted_count = 0
        failed_count = 0
        
        # 转换datalake表格
        print(f"🔄 转换datalake表格...")
        for json_file in datalake_dir.glob('*.json'):
            try:
                df = self.convert_nlctables_table_to_csv(json_file)
                if not df.empty:
                    csv_path = output_dir / f"{json_file.stem}.csv"
                    df.to_csv(csv_path, index=False)
                    converted_count += 1
                else:
                    # 空表也保存（可能是特征）
                    csv_path = output_dir / f"{json_file.stem}.csv"
                    with open(csv_path, 'w') as f:
                        f.write("")  # 空CSV
                    converted_count += 1
            except Exception as e:
                failed_count += 1
                if failed_count <= 3:  # 只显示前3个错误
                    print(f"  ⚠️ 转换失败 {json_file.name}: {e}")
        
        # 转换query表格
        print(f"🔄 转换query表格...")
        for json_file in query_dir.glob('*.json'):
            try:
                df = self.convert_nlctables_table_to_csv(json_file)
                if not df.empty:
                    csv_path = output_dir / f"{json_file.stem}.csv"
                    df.to_csv(csv_path, index=False)
                    converted_count += 1
                else:
                    csv_path = output_dir / f"{json_file.stem}.csv"
                    with open(csv_path, 'w') as f:
                        f.write("")
                    converted_count += 1
            except Exception as e:
                failed_count += 1
                if failed_count <= 3:
                    print(f"  ⚠️ 转换失败 {json_file.name}: {e}")
        
        print(f"✅ 成功转换 {converted_count} 个表格，失败 {failed_count} 个")
        
        # 3. 保存queries和ground truth的映射（用于评估）
        mapping = {
            'queries': queries,
            'ground_truth': ground_truth,
            'total_tables': converted_count,
            'dataset': 'nlctables',
            'task': task
        }
        
        mapping_file = output_dir / 'evaluation_mapping.json'
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        
        print(f"📁 数据已准备完成: {output_dir}")
        print(f"   - 表格数: {converted_count}")
        print(f"   - 查询数: {len(queries)}")
        print(f"   - 映射文件: {mapping_file}")
        
        return output_dir
    
    def prepare_all_datasets(self):
        """准备所有数据集"""
        results = {}
        
        # NLCTables
        for task in ['join', 'union']:
            nlc_dir = self.original_base / 'nlcTables' / f'nlcTables-{task[0].upper()}'
            if nlc_dir.exists():
                result_dir = self.prepare_nlctables_data(task)
                results[f'nlctables_{task}'] = result_dir
            else:
                print(f"⚠️ 找不到NLCTables {task} 数据: {nlc_dir}")
        
        # TODO: WebTable和OpenData的准备（如果需要）
        
        return results


def main():
    print("🚀 开始准备原始数据用于baseline评估")
    print("="*80)
    
    preparer = OriginalDataPreparer()
    
    # 准备数据
    results = preparer.prepare_all_datasets()
    
    print("\n" + "="*80)
    print("✅ 数据准备完成！")
    print("\n准备好的数据集:")
    for dataset, path in results.items():
        if path:
            print(f"  - {dataset}: {path}")
    
    print("\n下一步:")
    print("  1. 运行 run_unified_evaluation_original.py 使用原始数据评估")
    print("  2. 结果将与主系统使用相同的queries和ground truth")


if __name__ == "__main__":
    main()