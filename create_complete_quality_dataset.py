#!/usr/bin/env python
"""
为join和union两种任务类型创建高质量数据集
"""
import csv
import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import shutil

def load_csv_ground_truth(file_path: str) -> List[Dict]:
    """加载CSV格式的ground truth"""
    ground_truth = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth.append(row)
    return ground_truth

def get_available_tables(tables_dir: str) -> Set[str]:
    """获取所有可用的表名"""
    tables = set()
    for file_name in os.listdir(tables_dir):
        if file_name.endswith('.csv'):
            tables.add(file_name)
    return tables

def extract_quality_ground_truth(
    ground_truth: List[Dict],
    available_tables: Set[str],
    task_type: str,
    min_candidates: int = 1,
    max_candidates: int = 20,
    max_queries: int = 100
) -> Tuple[Dict[str, List[str]], Dict]:
    """
    提取高质量的ground truth
    """
    # 构建查询到候选表的映射
    query_to_candidates = defaultdict(set)
    
    for entry in ground_truth:
        query_table = entry.get('query_table', '')
        candidate_table = entry.get('candidate_table', '')
        
        # 过滤条件
        if not query_table or not candidate_table:
            continue
        if query_table == candidate_table:  # 自匹配
            continue
        if query_table not in available_tables:
            continue
        if candidate_table not in available_tables:
            continue
            
        query_to_candidates[query_table].add(candidate_table)
    
    # 分析数据分布
    candidate_distribution = Counter()
    for candidates in query_to_candidates.values():
        candidate_distribution[len(candidates)] += 1
    
    print(f"  {task_type.upper()} - 原始分布:")
    for count in sorted(candidate_distribution.keys())[:10]:
        print(f"    {count}个候选: {candidate_distribution[count]}个查询")
    
    # 选择高质量的查询
    quality_queries = {}
    
    # 1. 按候选表数量分组
    queries_by_candidate_count = defaultdict(list)
    for q, c in query_to_candidates.items():
        count = len(c)
        if min_candidates <= count <= max_candidates:
            queries_by_candidate_count[count].append((q, list(c)))
    
    # 2. 从每个组中选择查询，确保多样性
    queries_per_group = max(5, max_queries // len(queries_by_candidate_count)) if queries_by_candidate_count else 0
    
    for count in sorted(queries_by_candidate_count.keys()):
        queries_in_group = queries_by_candidate_count[count]
        for query, candidates in queries_in_group[:queries_per_group]:
            if len(quality_queries) < max_queries:
                quality_queries[query] = candidates
    
    # 统计信息
    stats = {
        'total_queries': len(quality_queries),
        'total_candidates': sum(len(c) for c in quality_queries.values()),
        'avg_candidates': sum(len(c) for c in quality_queries.values()) / len(quality_queries) if quality_queries else 0,
        'distribution': Counter(len(c) for c in quality_queries.values())
    }
    
    return quality_queries, stats

def load_table_data(table_path: str, max_rows: int = 100) -> Dict:
    """加载单个表的数据"""
    try:
        with open(table_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[:max_rows]
        
        if not lines:
            return None
            
        # 解析列信息
        header = lines[0].strip().split(',')
        columns = []
        
        for i, col_name in enumerate(header):
            sample_values = []
            for line in lines[1:6]:  # 前5行作为样本
                values = line.strip().split(',')
                if i < len(values):
                    sample_values.append(values[i])
            
            columns.append({
                'name': col_name,
                'type': 'string',
                'sample_values': sample_values[:3]
            })
        
        return {
            'table_name': Path(table_path).name,
            'columns': columns
        }
    except Exception as e:
        print(f"    警告: 无法读取表 {table_path}: {e}")
        return None

def create_dataset_for_task(
    task_type: str,
    quality_ground_truth: Dict[str, List[str]],
    tables_dir: str,
    output_dir: str
) -> Dict:
    """为特定任务创建数据集文件"""
    task_dir = output_dir / task_type
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有相关的表
    all_tables = set()
    for query in quality_ground_truth.keys():
        all_tables.add(query)
    for candidates in quality_ground_truth.values():
        all_tables.update(candidates)
    
    print(f"\n  处理 {len(all_tables)} 个表...")
    
    # 1. 创建tables.json
    tables_data = []
    for i, table_name in enumerate(all_tables, 1):
        if i % 50 == 0:
            print(f"    已处理 {i}/{len(all_tables)} 个表...")
        
        table_path = os.path.join(tables_dir, table_name)
        if os.path.exists(table_path):
            table_info = load_table_data(table_path)
            if table_info:
                tables_data.append(table_info)
    
    with open(task_dir / 'tables.json', 'w') as f:
        json.dump(tables_data, f, indent=2)
    print(f"  ✅ 创建 tables.json: {len(tables_data)} 个表")
    
    # 2. 创建queries.json
    queries_data = []
    for query_table in quality_ground_truth.keys():
        queries_data.append({
            'query_table': query_table,
            'query_type': task_type
        })
    
    with open(task_dir / 'queries.json', 'w') as f:
        json.dump(queries_data, f, indent=2)
    print(f"  ✅ 创建 queries.json: {len(queries_data)} 个查询")
    
    # 3. 创建ground_truth.json
    ground_truth_data = []
    for query_table, candidates in quality_ground_truth.items():
        for candidate in candidates:
            ground_truth_data.append({
                'query_table': query_table,
                'candidate_table': candidate,
                'query_type': task_type
            })
    
    with open(task_dir / 'ground_truth.json', 'w') as f:
        json.dump(ground_truth_data, f, indent=2)
    print(f"  ✅ 创建 ground_truth.json: {len(ground_truth_data)} 条记录")
    
    return {
        'tables': len(tables_data),
        'queries': len(queries_data),
        'ground_truth_entries': len(ground_truth_data)
    }

def main():
    print("="*80)
    print("🔧 创建完整的高质量数据集（JOIN + UNION）")
    print("="*80)
    
    base_dir = Path('/root/dataLakesMulti/Datasets/webtable')
    output_dir = Path('/root/dataLakesMulti/examples/separated_datasets')
    
    # 处理JOIN和UNION两种任务
    all_stats = {}
    
    for task_type in ['join', 'union']:
        print(f"\n{'='*60}")
        print(f"📊 处理 {task_type.upper()} 任务")
        print(f"{'='*60}")
        
        # 路径
        task_base = base_dir / task_type
        gt_file = task_base / f'webtable_{task_type}_ground_truth.csv'
        tables_dir = task_base / 'tables'
        
        # 1. 加载数据
        print(f"\n📚 加载 {task_type} 数据...")
        ground_truth = load_csv_ground_truth(gt_file)
        print(f"  原始ground truth: {len(ground_truth)} 条")
        
        available_tables = get_available_tables(tables_dir)
        print(f"  可用表: {len(available_tables)} 个")
        
        # 2. 提取高质量数据
        print(f"\n🎯 提取高质量ground truth...")
        quality_gt, stats = extract_quality_ground_truth(
            ground_truth,
            available_tables,
            task_type,
            min_candidates=1,
            max_candidates=20,
            max_queries=100
        )
        
        print(f"\n  选择了 {stats['total_queries']} 个查询")
        print(f"  平均候选表数: {stats['avg_candidates']:.2f}")
        
        # 显示分布
        print(f"\n  选中查询的分布:")
        for count, freq in sorted(stats['distribution'].items())[:10]:
            print(f"    {count}个候选: {freq}个查询")
        
        # 3. 创建数据集文件
        print(f"\n💾 创建 {task_type} 数据集...")
        task_stats = create_dataset_for_task(
            task_type,
            quality_gt,
            tables_dir,
            output_dir
        )
        
        # 保存完整名称的数据集（兼容旧代码）
        full_name_dir = output_dir / f"{task_type}_subset"
        full_name_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制文件到兼容目录
        for file_name in ['tables.json', 'queries.json', 'ground_truth.json']:
            src = output_dir / task_type / file_name
            dst = full_name_dir / file_name
            shutil.copy2(src, dst)
        
        print(f"  ✅ 同时创建了兼容目录: {full_name_dir}")
        
        all_stats[task_type] = {
            **stats,
            **task_stats
        }
    
    # 4. 创建综合统计
    print("\n" + "="*80)
    print("📈 数据集创建完成 - 总体统计")
    print("="*80)
    
    for task_type in ['join', 'union']:
        stats = all_stats[task_type]
        print(f"\n{task_type.upper()} 任务:")
        print(f"  查询数: {stats['total_queries']}")
        print(f"  表数: {stats['tables']}")
        print(f"  Ground Truth条目: {stats['ground_truth_entries']}")
        print(f"  平均候选表数: {stats['avg_candidates']:.2f}")
    
    # 5. 保存总体统计
    with open(output_dir / 'dataset_statistics.json', 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    # 6. 创建README
    readme_content = f"""# 高质量数据集

## 数据集结构
```
separated_datasets/
├── join/                    # JOIN任务数据
│   ├── tables.json          # {all_stats['join']['tables']}个表
│   ├── queries.json         # {all_stats['join']['total_queries']}个查询
│   └── ground_truth.json    # {all_stats['join']['ground_truth_entries']}条记录
├── union/                   # UNION任务数据
│   ├── tables.json          # {all_stats['union']['tables']}个表
│   ├── queries.json         # {all_stats['union']['total_queries']}个查询
│   └── ground_truth.json    # {all_stats['union']['ground_truth_entries']}条记录
├── join_subset/             # JOIN兼容目录（同join/）
├── union_subset/            # UNION兼容目录（同union/）
└── dataset_statistics.json # 统计信息
```

## 数据质量
- **JOIN**: 平均{all_stats['join']['avg_candidates']:.1f}个候选表/查询
- **UNION**: 平均{all_stats['union']['avg_candidates']:.1f}个候选表/查询
- **100%覆盖率**: 所有查询都有ground truth
- **过滤自匹配**: 移除了所有自匹配的候选
- **验证表存在**: 确保所有表都在数据集中

## 使用方法

### 测试单个任务
```bash
# 测试JOIN任务
python run_cached_experiments.py --task join --dataset subset --max-queries 10

# 测试UNION任务
python run_cached_experiments.py --task union --dataset subset --max-queries 10
```

### 测试两个任务
```bash
python run_cached_experiments.py --task both --dataset subset --max-queries 10
```

## 创建时间
{Path(__file__).stat().st_mtime}
"""
    
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print("\n✅ 完成！")
    print(f"\n输出目录: {output_dir}")
    print(f"包含:")
    print(f"  - join/ : JOIN任务数据集")
    print(f"  - union/ : UNION任务数据集")
    print(f"  - join_subset/ : JOIN兼容目录")
    print(f"  - union_subset/ : UNION兼容目录")
    print(f"  - dataset_statistics.json : 统计信息")
    print(f"  - README.md : 使用说明")
    
    print(f"\n使用示例:")
    print(f"  python run_cached_experiments.py --task both --dataset subset --max-queries 20")


if __name__ == "__main__":
    main()