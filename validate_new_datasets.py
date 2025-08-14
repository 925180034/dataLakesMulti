#!/usr/bin/env python
"""
验证新创建的JOIN和UNION高质量数据集
"""
import json
from pathlib import Path
from collections import Counter

def validate_dataset(dataset_dir: Path, task_type: str):
    """验证单个数据集的质量"""
    print(f"\n{'='*60}")
    print(f"📊 验证 {task_type.upper()} 数据集")
    print(f"{'='*60}")
    
    # 加载数据
    with open(dataset_dir / 'tables.json', 'r') as f:
        tables = json.load(f)
    with open(dataset_dir / 'queries.json', 'r') as f:
        queries = json.load(f)
    with open(dataset_dir / 'ground_truth.json', 'r') as f:
        ground_truth = json.load(f)
    
    # 基本统计
    print(f"\n📈 基本统计:")
    print(f"  表数量: {len(tables)}")
    print(f"  查询数量: {len(queries)}")
    print(f"  Ground Truth条目: {len(ground_truth)}")
    
    # 构建ground truth映射
    gt_map = {}
    for gt in ground_truth:
        qt = gt['query_table']
        ct = gt['candidate_table']
        if qt not in gt_map:
            gt_map[qt] = set()
        gt_map[qt].add(ct)
    
    # 质量检查
    print(f"\n✅ 质量检查:")
    
    # 1. 覆盖率
    query_tables = {q['query_table'] for q in queries}
    gt_queries = set(gt_map.keys())
    coverage = len(query_tables & gt_queries) / len(query_tables) if query_tables else 0
    print(f"  Ground Truth覆盖率: {coverage:.1%} ({len(query_tables & gt_queries)}/{len(query_tables)})")
    
    # 2. 自匹配检查
    self_matches = 0
    for gt in ground_truth:
        if gt['query_table'] == gt['candidate_table']:
            self_matches += 1
    print(f"  自匹配数量: {self_matches} {'❌' if self_matches > 0 else '✅'}")
    
    # 3. 候选表分布
    candidate_counts = Counter(len(candidates) for candidates in gt_map.values())
    avg_candidates = sum(len(c) for c in gt_map.values()) / len(gt_map) if gt_map else 0
    
    print(f"\n📊 候选表分布:")
    print(f"  平均候选表数: {avg_candidates:.2f}")
    print(f"  分布情况:")
    for count in sorted(candidate_counts.keys())[:10]:
        freq = candidate_counts[count]
        pct = freq / len(gt_map) * 100 if gt_map else 0
        print(f"    {count:2d}个候选: {freq:3d}个查询 ({pct:5.1f}%)")
    
    # 4. 表存在性检查
    table_names = {t['table_name'] for t in tables}
    missing_query_tables = []
    missing_candidate_tables = []
    
    for qt in query_tables:
        if qt not in table_names:
            missing_query_tables.append(qt)
    
    for candidates in gt_map.values():
        for ct in candidates:
            if ct not in table_names:
                missing_candidate_tables.append(ct)
    
    print(f"\n🔍 表存在性检查:")
    print(f"  缺失的查询表: {len(missing_query_tables)} {'❌' if missing_query_tables else '✅'}")
    print(f"  缺失的候选表: {len(set(missing_candidate_tables))} {'❌' if missing_candidate_tables else '✅'}")
    
    # 5. 最频繁的候选表
    candidate_freq = Counter()
    for candidates in gt_map.values():
        for c in candidates:
            candidate_freq[c] += 1
    
    print(f"\n🎯 最频繁的候选表 (检查是否过度集中):")
    for table, count in candidate_freq.most_common(5):
        pct = count / len(gt_map) * 100 if gt_map else 0
        print(f"  {table}: {count}次 ({pct:.1f}%)")
    
    # 最高频率候选表的占比
    if candidate_freq:
        max_freq = candidate_freq.most_common(1)[0][1]
        max_freq_pct = max_freq / len(gt_map) * 100 if gt_map else 0
        if max_freq_pct > 50:
            print(f"  ⚠️ 警告: 最频繁的候选表出现在 {max_freq_pct:.1f}% 的查询中")
        else:
            print(f"  ✅ 候选表分布相对均衡 (最高 {max_freq_pct:.1f}%)")
    
    return {
        'tables': len(tables),
        'queries': len(queries),
        'ground_truth': len(ground_truth),
        'coverage': coverage,
        'avg_candidates': avg_candidates,
        'self_matches': self_matches,
        'missing_tables': len(missing_query_tables) + len(set(missing_candidate_tables))
    }

def main():
    print("="*80)
    print("🔍 验证高质量数据集")
    print("="*80)
    
    base_dir = Path('/root/dataLakesMulti/examples/separated_datasets')
    
    # 验证两个任务的数据集
    all_stats = {}
    for task_type in ['join', 'union']:
        task_dir = base_dir / task_type
        if task_dir.exists():
            stats = validate_dataset(task_dir, task_type)
            all_stats[task_type] = stats
        else:
            print(f"\n❌ 未找到 {task_type} 数据集目录: {task_dir}")
    
    # 总结
    print("\n" + "="*80)
    print("📈 数据集质量总结")
    print("="*80)
    
    quality_score = 0
    max_score = 0
    
    for task_type, stats in all_stats.items():
        print(f"\n{task_type.upper()}:")
        
        # 评分
        task_score = 0
        task_max = 5
        
        # 覆盖率 (1分)
        if stats['coverage'] == 1.0:
            task_score += 1
            print(f"  ✅ 100%覆盖率")
        else:
            print(f"  ❌ 覆盖率只有 {stats['coverage']:.1%}")
        
        # 无自匹配 (1分)
        if stats['self_matches'] == 0:
            task_score += 1
            print(f"  ✅ 无自匹配")
        else:
            print(f"  ❌ 存在 {stats['self_matches']} 个自匹配")
        
        # 平均候选表数 (1分)
        if stats['avg_candidates'] >= 3:
            task_score += 1
            print(f"  ✅ 平均 {stats['avg_candidates']:.1f} 个候选表")
        else:
            print(f"  ⚠️ 平均只有 {stats['avg_candidates']:.1f} 个候选表")
        
        # 表完整性 (1分)
        if stats['missing_tables'] == 0:
            task_score += 1
            print(f"  ✅ 所有表都存在")
        else:
            print(f"  ❌ 缺失 {stats['missing_tables']} 个表")
        
        # 数据量 (1分)
        if stats['queries'] >= 50:
            task_score += 1
            print(f"  ✅ 充足的查询数 ({stats['queries']})")
        else:
            print(f"  ⚠️ 查询数较少 ({stats['queries']})")
        
        print(f"  得分: {task_score}/{task_max}")
        quality_score += task_score
        max_score += task_max
    
    # 最终评分
    print("\n" + "="*80)
    print(f"🎯 总体质量评分: {quality_score}/{max_score} ({quality_score/max_score*100:.0f}%)")
    print("="*80)
    
    if quality_score / max_score >= 0.8:
        print("\n✅ 数据集质量优秀，可以用于实验！")
    elif quality_score / max_score >= 0.6:
        print("\n⚠️ 数据集质量良好，但有改进空间")
    else:
        print("\n❌ 数据集质量需要改进")
    
    print("\n📝 使用建议:")
    print("1. 运行JOIN实验:")
    print("   python run_cached_experiments.py --task join --dataset subset --max-queries 20")
    print("2. 运行UNION实验:")
    print("   python run_cached_experiments.py --task union --dataset subset --max-queries 20")
    print("3. 运行两个任务:")
    print("   python run_cached_experiments.py --task both --dataset subset --max-queries 10")


if __name__ == "__main__":
    main()