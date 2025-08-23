#!/usr/bin/env python3
"""
运行OpenData的JOIN和UNION任务
支持subset和complete版本
"""

import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path

def run_experiment(dataset_base: str, max_queries: int, output_prefix: str):
    """运行JOIN和UNION实验"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}
    
    # 分别运行JOIN和UNION
    for task in ['join', 'union']:
        dataset_path = f"{dataset_base}/{task}_subset" if 'subset' in dataset_base else f"{dataset_base}/{task}_complete"
        
        print(f"\n{'='*60}")
        print(f"🚀 运行 {task.upper()} 任务")
        print(f"📂 数据集: {dataset_path}")
        print(f"📊 查询数: {max_queries}")
        print(f"{'='*60}")
        
        # 检查数据集是否存在
        if not Path(dataset_path).exists():
            print(f"❌ 数据集不存在: {dataset_path}")
            continue
            
        # 运行实验
        cmd = [
            "python", "three_layer_ablation_optimized.py",
            "--task", task,
            "--dataset", dataset_path,
            "--max-queries", str(max_queries),
            "--simple"  # 使用简单查询，不使用挑战性查询
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            
            # 提取结果
            output_file = f"experiment_results/ablation_optimized_{dataset_path.replace('/', '_')}_{timestamp}.json"
            if Path(output_file).exists():
                with open(output_file, 'r') as f:
                    task_results = json.load(f)
                    results[task] = task_results.get(task, {})
                    
        except subprocess.CalledProcessError as e:
            print(f"❌ 运行失败: {e}")
            print(f"错误输出: {e.stderr}")
    
    # 保存综合结果
    output_file = f"experiment_results/{output_prefix}_{timestamp}.json"
    Path("experiment_results").mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 综合结果保存到: {output_file}")
    
    # 打印结果摘要
    print("\n" + "="*80)
    print("📊 实验结果摘要")
    print("="*80)
    
    for task in results:
        print(f"\n{task.upper()} 任务:")
        if 'L1_L2_L3' in results[task]:
            metrics = results[task]['L1_L2_L3']['metrics']
            print(f"  L1+L2+L3: F1={metrics['f1_score']:.3f}, Hit@1={metrics['hit@1']:.3f}")
        else:
            print("  无结果")
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='运行OpenData JOIN和UNION实验')
    parser.add_argument('--version', choices=['subset', 'complete'], default='subset',
                       help='数据集版本')
    parser.add_argument('--max-queries', type=int, default=20,
                       help='最大查询数')
    
    args = parser.parse_args()
    
    # 确定数据集路径
    if args.version == 'subset':
        dataset_base = "examples/opendata"
        output_prefix = "opendata_subset_both"
    else:
        dataset_base = "examples/opendata"
        output_prefix = "opendata_complete_both"
    
    # 运行实验
    run_experiment(dataset_base, args.max_queries, output_prefix)

if __name__ == "__main__":
    main()