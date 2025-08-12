#!/usr/bin/env python
"""
分析实验结果
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List

def load_experiment_result(file_path: str) -> Dict:
    """加载实验结果"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_single_result(result: Dict, file_name: str):
    """分析单个实验结果"""
    print(f"\n📊 Analysis of: {file_name}")
    print("=" * 60)
    
    # 配置信息
    config = result.get('config', {})
    print(f"Dataset: {config.get('dataset', 'N/A')}")
    print(f"Queries: {config.get('queries', 'N/A')}")
    print(f"Workers: {config.get('workers', 'N/A')}")
    
    # 性能指标
    performance = result.get('performance', {})
    print(f"\n⏱️  Performance:")
    print(f"  Total Time: {performance.get('total_time', 0):.2f}s")
    print(f"  Avg Response: {performance.get('avg_response_time', 0):.3f}s")
    print(f"  Throughput: {performance.get('throughput', 0):.2f} QPS")
    
    # 准确率指标
    metrics = result.get('metrics', {})
    if metrics:
        print(f"\n🎯 Accuracy:")
        print(f"  Precision: {metrics.get('precision', 0):.3f}")
        print(f"  Recall: {metrics.get('recall', 0):.3f}")
        print(f"  F1-Score: {metrics.get('f1', 0):.3f}")
        print(f"  MRR: {metrics.get('mrr', 0):.3f}")
        
        # Hit@K指标
        print(f"\n📈 Hit@K:")
        for k in [1, 3, 5, 10]:
            hit_k = metrics.get(f'hit@{k}', 0)
            print(f"  Hit@{k}: {hit_k:.3f}")

def compare_results(results: List[tuple]):
    """比较多个实验结果"""
    print("\n" + "=" * 80)
    print("📊 COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    # 表头
    print(f"\n{'Experiment':<30} {'Time(s)':<10} {'QPS':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 80)
    
    for file_name, result in results:
        config = result.get('config', {})
        performance = result.get('performance', {})
        metrics = result.get('metrics', {})
        
        exp_name = f"{config.get('dataset', 'N/A')}_{config.get('queries', 0)}q"
        total_time = performance.get('total_time', 0)
        throughput = performance.get('throughput', 0)
        precision = metrics.get('precision', 0) if metrics else 0
        recall = metrics.get('recall', 0) if metrics else 0
        f1 = metrics.get('f1', 0) if metrics else 0
        
        print(f"{exp_name:<30} {total_time:<10.2f} {throughput:<10.2f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")

def main():
    """主函数"""
    # 实验结果目录
    result_dir = "experiment_results/multi_agent_llm"
    
    if not os.path.exists(result_dir):
        print(f"❌ Directory not found: {result_dir}")
        return
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(result_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"❌ No experiment results found in {result_dir}")
        return
    
    # 按时间排序（最新的在前）
    json_files.sort(reverse=True)
    
    print("=" * 80)
    print("🔬 MULTI-AGENT SYSTEM EXPERIMENT RESULTS ANALYSIS")
    print("=" * 80)
    print(f"\nFound {len(json_files)} experiment results")
    
    # 加载所有结果
    all_results = []
    for file_name in json_files[:10]:  # 只分析最近10个
        file_path = os.path.join(result_dir, file_name)
        try:
            result = load_experiment_result(file_path)
            all_results.append((file_name, result))
            
            # 分析单个结果
            analyze_single_result(result, file_name)
        except Exception as e:
            print(f"❌ Error loading {file_name}: {e}")
    
    # 比较结果
    if len(all_results) > 1:
        compare_results(all_results)
    
    # 最佳性能
    if all_results:
        print("\n" + "=" * 80)
        print("🏆 BEST PERFORMANCE")
        print("=" * 80)
        
        # 找出最佳QPS
        best_qps = max(all_results, key=lambda x: x[1].get('performance', {}).get('throughput', 0))
        print(f"\n Highest QPS: {best_qps[0]}")
        print(f"   QPS: {best_qps[1].get('performance', {}).get('throughput', 0):.2f}")
        
        # 找出最佳F1分数
        results_with_metrics = [(f, r) for f, r in all_results if r.get('metrics')]
        if results_with_metrics:
            best_f1 = max(results_with_metrics, key=lambda x: x[1].get('metrics', {}).get('f1', 0))
            print(f"\nHighest F1-Score: {best_f1[0]}")
            print(f"   F1: {best_f1[1].get('metrics', {}).get('f1', 0):.3f}")

if __name__ == "__main__":
    main()