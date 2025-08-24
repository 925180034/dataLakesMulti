#!/usr/bin/env python
"""
汇总实验结果文件夹中的结果
"""
import json
import sys
from pathlib import Path
from datetime import datetime

def summarize_results(results_dir="/root/dataLakesMulti/experiment_results"):
    """汇总所有实验结果"""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print("❌ 实验结果文件夹不存在")
        return
    
    # 获取所有json文件
    json_files = sorted(results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not json_files:
        print("❌ 没有找到任何实验结果文件")
        return
    
    print("\n" + "="*100)
    print("📊 实验结果汇总")
    print("="*100)
    print(f"共找到 {len(json_files)} 个结果文件\n")
    
    # 分类汇总
    ablation_results = []
    comparison_results = []
    
    for file_path in json_files:
        if file_path.name.startswith("comparison"):
            comparison_results.append(file_path)
        else:
            ablation_results.append(file_path)
    
    # 打印消融实验结果
    if ablation_results:
        print("\n📈 消融实验结果:")
        print("-"*100)
        print(f"{'文件名':<50} {'任务':<8} {'数据集':<10} {'模式':<10} {'最佳F1':<10} {'时间':<20}")
        print("-"*100)
        
        for file_path in ablation_results[:10]:  # 只显示最近10个
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                info = data.get('experiment_info', {})
                summary = data.get('summary', {})
                
                # 获取文件修改时间
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                time_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"{file_path.name:<50} {info.get('task_type', 'N/A'):<8} "
                      f"{info.get('dataset_type', 'N/A'):<10} {info.get('mode', 'N/A'):<10} "
                      f"{summary.get('best_f1', 0):<10.3f} {time_str:<20}")
            except Exception as e:
                print(f"❌ 读取文件失败 {file_path.name}: {e}")
    
    # 打印对比实验结果
    if comparison_results:
        print("\n\n📊 对比实验结果:")
        print("-"*100)
        print(f"{'文件名':<50} {'数据集':<10} {'时间':<20}")
        print("-"*100)
        
        for file_path in comparison_results[:5]:  # 只显示最近5个
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                info = data.get('experiment_info', {})
                
                # 获取文件修改时间
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                time_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"{file_path.name:<50} {info.get('dataset_type', 'N/A'):<10} {time_str:<20}")
                
                # 显示对比详情
                results = data.get('results', {})
                for task, task_results in results.items():
                    if 'static' in task_results and 'dynamic' in task_results:
                        static_l3 = task_results['static'].get('L1_L2_L3', {}).get('metrics', {})
                        dynamic_l3 = task_results['dynamic'].get('L1_L2_L3', {}).get('metrics', {})
                        
                        static_f1 = static_l3.get('f1_score', 0)
                        dynamic_f1 = dynamic_l3.get('f1_score', 0)
                        improvement = (dynamic_f1 - static_f1) / static_f1 * 100 if static_f1 > 0 else 0
                        
                        print(f"  └─ {task}: Static F1={static_f1:.3f}, Dynamic F1={dynamic_f1:.3f}, "
                              f"Improvement={improvement:+.1f}%")
                        
            except Exception as e:
                print(f"❌ 读取文件失败 {file_path.name}: {e}")
    
    print("\n" + "="*100)
    print(f"💡 提示: 使用 --enable-dynamic 标志启用动态优化")
    print(f"💡 提示: 使用 --compare 标志对比静态和动态模式")
    print("="*100)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        summarize_results(sys.argv[1])
    else:
        summarize_results()