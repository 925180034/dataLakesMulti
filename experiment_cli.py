#!/usr/bin/env python3
"""
实验管理CLI工具
用于查看、管理和分析实验结果
"""

import argparse
import json
from pathlib import Path
from tabulate import tabulate
from datetime import datetime

def setup_experiment_manager():
    """设置实验管理器"""
    try:
        from src.utils.experiment_manager import experiment_manager
        return experiment_manager
    except ImportError:
        print("❌ 无法导入实验管理器")
        return None

def list_experiments(args):
    """列出实验结果"""
    manager = setup_experiment_manager()
    if not manager:
        return
    
    experiments = manager.list_experiments(
        task=args.task,
        experiment_type=args.type,
        limit=args.limit
    )
    
    if not experiments:
        print("📭 没有找到符合条件的实验结果")
        return
    
    print(f"🧪 找到 {len(experiments)} 个实验结果:")
    print()
    
    # 准备表格数据
    table_data = []
    for exp in experiments:
        table_data.append([
            exp["task"],
            exp["experiment_type"], 
            exp["dataset"],
            f"{exp['avg_f1']:.1%}" if exp['avg_f1'] > 0 else "N/A",
            exp["tested_queries"],
            exp["timestamp"][:16],  # 只显示到分钟
            Path(exp["file_path"]).name
        ])
    
    headers = ["任务", "类型", "数据集", "F1分数", "查询数", "时间", "文件名"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def show_summary(args):
    """显示实验总结"""
    manager = setup_experiment_manager()
    if not manager:
        return
    
    print("📊 生成实验总结...")
    summary = manager.generate_summary()
    
    print("\n" + "="*60)
    print("🧪 实验总结报告")
    print("="*60)
    print(f"总实验数量: {summary['total_experiments']}")
    print(f"生成时间: {summary['generated_at'][:19]}")
    
    # 按任务显示统计
    print("\n📋 按任务统计:")
    task_table = []
    for task, stats in summary["tasks_summary"].items():
        task_table.append([
            task.upper(),
            stats["count"],
            f"{stats['best_f1']:.1%}",
            f"{stats['latest_f1']:.1%}",
            stats["total_queries"]
        ])
    
    if task_table:
        headers = ["任务", "实验数", "最佳F1", "最新F1", "总查询数"]
        print(tabulate(task_table, headers=headers, tablefmt="grid"))
    
    # 显示最佳结果
    print("\n🏆 最佳结果:")
    for task, best_exp in summary["best_results"].items():
        print(f"  {task.upper()}: F1={best_exp['avg_f1']:.1%} ({best_exp['timestamp'][:16]})")
    
    # 显示最近实验
    print("\n🕐 最近实验:")
    for exp in summary["recent_results"][:3]:
        print(f"  {exp['task']}-{exp['experiment_type']}: F1={exp['avg_f1']:.1%} ({exp['timestamp'][:16]})")
    
    # 保存总结
    if args.save:
        summary_file = manager.save_summary()
        print(f"\n💾 总结已保存到: {summary_file}")

def show_experiment(args):
    """显示特定实验的详细信息"""
    if not Path(args.file).exists():
        print(f"❌ 文件不存在: {args.file}")
        return
    
    try:
        with open(args.file, 'r') as f:
            data = json.load(f)
        
        info = data.get("experiment_info", {})
        results = data.get("results", {})
        
        print("="*60)
        print(f"🧪 实验详情: {Path(args.file).name}")
        print("="*60)
        
        # 基本信息
        print("📋 基本信息:")
        print(f"  任务: {info.get('task', 'unknown').upper()}")
        print(f"  数据集: {info.get('dataset', 'unknown')}")
        print(f"  类型: {info.get('experiment_type', 'evaluation')}")
        print(f"  时间: {info.get('timestamp', 'unknown')[:19]}")
        print(f"  描述: {info.get('description', '无描述')}")
        
        # 参数配置
        params = info.get("parameters", {})
        if params:
            print("\n⚙️ 参数配置:")
            for key, value in params.items():
                print(f"  {key}: {value}")
        
        # 结果
        print("\n📊 实验结果:")
        print(f"  测试查询数: {results.get('tested_queries', 0)}")
        print(f"  平均精确率: {results.get('avg_precision', 0):.1%}")
        print(f"  平均召回率: {results.get('avg_recall', 0):.1%}")
        print(f"  平均F1分数: {results.get('avg_f1', 0):.1%}")
        print(f"  平均查询时间: {results.get('avg_query_time', 0):.2f}秒")
        
        # 详细结果统计
        detailed = data.get("detailed_results", [])
        if detailed:
            f1_scores = [r.get('f1', 0) for r in detailed if isinstance(r, dict)]
            if f1_scores:
                print(f"\n📈 F1分数分布:")
                print(f"  最高: {max(f1_scores):.1%}")
                print(f"  最低: {min(f1_scores):.1%}")
                print(f"  中位数: {sorted(f1_scores)[len(f1_scores)//2]:.1%}")
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ 解析文件错误: {e}")

def archive_old(args):
    """归档旧的实验结果"""
    manager = setup_experiment_manager()
    if not manager:
        return
    
    print(f"📦 归档 {args.days} 天前的实验结果...")
    archived_count = manager.archive_old_results(days_old=args.days)
    print(f"✅ 已归档 {archived_count} 个文件")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="实验管理CLI工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # list 命令
    list_parser = subparsers.add_parser("list", help="列出实验结果")
    list_parser.add_argument("--task", choices=["join", "union"], help="过滤任务类型")
    list_parser.add_argument("--type", help="过滤实验类型")
    list_parser.add_argument("--limit", type=int, default=10, help="显示数量限制")
    
    # summary 命令
    summary_parser = subparsers.add_parser("summary", help="显示实验总结")
    summary_parser.add_argument("--save", action="store_true", help="保存总结到文件")
    
    # show 命令
    show_parser = subparsers.add_parser("show", help="显示特定实验详情")
    show_parser.add_argument("file", help="实验结果文件路径")
    
    # archive 命令
    archive_parser = subparsers.add_parser("archive", help="归档旧实验结果")
    archive_parser.add_argument("--days", type=int, default=30, help="归档多少天前的结果")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_experiments(args)
    elif args.command == "summary":
        show_summary(args)
    elif args.command == "show":
        show_experiment(args)
    elif args.command == "archive":
        archive_old(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()