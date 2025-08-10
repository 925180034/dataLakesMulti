#!/usr/bin/env python3
"""
实验结果统一管理器
统一保存、组织和分析实验结果
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import shutil

class ExperimentManager:
    """实验结果统一管理器"""
    
    def __init__(self, project_root: str = "/root/dataLakesMulti"):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / "experiment_results"
        self.ensure_directories()
    
    def ensure_directories(self):
        """确保实验结果目录结构存在"""
        dirs_to_create = [
            self.results_dir,
            self.results_dir / "latest",
            self.results_dir / "final", 
            self.results_dir / "archive",
            self.results_dir / "join",
            self.results_dir / "union",
            self.results_dir / "ablation",
            self.results_dir / "optimization"
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_filename(self, 
                         task: str,
                         dataset: str, 
                         experiment_type: str = "evaluation",
                         timestamp: Optional[str] = None) -> str:
        """生成标准化的文件名"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{task}_{dataset}_{timestamp}_{experiment_type}.json"
    
    def save_experiment(self,
                       results: Dict[str, Any],
                       task: str,
                       dataset: str = "subset", 
                       experiment_type: str = "evaluation",
                       description: str = "") -> str:
        """保存实验结果到统一目录"""
        
        # 生成文件名和路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.generate_filename(task, dataset, experiment_type, timestamp)
        
        # 根据任务类型选择子目录
        if experiment_type == "ablation":
            save_dir = self.results_dir / "ablation"
        elif experiment_type == "optimization":
            save_dir = self.results_dir / "optimization"  
        elif task in ["join", "union"]:
            save_dir = self.results_dir / task
        else:
            save_dir = self.results_dir
        
        file_path = save_dir / filename
        
        # 准备标准化的实验数据
        experiment_data = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "task": task,
                "dataset": dataset,
                "experiment_type": experiment_type,
                "description": description,
                "file_path": str(file_path),
                "parameters": results.get("parameters", {}),
                "system_config": results.get("system_config", {})
            },
            "results": results.get("results", {}),
            "detailed_results": results.get("detailed_results", []),
            "metadata": {
                "version": "1.0",
                "generated_by": "ExperimentManager",
                "total_queries": len(results.get("detailed_results", [])),
                "execution_time": results.get("execution_time", 0)
            }
        }
        
        # 保存到文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False)
        
        # 同时保存到latest目录（覆盖）
        latest_path = self.results_dir / "latest" / filename
        shutil.copy2(file_path, latest_path)
        
        print(f"✅ 实验结果已保存:")
        print(f"   📁 主要位置: {file_path}")
        print(f"   📁 最新位置: {latest_path}")
        
        return str(file_path)
    
    def load_experiment(self, file_path: str) -> Dict[str, Any]:
        """加载实验结果"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_experiments(self, 
                        task: Optional[str] = None,
                        experiment_type: Optional[str] = None,
                        limit: int = 10) -> List[Dict[str, Any]]:
        """列出实验结果"""
        all_files = []
        
        # 搜索所有JSON文件
        for json_file in self.results_dir.rglob("*.json"):
            if json_file.parent.name in ["latest", "archive"]:
                continue  # 跳过latest和archive的重复文件
                
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                info = data.get("experiment_info", {})
                
                # 过滤条件
                if task and info.get("task") != task:
                    continue
                if experiment_type and info.get("experiment_type") != experiment_type:
                    continue
                
                all_files.append({
                    "file_path": str(json_file),
                    "timestamp": info.get("timestamp", ""),
                    "task": info.get("task", "unknown"),
                    "dataset": info.get("dataset", "unknown"),
                    "experiment_type": info.get("experiment_type", "evaluation"),
                    "description": info.get("description", ""),
                    "avg_f1": data.get("results", {}).get("avg_f1", 0),
                    "tested_queries": data.get("metadata", {}).get("total_queries", 0)
                })
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        # 按时间戳排序，返回最新的
        all_files.sort(key=lambda x: x["timestamp"], reverse=True)
        return all_files[:limit]
    
    def get_latest_experiment(self, task: str) -> Optional[Dict[str, Any]]:
        """获取指定任务的最新实验"""
        experiments = self.list_experiments(task=task, limit=1)
        if experiments:
            return self.load_experiment(experiments[0]["file_path"])
        return None
    
    def generate_summary(self) -> Dict[str, Any]:
        """生成实验总结报告"""
        experiments = self.list_experiments(limit=100)
        
        summary = {
            "total_experiments": len(experiments),
            "tasks_summary": {},
            "recent_results": [],
            "best_results": {},
            "generated_at": datetime.now().isoformat()
        }
        
        # 按任务统计
        for exp in experiments:
            task = exp["task"]
            if task not in summary["tasks_summary"]:
                summary["tasks_summary"][task] = {
                    "count": 0,
                    "best_f1": 0,
                    "latest_f1": 0,
                    "total_queries": 0
                }
            
            task_summary = summary["tasks_summary"][task]
            task_summary["count"] += 1
            task_summary["total_queries"] += exp.get("tested_queries", 0)
            
            f1_score = exp.get("avg_f1", 0)
            if f1_score > task_summary["best_f1"]:
                task_summary["best_f1"] = f1_score
                summary["best_results"][task] = exp
            
            # 最新的F1分数
            if task_summary["count"] == 1:  # 第一个就是最新的（已排序）
                task_summary["latest_f1"] = f1_score
        
        # 最近的实验
        summary["recent_results"] = experiments[:5]
        
        return summary
    
    def save_summary(self) -> str:
        """保存实验总结到文件"""
        summary = self.generate_summary()
        summary_path = self.results_dir / "EXPERIMENT_SUMMARY.json"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 实验总结已保存: {summary_path}")
        return str(summary_path)
    
    def archive_old_results(self, days_old: int = 30):
        """归档旧的实验结果"""
        import time
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        
        archived_count = 0
        for json_file in self.results_dir.rglob("*.json"):
            if json_file.parent.name in ["archive", "final", "latest"]:
                continue  # 跳过已归档和最终结果
            
            if json_file.stat().st_mtime < cutoff_time:
                # 移动到archive目录
                archive_path = self.results_dir / "archive" / json_file.name
                shutil.move(str(json_file), str(archive_path))
                archived_count += 1
        
        print(f"📦 已归档 {archived_count} 个旧的实验结果")
        return archived_count

# 全局实例
experiment_manager = ExperimentManager()

def save_experiment_result(results: Dict[str, Any], 
                          task: str,
                          dataset: str = "subset",
                          experiment_type: str = "evaluation", 
                          description: str = "") -> str:
    """便捷的实验结果保存函数"""
    return experiment_manager.save_experiment(
        results, task, dataset, experiment_type, description
    )