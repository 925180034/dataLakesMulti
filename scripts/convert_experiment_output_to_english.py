#!/usr/bin/env python3
"""
Convert experiment output from Chinese to English format.
Ensures academic terminology and standard metrics reporting.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
import argparse


class ExperimentOutputConverter:
    """Convert experiment outputs to English with academic standards."""
    
    # Translation mappings
    TRANSLATIONS = {
        # Metadata fields
        "实验元数据": "experiment_metadata",
        "实验ID": "experiment_id",
        "时间戳": "timestamp",
        "系统版本": "system_version",
        "配置": "configuration",
        "数据集": "dataset",
        "表总数": "total_tables",
        "查询总数": "total_queries",
        "启用LLM匹配": "enable_llm_matching",
        "批次大小": "batch_size",
        "最大并发请求": "max_concurrent_requests",
        
        # Performance metrics
        "性能指标": "performance_metrics",
        "总执行时间": "total_execution_time",
        "平均查询延迟": "average_query_latency",
        "中位数查询延迟": "median_query_latency",
        "最小查询延迟": "min_query_latency",
        "最大查询延迟": "max_query_latency",
        "每秒查询数": "queries_per_second",
        "成功率": "success_rate",
        "缓存命中率": "cache_hit_rate",
        
        # Accuracy metrics
        "准确率指标": "accuracy_metrics",
        "精确率": "precision",
        "召回率": "recall",
        "真阳性": "true_positives",
        "假阳性": "false_positives",
        "假阴性": "false_negatives",
        "真阴性": "true_negatives",
        
        # Resource utilization
        "资源利用": "resource_utilization",
        "峰值内存使用_MB": "peak_memory_usage_mb",
        "平均CPU使用率": "average_cpu_usage_percent",
        "LLM API调用总数": "total_llm_api_calls",
        "消耗的Token总数": "total_tokens_consumed",
        "缓存大小_MB": "cache_size_mb",
        
        # Query results
        "查询结果": "query_results",
        "查询ID": "query_id",
        "查询表": "query_table",
        "查询类型": "query_type",
        "真实标签": "ground_truth",
        "预测结果": "predictions",
        "延迟_秒": "latency_seconds",
        "阶段分解": "stages_breakdown",
        "元数据过滤_毫秒": "metadata_filtering_ms",
        "向量搜索_毫秒": "vector_search_ms",
        "LLM验证_毫秒": "llm_verification_ms",
        "匹配详情": "matching_details",
        "置信度分数": "confidence_score",
        "匹配的列": "matched_columns",
        "源列": "source",
        "目标列": "target",
        "相似度": "similarity",
        "匹配类型": "match_type",
        
        # Error analysis
        "错误分析": "error_analysis",
        "错误总数": "total_errors",
        "错误分类": "error_categories",
        "失败的查询": "failed_queries",
        
        # Statistical summary
        "统计摘要": "statistical_summary",
        "查询分布": "query_distribution",
        "性能分位数": "performance_quantiles",
        "按查询类型的准确率": "accuracy_by_query_type",
        
        # System logs
        "系统日志": "system_logs",
        "警告": "warnings",
        "应用的优化": "optimizations_applied",
        
        # Common values
        "连接": "join",
        "合并": "union",
        "语义": "semantic",
        "精确": "exact",
        "近似": "approximate"
    }
    
    def __init__(self):
        """Initialize the converter."""
        pass
    
    def translate_key(self, key: str) -> str:
        """Translate a Chinese key to English."""
        return self.TRANSLATIONS.get(key, key)
    
    def translate_value(self, value: Any) -> Any:
        """Translate Chinese values to English where applicable."""
        if isinstance(value, str):
            return self.TRANSLATIONS.get(value, value)
        elif isinstance(value, dict):
            return self.translate_dict(value)
        elif isinstance(value, list):
            return [self.translate_value(item) for item in value]
        return value
    
    def translate_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively translate a dictionary from Chinese to English."""
        result = {}
        for key, value in data.items():
            english_key = self.translate_key(key)
            english_value = self.translate_value(value)
            result[english_key] = english_value
        return result
    
    def convert_file(self, input_path: Path, output_path: Path) -> None:
        """Convert a single experiment output file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to English
        english_data = self.translate_dict(data)
        
        # Add metadata if not present
        if 'experiment_metadata' not in english_data:
            english_data = {
                'experiment_metadata': {
                    'original_file': str(input_path),
                    'conversion_timestamp': datetime.now().isoformat()
                },
                **english_data
            }
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(english_data, f, indent=2, ensure_ascii=False)
        
        print(f"Converted: {input_path} -> {output_path}")
    
    def convert_directory(self, input_dir: Path, output_dir: Path) -> None:
        """Convert all JSON files in a directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for json_file in input_dir.glob("*.json"):
            output_file = output_dir / f"{json_file.stem}_en.json"
            self.convert_file(json_file, output_file)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert experiment outputs from Chinese to English"
    )
    parser.add_argument(
        "input",
        help="Input file or directory containing experiment outputs"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file or directory (default: add '_en' suffix)"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process directories recursively"
    )
    
    args = parser.parse_args()
    
    converter = ExperimentOutputConverter()
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file conversion
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_en.json"
        
        converter.convert_file(input_path, output_path)
    
    elif input_path.is_dir():
        # Directory conversion
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.name}_en"
        
        converter.convert_directory(input_path, output_path)
    
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    from datetime import datetime
    main()