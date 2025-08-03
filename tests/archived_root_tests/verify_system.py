#!/usr/bin/env python
"""
系统验证脚本
快速检查所有组件是否正常工作
"""

import asyncio
import json
from pathlib import Path
import sys


def check_file_exists(file_path: str, description: str) -> bool:
    """检查文件是否存在"""
    path = Path(file_path)
    if path.exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description} 不存在: {file_path}")
        return False


async def check_imports():
    """检查关键模块是否可以导入"""
    print("\n🔍 检查模块导入...")
    
    modules_to_check = [
        ("src.core.workflow", "工作流模块"),
        ("src.utils.data_parser", "数据解析模块"),
        ("src.utils.table_name_utils", "表名工具模块"),
        ("src.tools.vector_search", "向量搜索模块"),
        ("src.agents.table_discovery", "表发现代理"),
    ]
    
    all_ok = True
    for module_name, description in modules_to_check:
        try:
            __import__(module_name)
            print(f"✅ {description}: {module_name}")
        except ImportError as e:
            print(f"❌ {description} 导入失败: {e}")
            all_ok = False
    
    return all_ok


async def check_vector_index():
    """检查向量索引是否存在"""
    print("\n🔍 检查向量索引...")
    
    from src.config.settings import settings
    
    vector_db_path = Path(settings.vector_db.db_path)
    if vector_db_path.exists() and any(vector_db_path.iterdir()):
        print(f"✅ 向量索引目录存在: {vector_db_path}")
        return True
    else:
        print(f"❌ 向量索引未初始化: {vector_db_path}")
        print("   请运行: python init_vector_index.py --tables examples/final_subset_tables.json")
        return False


async def check_data_parsing():
    """检查数据解析是否正常"""
    print("\n🔍 检查数据解析...")
    
    try:
        from src.utils.data_parser import parse_table_data
        from src.utils.table_name_utils import normalize_table_name
        
        # 测试数据
        test_table = {
            "table_name": "test_table.csv",
            "columns": [
                {
                    "table_name": "test_table.csv",
                    "column_name": "id",
                    "data_type": "int",
                    "sample_values": ["1", "2"]
                }
            ]
        }
        
        # 解析测试
        parsed = parse_table_data(test_table)
        
        # 检查表名标准化
        if parsed.table_name == "test_table":  # 应该去掉.csv
            print("✅ 数据解析正常")
            print(f"   原始表名: test_table.csv")
            print(f"   标准化后: {parsed.table_name}")
            return True
        else:
            print(f"❌ 表名标准化失败: {parsed.table_name}")
            return False
            
    except Exception as e:
        print(f"❌ 数据解析测试失败: {e}")
        return False


async def check_workflow():
    """检查工作流是否可以创建"""
    print("\n🔍 检查工作流创建...")
    
    try:
        from src.core.workflow import create_workflow
        
        # 创建基础工作流
        basic_workflow = create_workflow(use_optimized=False)
        print("✅ 基础工作流创建成功")
        
        # 创建优化工作流
        optimized_workflow = create_workflow(use_optimized=True)
        print("✅ 优化工作流创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 工作流创建失败: {e}")
        return False


async def main():
    """主验证函数"""
    print("🚀 数据湖多智能体系统验证")
    print("="*50)
    
    all_checks_passed = True
    
    # 1. 检查必要文件
    print("\n📁 检查必要文件...")
    files_to_check = [
        ("examples/final_subset_tables.json", "表数据文件"),
        ("examples/query_table_example.json", "查询示例文件"),
        ("examples/final_subset_ground_truth.json", "Ground Truth文件"),
        ("config.yml", "配置文件"),
        (".env", "环境变量文件"),
    ]
    
    for file_path, description in files_to_check:
        if not check_file_exists(file_path, description):
            all_checks_passed = False
    
    # 2. 检查模块导入
    if not await check_imports():
        all_checks_passed = False
    
    # 3. 检查向量索引
    if not await check_vector_index():
        all_checks_passed = False
    
    # 4. 检查数据解析
    if not await check_data_parsing():
        all_checks_passed = False
    
    # 5. 检查工作流
    if not await check_workflow():
        all_checks_passed = False
    
    # 总结
    print("\n" + "="*50)
    if all_checks_passed:
        print("✅ 所有检查通过！系统已准备就绪。")
        print("\n下一步：")
        print("1. 如果向量索引未初始化，运行：")
        print("   python init_vector_index.py --tables examples/final_subset_tables.json")
        print("\n2. 运行示例查询：")
        print("   python run_cli.py discover -q \"find similar tables\" \\")
        print("     -t examples/query_table_example.json \\")
        print("     --all-tables examples/final_subset_tables.json")
        return 0
    else:
        print("❌ 部分检查失败，请修复上述问题。")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)