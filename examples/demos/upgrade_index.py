#!/usr/bin/env python3
"""
索引系统升级脚本 - 从FAISS升级到更好的索引方案
"""

import asyncio
import json
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.absolute()))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def install_and_test_qdrant():
    """安装并测试Qdrant"""
    print("=== 安装和测试Qdrant ===")
    
    try:
        # 检查是否已安装
        try:
            import qdrant_client
            print("✅ Qdrant已安装")
        except ImportError:
            print("📦 安装Qdrant...")
            import subprocess
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "qdrant-client"
            ], check=True, capture_output=True, text=True)
            print("✅ Qdrant安装完成")
        
        # 测试Qdrant搜索引擎
        from src.tools.qdrant_search import create_qdrant_search
        
        qdrant_engine = create_qdrant_search()
        stats = qdrant_engine.get_collection_stats()
        
        print(f"✅ Qdrant测试成功")
        print(f"   集合统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Qdrant安装/测试失败: {e}")
        return False

async def install_and_test_chroma():
    """安装并测试ChromaDB"""
    print("\n=== 安装和测试ChromaDB ===")
    
    try:
        # 检查是否已安装
        try:
            import chromadb
            print("✅ ChromaDB已安装")
        except ImportError:
            print("📦 安装ChromaDB...")
            import subprocess
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "chromadb"
            ], check=True, capture_output=True, text=True)
            print("✅ ChromaDB安装完成")
        
        # 测试ChromaDB搜索引擎
        from src.tools.chroma_search import create_chroma_search
        
        chroma_engine = create_chroma_search()
        stats = chroma_engine.get_collection_stats()
        
        print(f"✅ ChromaDB测试成功")
        print(f"   集合统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB安装/测试失败: {e}")
        return False

async def benchmark_performance():
    """性能基准测试"""
    print("\n=== 性能基准测试 ===")
    
    try:
        import time
        import numpy as np
        from src.core.models import ColumnInfo
        
        # 创建测试数据
        test_columns = []
        for i in range(100):
            col = ColumnInfo(
                table_name=f"test_table_{i%10}",
                column_name=f"col_{i}",
                data_type="string" if i % 2 else "numeric",
                sample_values=[f"value_{j}" for j in range(3)]
            )
            test_columns.append(col)
        
        test_embeddings = [np.random.rand(384).tolist() for _ in range(100)]
        query_embedding = np.random.rand(384).tolist()
        
        print("📊 测试数据准备完成: 100个列向量")
        
        engines = {}
        
        # 测试Qdrant
        try:
            from src.tools.qdrant_search import create_qdrant_search
            qdrant_engine = create_qdrant_search()
            
            start_time = time.time()
            for col, emb in zip(test_columns, test_embeddings):
                await qdrant_engine.add_column_vector(col, emb)
            add_time = time.time() - start_time
            
            start_time = time.time()
            results = await qdrant_engine.search_similar_columns(query_embedding, k=10)
            search_time = time.time() - start_time
            
            engines["Qdrant"] = {
                "add_time": add_time,
                "search_time": search_time,
                "results_count": len(results)
            }
            
        except Exception as e:
            print(f"⚠️ Qdrant基准测试失败: {e}")
        
        # 测试ChromaDB
        try:
            from src.tools.chroma_search import create_chroma_search
            chroma_engine = create_chroma_search()
            
            start_time = time.time()
            for col, emb in zip(test_columns, test_embeddings):
                await chroma_engine.add_column_vector(col, emb)
            add_time = time.time() - start_time
            
            start_time = time.time()
            results = await chroma_engine.search_similar_columns(query_embedding, k=10)
            search_time = time.time() - start_time
            
            engines["ChromaDB"] = {
                "add_time": add_time,
                "search_time": search_time,
                "results_count": len(results)
            }
            
        except Exception as e:
            print(f"⚠️ ChromaDB基准测试失败: {e}")
        
        # 显示结果
        print("\n📈 性能测试结果:")
        for engine_name, metrics in engines.items():
            print(f"   {engine_name}:")
            print(f"     添加100个向量: {metrics['add_time']:.3f}s")
            print(f"     搜索时间: {metrics['search_time']:.3f}s")
            print(f"     返回结果: {metrics['results_count']}个")
        
        return engines
        
    except Exception as e:
        print(f"❌ 性能基准测试失败: {e}")
        return {}

def update_config_for_new_engine(engine_name: str):
    """更新配置文件使用新的向量引擎"""
    try:
        config_path = Path("config.yml")
        
        # 读取配置
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 更新provider
        if engine_name.lower() == "qdrant":
            content = content.replace('provider: "faiss"', 'provider: "qdrant"')
        elif engine_name.lower() == "chromadb":
            content = content.replace('provider: "faiss"', 'provider: "chromadb"')
        
        # 写回配置
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 配置文件已更新为使用 {engine_name}")
        
    except Exception as e:
        print(f"❌ 更新配置文件失败: {e}")

async def main():
    """主函数"""
    print("🚀 开始索引系统升级")
    
    # 1. 安装和测试新引擎
    qdrant_ok = await install_and_test_qdrant()
    chroma_ok = await install_and_test_chroma()
    
    if not (qdrant_ok or chroma_ok):
        print("❌ 所有新引擎都无法使用，保持FAISS")
        return
    
    # 2. 性能基准测试
    performance_results = await benchmark_performance()
    
    # 3. 推荐最佳引擎
    if qdrant_ok and chroma_ok:
        print("\n🎯 推荐结果:")
        print("   两个引擎都可用，推荐使用 Qdrant（更高性能）")
        
        user_input = input("   选择引擎 (1=Qdrant, 2=ChromaDB, 0=保持FAISS): ")
        
        if user_input == "1":
            update_config_for_new_engine("qdrant")
            print("✅ 已切换到Qdrant引擎")
        elif user_input == "2":
            update_config_for_new_engine("chromadb")
            print("✅ 已切换到ChromaDB引擎")
        else:
            print("✅ 保持使用FAISS引擎")
    
    elif qdrant_ok:
        update_config_for_new_engine("qdrant")
        print("✅ 自动切换到Qdrant引擎")
    
    elif chroma_ok:
        update_config_for_new_engine("chromadb")
        print("✅ 自动切换到ChromaDB引擎")
    
    print("\n🎉 索引系统升级完成！")
    print("💡 下一步: 运行 'python run_cli.py index-tables examples/webtable_join_tables.json' 建立索引")

if __name__ == "__main__":
    asyncio.run(main())