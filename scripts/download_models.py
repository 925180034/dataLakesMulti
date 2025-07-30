#!/usr/bin/env python3
"""
模型下载和环境检查脚本
用于完整测试的模型准备
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_internet_connection() -> bool:
    """检查网络连接"""
    try:
        import urllib.request
        urllib.request.urlopen('https://www.google.com', timeout=5)
        return True
    except:
        return False

def check_disk_space(required_gb: float = 2.0) -> bool:
    """检查磁盘空间"""
    try:
        import shutil
        free_bytes = shutil.disk_usage('.').free
        free_gb = free_bytes / (1024**3)
        logger.info(f"可用磁盘空间: {free_gb:.2f} GB")
        return free_gb >= required_gb
    except Exception as e:
        logger.error(f"检查磁盘空间失败: {e}")
        return False

def install_dependencies() -> Dict[str, bool]:
    """安装必需的依赖包"""
    dependencies = {
        'sentence-transformers': 'sentence-transformers>=2.7.0',
        'transformers': 'transformers>=4.21.0',
        'torch': 'torch>=1.12.0',
        'numpy': 'numpy>=1.21.0',
        'faiss-cpu': 'faiss-cpu>=1.7.4',
        'hnswlib': 'hnswlib>=0.7.0',
        'chromadb': 'chromadb>=0.4.22',
        'whoosh': 'whoosh>=2.7.4'
    }
    
    results = {}
    
    for name, package in dependencies.items():
        try:
            logger.info(f"检查依赖: {name}")
            __import__(name.replace('-', '_'))
            results[name] = True
            logger.info(f"✅ {name} 已安装")
        except ImportError:
            logger.info(f"📦 安装 {name}...")
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    results[name] = True
                    logger.info(f"✅ {name} 安装成功")
                else:
                    results[name] = False
                    logger.error(f"❌ {name} 安装失败: {result.stderr}")
            except Exception as e:
                results[name] = False
                logger.error(f"❌ {name} 安装异常: {e}")
    
    return results

def download_sentence_transformer_models() -> Dict[str, bool]:
    """下载 SentenceTransformer 模型"""
    models = {
        'all-MiniLM-L6-v2': '主要嵌入模型 (384维, ~90MB)',
        'all-mpnet-base-v2': '高精度嵌入模型 (768维, ~420MB)',
        'paraphrase-MiniLM-L6-v2': '释义专用模型 (384维, ~90MB)'
    }
    
    results = {}
    
    try:
        from sentence_transformers import SentenceTransformer
        
        for model_name, description in models.items():
            try:
                logger.info(f"📥 下载模型: {model_name} - {description}")
                start_time = time.time()
                
                # 下载并初始化模型
                model = SentenceTransformer(model_name)
                
                # 测试模型是否工作
                test_embedding = model.encode("test sentence")
                
                download_time = time.time() - start_time
                results[model_name] = True
                logger.info(f"✅ {model_name} 下载完成 ({download_time:.1f}s, 维度: {len(test_embedding)})")
                
            except Exception as e:
                results[model_name] = False
                logger.error(f"❌ {model_name} 下载失败: {e}")
    
    except ImportError as e:
        logger.error(f"sentence-transformers 未正确安装: {e}")
        for model_name in models:
            results[model_name] = False
    
    return results

def verify_model_functionality() -> Dict[str, bool]:
    """验证模型功能"""
    tests = {}
    
    # 测试 SentenceTransformer
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 测试单个文本编码
        embedding = model.encode("This is a test sentence")
        assert len(embedding) == 384, f"嵌入维度错误: {len(embedding)}"
        
        # 测试批量编码
        batch_embeddings = model.encode(["sentence 1", "sentence 2", "sentence 3"])
        assert batch_embeddings.shape == (3, 384), f"批量嵌入形状错误: {batch_embeddings.shape}"
        
        tests['sentence_transformer'] = True
        logger.info("✅ SentenceTransformer 功能测试通过")
        
    except Exception as e:
        tests['sentence_transformer'] = False
        logger.error(f"❌ SentenceTransformer 测试失败: {e}")
    
    # 测试 FAISS
    try:
        import faiss
        import numpy as np
        
        # 创建测试索引
        dimension = 384
        index = faiss.IndexFlatL2(dimension)
        
        # 添加测试向量
        test_vectors = np.random.random((100, dimension)).astype('float32')
        index.add(test_vectors)
        
        # 搜索测试
        query_vector = np.random.random((1, dimension)).astype('float32')
        distances, indices = index.search(query_vector, 5)
        
        assert len(indices[0]) == 5, f"搜索结果数量错误: {len(indices[0])}"
        
        tests['faiss'] = True
        logger.info("✅ FAISS 功能测试通过")
        
    except Exception as e:
        tests['faiss'] = False
        logger.error(f"❌ FAISS 测试失败: {e}")
    
    # 测试 HNSWlib
    try:
        import hnswlib
        import numpy as np
        
        # 创建测试索引
        dimension = 384
        index = hnswlib.Index(space='cosine', dim=dimension)
        index.init_index(max_elements=1000, ef_construction=100, M=32)
        
        # 添加测试向量
        test_vectors = np.random.random((100, dimension)).astype('float32')
        labels = np.arange(100)
        index.add_items(test_vectors, labels)
        
        # 搜索测试
        query_vector = np.random.random((1, dimension)).astype('float32')
        indices, distances = index.knn_query(query_vector, k=5)
        
        assert len(indices[0]) == 5, f"HNSW搜索结果数量错误: {len(indices[0])}"
        
        tests['hnswlib'] = True
        logger.info("✅ HNSWlib 功能测试通过")
        
    except Exception as e:
        tests['hnswlib'] = False
        logger.error(f"❌ HNSWlib 测试失败: {e}")
    
    return tests

def check_api_keys() -> Dict[str, bool]:
    """检查API密钥配置"""
    api_keys = {
        'GEMINI_API_KEY': 'Google Gemini API',
        'OPENAI_API_KEY': 'OpenAI API',
        'ANTHROPIC_API_KEY': 'Anthropic Claude API'
    }
    
    results = {}
    for key, description in api_keys.items():
        value = os.getenv(key)
        if value and len(value) > 10:
            results[key] = True
            logger.info(f"✅ {description} 密钥已配置")
        else:
            results[key] = False
            logger.warning(f"⚠️  {description} 密钥未配置")
    
    return results

def generate_test_readiness_report(
    internet: bool,
    disk_space: bool,
    dependencies: Dict[str, bool],
    models: Dict[str, bool], 
    functionality: Dict[str, bool],
    api_keys: Dict[str, bool]
) -> str:
    """生成测试就绪报告"""
    
    report = []
    report.append("=" * 80)
    report.append("🧪 完整测试环境就绪报告")
    report.append("=" * 80)
    report.append("")
    
    # 基础环境
    report.append("🌐 基础环境检查:")
    report.append(f"  网络连接: {'✅ 正常' if internet else '❌ 无连接'}")
    report.append(f"  磁盘空间: {'✅ 充足' if disk_space else '❌ 不足'}")
    report.append("")
    
    # 依赖包
    report.append("📦 依赖包安装状态:")
    for name, status in dependencies.items():
        status_icon = "✅" if status else "❌"
        report.append(f"  {name}: {status_icon}")
    dependency_success_rate = sum(dependencies.values()) / len(dependencies) * 100
    report.append(f"  安装成功率: {dependency_success_rate:.1f}%")
    report.append("")
    
    # 模型下载
    report.append("🤖 模型下载状态:")
    for name, status in models.items():
        status_icon = "✅" if status else "❌"
        report.append(f"  {name}: {status_icon}")
    model_success_rate = sum(models.values()) / len(models) * 100 if models else 0
    report.append(f"  下载成功率: {model_success_rate:.1f}%")
    report.append("")
    
    # 功能测试
    report.append("⚙️  功能验证状态:")
    for name, status in functionality.items():
        status_icon = "✅" if status else "❌"
        report.append(f"  {name}: {status_icon}")
    functionality_success_rate = sum(functionality.values()) / len(functionality) * 100 if functionality else 0
    report.append(f"  功能验证率: {functionality_success_rate:.1f}%")
    report.append("")
    
    # API密钥
    report.append("🔑 API密钥配置:")
    configured_keys = sum(api_keys.values())
    for name, status in api_keys.items():
        status_icon = "✅" if status else "⚠️ "
        report.append(f"  {name}: {status_icon}")
    report.append(f"  配置密钥数: {configured_keys}/{len(api_keys)}")
    report.append("")
    
    # 总体评估
    report.append("🎯 测试就绪评估:")
    
    # 必需条件检查
    essential_ready = (
        internet and 
        disk_space and 
        dependency_success_rate >= 80 and
        functionality_success_rate >= 80 and
        configured_keys >= 1  # 至少有一个API密钥
    )
    
    if essential_ready:
        report.append("  ✅ 环境已就绪，可以进行完整测试")
        report.append("  📋 建议测试流程:")
        report.append("    1. python tests/test_webtable_phase2.py")
        report.append("    2. python tests/test_webtable_phase2_optimized.py") 
        report.append("    3. python run_cli.py discover -q 'test' -t examples/webtable_join_tables.json")
    else:
        report.append("  ❌ 环境未完全就绪，存在以下问题:")
        if not internet:
            report.append("    • 需要网络连接下载模型")
        if not disk_space:
            report.append("    • 需要至少2GB可用磁盘空间")
        if dependency_success_rate < 80:
            report.append("    • 部分依赖包安装失败")
        if functionality_success_rate < 80:
            report.append("    • 功能验证存在问题")
        if configured_keys < 1:
            report.append("    • 需要配置至少一个LLM API密钥")
    
    # 可选优化建议
    report.append("")
    report.append("💡 可选优化建议:")
    if model_success_rate < 100:
        report.append("  • 建议下载所有嵌入模型以获得最佳性能")
    if configured_keys < 3:
        report.append("  • 配置多个API密钥可提供更好的备份")
    report.append("  • 考虑使用GPU加速以提升向量计算性能")
    report.append("  • 定期更新模型版本以获得最新性能")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    """主函数"""
    logger.info("🚀 开始完整测试环境检查和准备")
    
    # 1. 基础环境检查
    logger.info("🔍 检查基础环境...")
    internet = check_internet_connection()
    disk_space = check_disk_space()
    
    if not internet:
        logger.error("❌ 无网络连接，无法下载模型")
        logger.info("💡 请检查网络连接后重试")
        return False
    
    if not disk_space:
        logger.error("❌ 磁盘空间不足")
        logger.info("💡 请清理磁盘空间后重试")
        return False
    
    # 2. 安装依赖
    logger.info("📦 检查和安装依赖包...")
    dependencies = install_dependencies()
    
    # 3. 下载模型
    logger.info("🤖 下载嵌入模型...")
    models = download_sentence_transformer_models()
    
    # 4. 功能验证
    logger.info("⚙️  验证功能...")
    functionality = verify_model_functionality()
    
    # 5. API密钥检查
    logger.info("🔑 检查API密钥...")
    api_keys = check_api_keys()
    
    # 6. 生成报告
    report = generate_test_readiness_report(
        internet, disk_space, dependencies, models, functionality, api_keys
    )
    
    print(report)
    
    # 保存报告
    report_file = Path(__file__).parent.parent / "test_readiness_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"📄 详细报告已保存到: {report_file}")
    
    # 返回是否就绪
    essential_ready = (
        internet and 
        disk_space and 
        sum(dependencies.values()) / len(dependencies) >= 0.8 and
        sum(functionality.values()) / len(functionality) >= 0.8 and
        sum(api_keys.values()) >= 1
    )
    
    return essential_ready

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)