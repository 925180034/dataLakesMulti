#!/usr/bin/env python3
"""
简化的模型下载和配置脚本
"""

import os
import sys
from pathlib import Path

def install_dependencies():
    """安装必要的依赖"""
    print("📦 安装依赖包...")
    
    dependencies = [
        "sentence-transformers>=2.7.0",
        "torch>=1.12.0", 
        "transformers>=4.21.0",
        "numpy>=1.21.0"
    ]
    
    import subprocess
    for dep in dependencies:
        print(f"安装 {dep}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", dep
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {dep} 安装成功")
        else:
            print(f"❌ {dep} 安装失败: {result.stderr}")
            
def download_embedding_model():
    """下载嵌入模型"""
    print("\n🤖 下载嵌入模型...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("下载 all-MiniLM-L6-v2 模型（约90MB）...")
        print("首次下载可能需要几分钟，请耐心等待...")
        
        # 下载并初始化模型
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 测试模型
        print("测试模型功能...")
        test_embedding = model.encode("这是一个测试句子")
        
        print(f"✅ 模型下载成功！")
        print(f"   模型维度: {len(test_embedding)}")
        print(f"   缓存位置: ~/.cache/huggingface/transformers/")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        return False

def setup_api_keys():
    """设置API密钥的指导"""
    print("\n🔑 API密钥配置指南")
    print("=" * 50)
    
    print("\n选择一个LLM服务商（推荐Gemini，免费且稳定）：")
    
    print("\n1️⃣ Google Gemini (推荐)")
    print("   • 免费，每分钟15次请求")
    print("   • 注册地址: https://aistudio.google.com/")
    print("   • 获取API密钥后，设置环境变量:")
    print("     export GEMINI_API_KEY='你的密钥'")
    
    print("\n2️⃣ OpenAI")
    print("   • 付费服务，$0.002/1K tokens")
    print("   • 注册地址: https://platform.openai.com/")
    print("   • 设置环境变量:")
    print("     export OPENAI_API_KEY='你的密钥'")
    
    print("\n3️⃣ Anthropic Claude")
    print("   • 付费服务，有免费额度")
    print("   • 注册地址: https://console.anthropic.com/")
    print("   • 设置环境变量:")
    print("     export ANTHROPIC_API_KEY='你的密钥'")

def check_current_setup():
    """检查当前配置状态"""
    print("\n🔍 检查当前配置状态")
    print("=" * 50)
    
    # 检查模型
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ 嵌入模型: 已安装")
    except:
        print("❌ 嵌入模型: 未安装")
    
    # 检查API密钥
    api_keys = {
        'GEMINI_API_KEY': 'Gemini',
        'OPENAI_API_KEY': 'OpenAI', 
        'ANTHROPIC_API_KEY': 'Anthropic'
    }
    
    configured_keys = 0
    for key, name in api_keys.items():
        if os.getenv(key):
            print(f"✅ {name} API: 已配置")
            configured_keys += 1
        else:
            print(f"❌ {name} API: 未配置")
    
    return configured_keys > 0

def main():
    print("🚀 数据湖多智能体系统 - 模型配置向导")
    print("=" * 60)
    
    # 1. 安装依赖
    install_dependencies()
    
    # 2. 下载模型
    model_success = download_embedding_model()
    
    # 3. API密钥指导
    setup_api_keys()
    
    # 4. 检查配置
    print("\n" + "=" * 60)
    has_api_key = check_current_setup()
    
    # 5. 下一步指导
    print("\n📋 下一步操作:")
    if model_success and has_api_key:
        print("✅ 配置完成！可以运行完整测试:")
        print("   python tests/test_webtable_phase2.py")
        print("   python run_cli.py discover -q 'test' -t examples/webtable_join_tables.json")
    elif model_success:
        print("⚠️  模型已下载，但需要配置API密钥")
        print("   请按上述指南配置API密钥")
    else:
        print("❌ 模型下载失败，请检查网络连接")
        print("   或手动安装: pip install sentence-transformers")

if __name__ == "__main__":
    main()