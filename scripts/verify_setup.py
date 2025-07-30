#!/usr/bin/env python3
"""
配置验证脚本 - 检查模型和API密钥是否正确配置
"""

import os
import sys
from pathlib import Path

def load_env_file():
    """加载.env文件"""
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✅ 加载.env文件: {env_file}")
    else:
        print(f"⚠️  .env文件不存在: {env_file}")

# 在脚本开始时加载.env文件
load_env_file()

def check_model():
    """检查嵌入模型"""
    print("🤖 检查嵌入模型...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # 尝试加载模型
        print("  加载 all-MiniLM-L6-v2...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 测试模型功能
        print("  测试模型功能...")
        embedding = model.encode("测试文本")
        
        print(f"  ✅ 模型正常工作")
        print(f"     维度: {len(embedding)}")
        print(f"     类型: {type(embedding)}")
        
        return True
        
    except ImportError:
        print("  ❌ sentence-transformers 未安装")
        print("     解决方案: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        print("     解决方案: 检查网络连接并重新下载")
        return False

def check_api_keys():
    """检查API密钥配置"""
    print("\n🔑 检查API密钥...")
    
    api_keys = {
        'GEMINI_API_KEY': {
            'name': 'Google Gemini',
            'pattern': 'AIzaSy',
            'length': 39
        },
        'OPENAI_API_KEY': {
            'name': 'OpenAI',
            'pattern': 'sk-',
            'length': 51  # 大概长度
        },
        'ANTHROPIC_API_KEY': {
            'name': 'Anthropic Claude',
            'pattern': 'sk-ant-',
            'length': 108  # 大概长度
        }
    }
    
    configured_keys = []
    
    for key_name, info in api_keys.items():
        key_value = os.getenv(key_name)
        
        if key_value:
            print(f"  ✅ {info['name']}: 已配置")
            
            # 验证密钥格式
            if key_value.startswith(info['pattern']):
                print(f"     格式: 正确 (以 {info['pattern']} 开头)")
            else:
                print(f"     ⚠️  格式可能有误 (应以 {info['pattern']} 开头)")
            
            # 验证长度
            if abs(len(key_value) - info['length']) <= 10:
                print(f"     长度: 正常 ({len(key_value)} 字符)")
            else:
                print(f"     ⚠️  长度可能有误 ({len(key_value)} 字符，预期约 {info['length']})")
            
            configured_keys.append(info['name'])
        else:
            print(f"  ❌ {info['name']}: 未配置")
    
    return len(configured_keys) > 0, configured_keys

def check_system_config():
    """检查系统配置"""
    print("\n⚙️  检查系统配置...")
    
    try:
        # 添加项目路径
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from src.config.settings import settings
        
        print(f"  LLM提供商: {settings.llm.provider}")
        print(f"  模型名称: {settings.llm.model_name}")
        print(f"  向量维度: {settings.vector_db.dimension}")
        print(f"  数据目录: {settings.data_dir}")
        
        # 检查必要目录
        data_dir = Path(settings.data_dir)
        if data_dir.exists():
            print(f"  ✅ 数据目录存在: {data_dir}")
        else:
            print(f"  ⚠️  数据目录不存在，将自动创建: {data_dir}")
            data_dir.mkdir(parents=True, exist_ok=True)
        
        return True
        
    except Exception as e:
        print(f"  ❌ 系统配置检查失败: {e}")
        return False

def test_api_connection():
    """测试API连接"""
    print("\n🌐 测试API连接...")
    
    try:
        # 添加项目路径
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from src.utils.llm_client import get_llm_client
        
        print("  获取LLM客户端...")
        client = get_llm_client()
        
        print("  发送测试请求...")
        response = client.generate_text(
            "请回答：1+1=?",
            max_tokens=50
        )
        
        if response and len(response.strip()) > 0:
            print(f"  ✅ API连接正常")
            print(f"     响应: {response[:50]}...")
            return True
        else:
            print(f"  ❌ API响应为空")
            return False
            
    except Exception as e:
        print(f"  ❌ API连接失败: {e}")
        print(f"     建议检查API密钥和网络连接")
        return False

def run_mini_test():
    """运行小型功能测试"""
    print("\n🧪 运行功能测试...")
    
    try:
        # 添加项目路径
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        # 测试嵌入生成
        from src.tools.embedding import get_embedding_generator
        
        print("  测试嵌入生成...")
        generator = get_embedding_generator()
        
        # 生成测试嵌入
        import asyncio
        async def test_embedding():
            embedding = await generator.generate_text_embedding("测试文本")
            return embedding
        
        embedding = asyncio.run(test_embedding())
        
        if embedding and len(embedding) > 0:
            print(f"  ✅ 嵌入生成正常 (维度: {len(embedding)})")
            return True
        else:
            print(f"  ❌ 嵌入生成失败")
            return False
            
    except Exception as e:
        print(f"  ❌ 功能测试失败: {e}")
        return False

def main():
    """主验证流程"""
    print("🔍 数据湖多智能体系统 - 配置验证")
    print("=" * 60)
    
    results = {}
    
    # 1. 检查模型
    results['model'] = check_model()
    
    # 2. 检查API密钥
    has_api_key, configured_keys = check_api_keys()
    results['api_key'] = has_api_key
    
    # 3. 检查系统配置
    results['system'] = check_system_config()
    
    # 4. 测试API连接（如果有密钥）
    if has_api_key:
        results['api_connection'] = test_api_connection()
    else:
        results['api_connection'] = False
        print("\n🌐 跳过API连接测试（无API密钥）")
    
    # 5. 运行功能测试
    if results['model']:
        results['functionality'] = run_mini_test()
    else:
        results['functionality'] = False
        print("\n🧪 跳过功能测试（模型未就绪）")
    
    # 生成总结报告
    print("\n" + "=" * 60)
    print("📊 验证结果总结")
    print("=" * 60)
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    for check, status in results.items():
        status_icon = "✅" if status else "❌"
        check_name = {
            'model': '嵌入模型',
            'api_key': 'API密钥', 
            'system': '系统配置',
            'api_connection': 'API连接',
            'functionality': '功能测试'
        }.get(check, check)
        
        print(f"{status_icon} {check_name}")
    
    success_rate = passed_checks / total_checks * 100
    print(f"\n✨ 总体成功率: {success_rate:.1f}% ({passed_checks}/{total_checks})")
    
    # 给出建议
    if success_rate >= 80:
        print("\n🎉 配置验证通过！可以运行完整测试:")
        print("   python tests/test_webtable_phase2.py")
        print("   python run_cli.py discover -q 'test' -t examples/webtable_join_tables.json")
    elif success_rate >= 60:
        print("\n⚠️  基本配置完成，但建议优化:")
        if not results['model']:
            print("   • 下载嵌入模型: python scripts/setup_models.py")
        if not results['api_key']:
            print("   • 配置API密钥: 参考 API_SETUP_GUIDE.md")
        if not results['api_connection']:
            print("   • 检查网络连接和API密钥有效性")
    else:
        print("\n❌ 配置需要完善，建议:")
        print("   1. 运行: python scripts/setup_models.py")
        print("   2. 参考: API_SETUP_GUIDE.md 配置API密钥")
        print("   3. 重新运行验证: python scripts/verify_setup.py")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)