#!/usr/bin/env python
"""
诊断LLM调用卡住问题
"""

import asyncio
import time
import signal
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

# 设置超时信号
def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

async def test_gemini_direct():
    """直接测试Gemini API"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    # 配置API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = "Return JSON: {\"test\": true}"
    
    print("\n1. Testing direct Gemini call (with 10s timeout)...")
    
    # 设置10秒超时
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    
    try:
        start = time.time()
        response = model.generate_content(prompt)
        elapsed = time.time() - start
        signal.alarm(0)  # 取消超时
        
        print(f"✅ Success in {elapsed:.2f}s")
        print(f"Response: {response.text[:100]}")
        return True
    except TimeoutError:
        print("❌ TIMEOUT after 10 seconds - API call is hanging!")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        print(f"❌ Error: {e}")
        return False

async def test_async_gather():
    """测试异步gather是否有问题"""
    async def slow_task(n):
        print(f"  Task {n} started")
        await asyncio.sleep(2)
        print(f"  Task {n} completed")
        return n
    
    print("\n2. Testing asyncio.gather with 3 tasks...")
    start = time.time()
    
    tasks = [slow_task(i) for i in range(3)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    elapsed = time.time() - start
    print(f"✅ Completed in {elapsed:.2f}s")
    print(f"Results: {results}")

async def test_network_connectivity():
    """测试网络连接"""
    import socket
    
    print("\n3. Testing network connectivity...")
    
    # 测试DNS解析
    try:
        ip = socket.gethostbyname('generativelanguage.googleapis.com')
        print(f"✅ DNS resolved: generativelanguage.googleapis.com -> {ip}")
    except Exception as e:
        print(f"❌ DNS resolution failed: {e}")
        return False
    
    # 测试端口连接
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((ip, 443))
        sock.close()
        
        if result == 0:
            print(f"✅ Can connect to port 443")
            return True
        else:
            print(f"❌ Cannot connect to port 443")
            return False
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

async def check_proxy_settings():
    """检查代理设置"""
    print("\n4. Checking proxy settings...")
    
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    has_proxy = False
    
    for var in proxy_vars:
        value = os.getenv(var)
        if value:
            print(f"  {var} = {value}")
            has_proxy = True
    
    if not has_proxy:
        print("  No proxy configured")
    else:
        print("  ⚠️ Proxy is configured - this might cause issues with Gemini API")
    
    return not has_proxy

async def main():
    print("="*60)
    print("🔍 诊断LLM调用卡住问题")
    print("="*60)
    
    # 1. 检查网络
    network_ok = await test_network_connectivity()
    
    # 2. 检查代理
    proxy_ok = await check_proxy_settings()
    
    # 3. 测试asyncio
    await test_async_gather()
    
    # 4. 测试Gemini API
    if network_ok:
        api_ok = await test_gemini_direct()
    else:
        print("\n⚠️ Skipping API test due to network issues")
        api_ok = False
    
    # 诊断结果
    print("\n" + "="*60)
    print("📊 诊断结果:")
    print("="*60)
    
    if not network_ok:
        print("❌ 网络连接问题：无法连接到Google API服务器")
        print("   解决方案：")
        print("   1. 检查网络连接")
        print("   2. 检查防火墙设置")
        print("   3. 可能需要配置代理")
    elif not proxy_ok:
        print("⚠️ 代理配置可能影响API调用")
        print("   解决方案：")
        print("   1. 确认代理设置正确")
        print("   2. 或临时禁用代理测试")
    elif not api_ok:
        print("❌ API调用超时或失败")
        print("   可能原因：")
        print("   1. API密钥无效或超过配额")
        print("   2. 网络延迟太高")
        print("   3. API服务暂时不可用")
        print("   解决方案：")
        print("   1. 检查API密钥是否有效")
        print("   2. 添加超时机制")
        print("   3. 减少并发调用数量")
    else:
        print("✅ 所有测试通过！")
        print("   问题可能是：")
        print("   1. 并发20个调用太多，导致API限流")
        print("   2. 某些特定查询导致API响应特别慢")
        print("   建议：")
        print("   1. 减少并发数量（如改为3-5个）")
        print("   2. 添加单个调用超时机制")
    
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())