import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from three_layer_ablation_optimized import CacheManager
import json

# 创建缓存管理器
cm = CacheManager("cache/test_cache")

# 测试数据
query = {"query_table": "test_table", "task_type": "join"}
result = {"query_table": "test_table", "predictions": ["table1", "table2"]}

# 保存缓存
cm.set("test_op", query, result)

# 读取缓存
cached = cm.get("test_op", query)
print(f"缓存结果: {cached}")

# 打印统计
stats = cm.get_stats()
print(f"缓存统计: {stats}")

# 检查磁盘文件
import os
print(f"缓存文件: {os.listdir('cache/test_cache')}")
