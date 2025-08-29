# 多进程重复初始化问题修复方案

## 问题诊断

### 核心问题
1. **64个进程各自初始化资源**
   - 每个进程占用11GB+内存
   - 总计占用715GB+内存
   - SentenceTransformer模型被加载64次
   - LLMMatcherTool被重复初始化

2. **性能瓶颈**
   - OpenData complete: 3595个查询（500 JOIN + 3095 UNION）
   - 每个查询创建一个进程任务
   - 进程间无法共享Python对象

## 已完成的修复

### 1. 降低默认进程数
- 从64降到8-16个进程
- Complete数据集自动限制为16个
- OpenData complete特别限制

### 2. 创建优化版脚本
- `run_unified_experiment_optimized.py`: 批处理优化版本
- 主进程初始化一次，子进程复用

### 3. 修改原脚本
- `run_unified_experiment.py`: 添加进程数限制和警告

## 立即运行的解决方案

### 方案1: 使用优化版脚本（推荐）
```bash
# 测试小批量
python run_unified_experiment_optimized.py \
    --dataset opendata \
    --task join \
    --dataset-type subset \
    --max-queries 50 \
    --workers 8 \
    --layer L1+L2

# 如果成功，运行完整版
python run_unified_experiment_optimized.py \
    --dataset opendata \
    --task both \
    --dataset-type complete \
    --max-queries 100 \
    --workers 8 \
    --layer all
```

### 方案2: 使用修复的原脚本
```bash
# 限制查询数和进程数
python run_unified_experiment.py \
    --dataset opendata \
    --task join \
    --dataset-type complete \
    --max-queries 100 \
    --workers 8 \
    --layer L1+L2
```

### 方案3: 分批运行
```bash
# 分成多个小批次运行
for i in 0 100 200 300 400; do
    python run_unified_experiment.py \
        --dataset opendata \
        --task join \
        --dataset-type complete \
        --max-queries 100 \
        --workers 8 \
        --layer L1+L2 \
        --skip-offset $i
done
```

## 长期优化方案

### 1. 使用Ray或Dask
替换ProcessPoolExecutor为分布式计算框架：
- Ray: 更好的内存共享
- Dask: 自动任务调度

### 2. 缓存优化
- 使用Redis作为共享缓存
- 内存映射文件（mmap）共享大数据

### 3. 批处理架构
- 将查询分组批处理
- 减少进程创建开销

## 监控和验证

### 检查进程状态
```bash
# 监控内存使用
watch -n 1 "ps aux | grep python | head -20"

# 检查总内存
free -h

# 终止失控进程
pkill -f "run_unified_experiment.*opendata"
```

### 验证缓存
```bash
# 检查缓存文件
ls -lh cache/opendata/

# 验证嵌入是真实的
python -c "
import pickle
with open('cache/opendata/table_embeddings_500.pkl', 'rb') as f:
    emb = pickle.load(f)
print(f'Embeddings: {len(emb)} tables')
print(f'Dimension: {len(list(emb.values())[0])}')
"
```

## 性能预期

| 配置 | 查询数 | 进程数 | 预计时间 | 内存使用 |
|------|--------|--------|----------|----------|
| 优化前 | 3595 | 64 | 4-5小时+ | 700GB+ |
| 优化后 | 100 | 8 | 10-15分钟 | 50GB |
| 优化后 | 500 | 8 | 30-45分钟 | 50GB |
| 优化后 | 3595 | 8 | 2-3小时 | 50GB |

## 结论

通过以下优化，可以大幅提升性能：
1. ✅ 减少进程数（64→8）
2. ✅ 主进程初始化，子进程复用
3. ✅ 批处理查询
4. ✅ 使用真实嵌入缓存
5. ✅ 添加资源限制和警告

**立即可用的命令**：
```bash
# 安全测试版本
python run_unified_experiment.py \
    --dataset opendata \
    --task join \
    --dataset-type subset \
    --max-queries 50 \
    --workers 8 \
    --layer L1+L2
```