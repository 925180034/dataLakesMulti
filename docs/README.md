# 📚 数据湖多智能体系统文档中心

欢迎来到数据湖多智能体系统的文档中心。本系统采用**LangGraph多智能体协同架构**，集成**三层加速工具**，实现高效的数据湖发现（Data Lake Discovery）。

## 🎯 核心架构文档（最新版）

### 系统实现与架构
1. **[系统实现文档](SYSTEM_IMPLEMENTATION.md)** ⭐⭐⭐
   - LangGraph多智能体系统详细实现
   - 6个专门Agent技术细节
   - 三层加速工具集成
   - 性能指标与优化方向
   - 当前系统运行方式

2. **[完整系统架构](COMPLETE_SYSTEM_ARCHITECTURE.md)** ⭐⭐⭐
   - 多智能体系统设计理念
   - 数据湖发现能力说明
   - 技术栈与框架选择
   - 系统部署与扩展策略

3. **[架构图表集合](ARCHITECTURE_DIAGRAMS.md)** ⭐⭐
   - 系统架构可视化
   - 数据流程图
   - Agent协同流程
   - 性能指标对比

### 快速开始
1. **[快速开始指南](QUICK_START.md)**
   - 环境搭建步骤
   - 基本使用示例
   - 常见问题解答

2. **[项目设计文档](Project-Design-Document.md)**
   - 原始设计理念
   - 多智能体协作框架
   - 数据格式说明

### 技术分析

3. **[LakeBench项目分析](lakebench_analysis.md)**
   - LakeBench论文深度解析
   - 11种数据发现算法对比
   - 性能优化insights

4. **[WebTable测试报告](WEBTABLE_TEST_REPORT.md)**
   - WebTable数据集测试结果
   - 性能基准测试
   - 问题分析与解决方案

5. **[环境需求说明](environment_requirements.md)**
   - 硬件要求
   - 软件依赖
   - 配置建议

## 🚀 快速导航

### 新用户入门
1. 阅读 [快速开始指南](QUICK_START.md)
2. 查看 [系统架构与实施计划](SYSTEM_ARCHITECTURE_AND_PLAN.md) 的环境配置部分
3. 运行第一个实验

### 开发者指南
1. 深入了解 [项目设计文档](Project-Design-Document.md)
2. 研究 [LakeBench项目分析](lakebench_analysis.md)
3. 查看架构实现细节

### 系统运行与测试
1. 使用LangGraph多智能体系统：
   ```bash
   python run_langgraph_system.py --dataset subset --max-queries 5 --task join
   ```
2. 查看测试结果和性能指标
3. 分析匹配质量和系统性能

## 📊 实验运行指南

### 推荐实验流程

1. **快速测试（5个查询）**
   ```bash
   python run_langgraph_system.py --dataset subset --max-queries 5 --task join
   ```

2. **标准测试（JOIN任务）**
   ```bash
   python run_langgraph_system.py --dataset subset --max-queries 10 --task join
   ```

3. **完整测试（JOIN+UNION）**
   ```bash
   python run_langgraph_system.py --dataset subset --max-queries 10 --task both
   ```

4. **大规模测试（完整数据集）**
   ```bash
   python run_langgraph_system.py --dataset complete --max-queries 10 --task join
   ```

### 评价指标说明

系统自动计算以下指标：
- **Precision（精确率）**：返回结果的准确性
- **Recall（召回率）**：找到所有正确答案的能力
- **F1-Score**：精确率和召回率的综合指标
- **成功率**：查询完成的比例
- **响应时间**：平均处理时间

结果自动输出到控制台，可使用 `--output` 参数保存到文件。

## 🔧 配置优化

### 环境变量配置
创建 `.env` 文件：
```bash
# 推荐使用Gemini API（免费且稳定）
GEMINI_API_KEY=your_gemini_api_key

# 或使用其他LLM API
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# 可选配置
CACHE_ENABLED=true
DEBUG=false
```

### 系统配置
编辑 `config.yml` 调整系统参数：
```yaml
llm:
  provider: "gemini"  # 或 "openai", "anthropic"
  
performance:
  batch_size: 10
  timeout: 60
```

## 📈 系统状态

- ✅ **LangGraph多智能体系统**：完全实现并运行稳定
- ✅ **6个专门Agent**：协同工作，各司其职
- ✅ **三层加速架构**：Layer 1-3全部集成
- ✅ **性能表现**：10-15秒/查询，100%成功率
- 🔄 **性能优化**：Layer 2向量搜索优化中

## 🤝 贡献指南

欢迎贡献代码和文档！请：
1. 遵循现有的文档格式
2. 更新相关的索引和链接
3. 运行测试确保功能正常

## 📞 获取帮助

- 查看 [快速开始指南](QUICK_START.md) 的故障排除部分
- 参考 [系统实现文档](SYSTEM_IMPLEMENTATION.md) 了解技术细节
- 查看系统输出日志进行问题诊断

---

**文档更新日期**: 2024-08-12  
**版本**: v4.0 - LangGraph多智能体系统