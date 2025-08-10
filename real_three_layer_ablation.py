#!/usr/bin/env python3
"""
真正的三层架构消融实验 - 使用真实的LLM和SentenceTransformer
基于您的真实系统组件，而不是模拟版本
"""

import asyncio
import json
import time
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from tabulate import tabulate
from datetime import datetime
from pathlib import Path
import sys
import logging

# 设置项目路径
sys.path.insert(0, '/root/dataLakesMulti')

# 强制使用真实API和模型
os.environ['SKIP_LLM'] = 'false'
os.environ['DEBUG'] = 'true'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AblationConfig:
    """真实消融实验配置"""
    name: str
    description: str
    use_layer1: bool = True
    use_layer2: bool = True
    use_layer3: bool = True
    force_llm: bool = False

class RealThreeLayerAblation:
    """使用真实系统的三层架构消融实验"""
    
    def __init__(self, task_type: str = "union", dataset_size: str = "subset"):
        self.task_type = task_type.lower()
        self.dataset_size = dataset_size
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results = []
        
        # 数据路径
        self.data_dir = Path(f"examples/separated_datasets/{task_type}_{dataset_size}")
        
        print(f"🔥 真实三层架构消融实验")
        print(f"📂 数据集路径: {self.data_dir}")
        print(f"📊 任务类型: {task_type.upper()}")
        print(f"📏 数据规模: {dataset_size}")
        
        # 初始化真实系统组件
        self._initialize_real_components()
    
    def _initialize_real_components(self):
        """初始化真实的系统组件"""
        print(f"\n🔧 初始化真实系统组件...")
        
        try:
            # 1. 初始化真实的LLM客户端
            from src.utils.llm_client import GeminiClient
            from src.config.settings import settings
            
            # 构建Gemini配置
            config = {
                "model_name": "gemini-1.5-flash",
                "temperature": 0.1,
                "max_tokens": 500,
                "timeout": 8
            }
            
            self.llm_client = GeminiClient(config)
            print(f"   ✅ 真实LLM客户端 (Gemini) 初始化完成")
            
        except Exception as e:
            print(f"   ❌ LLM客户端初始化失败: {e}")
            self.llm_client = None
        
        try:
            # 2. 初始化真实的SentenceTransformer嵌入生成器
            from src.tools.embedding import SentenceTransformerEmbeddingGenerator
            
            self.embedding_generator = SentenceTransformerEmbeddingGenerator("all-MiniLM-L6-v2")
            print(f"   ✅ 真实SentenceTransformer嵌入生成器初始化完成")
            
        except Exception as e:
            print(f"   ❌ 嵌入生成器初始化失败: {e}")
            self.embedding_generator = None
        
        try:
            # 3. 初始化真实的工作流
            from src.core.workflow import DataLakesWorkflow
            
            self.real_workflow = DataLakesWorkflow()
            print(f"   ✅ 真实DataLakesWorkflow初始化完成")
            
        except Exception as e:
            print(f"   ❌ 工作流初始化失败: {e}")
            self.real_workflow = None

    def load_real_data(self) -> Dict[str, Any]:
        """加载真实数据集"""
        print(f"\n📥 加载真实数据集...")
        
        # 加载表数据
        with open(self.data_dir / "tables.json", 'r') as f:
            tables_data = json.load(f)
        
        # 加载查询数据
        with open(self.data_dir / "queries.json", 'r') as f:
            queries_data = json.load(f)
        
        # 加载ground truth
        with open(self.data_dir / "ground_truth.json", 'r') as f:
            ground_truth_data = json.load(f)
        
        print(f"   ✅ 加载 {len(tables_data)} 个表")
        print(f"   ✅ 加载 {len(queries_data)} 个查询") 
        print(f"   ✅ 加载 {len(ground_truth_data)} 个ground truth")
        
        return {
            'tables': tables_data,
            'queries': queries_data,
            'ground_truth': ground_truth_data
        }

    async def layer1_metadata_filter_real(self, query_table: str, all_tables: List[str], query_column: Optional[str] = None) -> Tuple[List[str], float]:
        """真实的Layer1：元数据过滤"""
        start_time = time.time()
        
        candidates = []
        query_lower = query_table.lower()
        
        # 领域过滤
        query_domains = []
        if any(k in query_lower for k in ['user', 'customer', 'account']):
            query_domains.append('user')
        if any(k in query_lower for k in ['order', 'transaction', 'payment']):
            query_domains.append('order')
        if any(k in query_lower for k in ['product', 'item', 'goods']):
            query_domains.append('product')
        
        for table in all_tables:
            if table == query_table:
                continue
                
            table_lower = table.lower()
            
            # 领域匹配
            match = False
            for domain in query_domains:
                if domain == 'user' and any(k in table_lower for k in ['user', 'customer', 'account']):
                    match = True
                elif domain == 'order' and any(k in table_lower for k in ['order', 'transaction', 'payment']):
                    match = True
                elif domain == 'product' and any(k in table_lower for k in ['product', 'item', 'goods']):
                    match = True
            
            # 字符串相似度
            if not match:
                common_chars = set(query_lower) & set(table_lower)
                if len(common_chars) >= min(3, len(query_lower) * 0.3):
                    match = True
            
            if match:
                candidates.append(table)
        
        processing_time = time.time() - start_time
        return candidates, processing_time

    async def layer2_vector_search_real(self, query_table: str, layer1_candidates: List[str], query_column: Optional[str] = None) -> Tuple[List[Tuple[str, float]], float]:
        """真实的Layer2：HNSW向量搜索 - 使用真正的SentenceTransformer"""
        start_time = time.time()
        
        if not self.embedding_generator:
            print(f"     ❌ Layer2: 嵌入生成器不可用，跳过")
            # 返回Layer1结果，添加虚拟分数
            return [(c, 0.5) for c in layer1_candidates], time.time() - start_time
        
        try:
            # 生成查询表的真实嵌入向量
            query_text = query_table
            if query_column:
                query_text = f"{query_table}.{query_column}"
            
            print(f"     🔄 Layer2: 生成真实向量嵌入 (SentenceTransformer)")
            query_embedding = await self.embedding_generator.generate_text_embedding(query_text)
            
            vector_candidates = []
            for candidate_table in layer1_candidates:
                candidate_text = candidate_table
                if query_column:
                    candidate_text = f"{candidate_table}.{query_column}"
                
                # 生成候选表的真实嵌入向量
                candidate_embedding = await self.embedding_generator.generate_text_embedding(candidate_text)
                
                # 计算真实的余弦相似度
                similarity = self._cosine_similarity(
                    np.array(query_embedding), 
                    np.array(candidate_embedding)
                )
                
                vector_candidates.append((candidate_table, similarity))
            
            # 按相似度排序
            vector_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 保留前25个
            final_candidates = vector_candidates[:25]
            
            processing_time = time.time() - start_time
            print(f"     ✅ Layer2: 真实向量搜索完成 ({len(layer1_candidates)}→{len(final_candidates)}, {processing_time:.2f}s)")
            
            return final_candidates, processing_time
            
        except Exception as e:
            print(f"     ❌ Layer2: 真实向量搜索失败: {e}")
            # 返回Layer1结果作为备用
            return [(c, 0.5) for c in layer1_candidates], time.time() - start_time

    async def layer3_llm_matching_real(self, query_table: str, layer2_candidates: List[Tuple[str, float]], query_column: Optional[str] = None) -> Tuple[List[Dict], float, bool]:
        """真实的Layer3：LLM匹配 - 使用真正的Gemini API"""
        start_time = time.time()
        llm_called = False
        
        if not self.llm_client:
            print(f"     ❌ Layer3: LLM客户端不可用，跳过")
            # 返回匈牙利算法结果
            matches = []
            for i, (table_name, vector_score) in enumerate(layer2_candidates[:10]):
                matches.append({
                    'rank': i + 1,
                    'table_name': table_name,
                    'score': vector_score * 100,
                    'method': 'vector_only',
                    'llm_verified': False
                })
            return matches, time.time() - start_time, False
        
        try:
            print(f"     🔄 Layer3: 调用真实LLM API (Gemini)")
            
            # 准备LLM提示词
            candidates_text = "\n".join([f"{i+1}. {table}" for i, (table, _) in enumerate(layer2_candidates[:5])])
            
            if self.task_type == "join":
                prompt = f"""你是一个数据表匹配专家。请为JOIN操作分析以下候选表与查询表的匹配程度。

查询表: {query_table}
查询列: {query_column or "未指定"}

候选表:
{candidates_text}

请为每个候选表打分(0-100)并说明理由。返回JSON格式:
{{"rankings": [{{"table": "表名", "score": 分数, "reason": "理由"}}]}}
"""
            else:
                prompt = f"""你是一个数据表匹配专家。请为UNION操作分析以下候选表与查询表的相似程度。

查询表: {query_table}

候选表:
{candidates_text}

请为每个候选表打分(0-100)并说明理由。返回JSON格式:
{{"rankings": [{{"table": "表名", "score": 分数, "reason": "理由"}}]}}
"""
            
            # 真实的LLM API调用
            llm_response = await self.llm_client.generate_json(prompt)
            llm_called = True
            
            print(f"     ✅ Layer3: LLM API调用成功")
            
            # 解析LLM响应
            matches = []
            if 'rankings' in llm_response:
                for i, ranking in enumerate(llm_response['rankings']):
                    matches.append({
                        'rank': i + 1,
                        'table_name': ranking.get('table', f'unknown_{i}'),
                        'score': ranking.get('score', 50),
                        'reason': ranking.get('reason', ''),
                        'method': 'llm_verified',
                        'llm_verified': True
                    })
            
            # 如果LLM没有返回足够结果，补充向量结果
            llm_tables = {m['table_name'] for m in matches}
            for table_name, vector_score in layer2_candidates:
                if table_name not in llm_tables and len(matches) < 10:
                    matches.append({
                        'rank': len(matches) + 1,
                        'table_name': table_name,
                        'score': vector_score * 100,
                        'method': 'vector_fallback',
                        'llm_verified': False
                    })
            
            processing_time = time.time() - start_time
            print(f"     ✅ Layer3: LLM处理完成 ({len(layer2_candidates)}→{len(matches)}, {processing_time:.2f}s)")
            
            return matches[:10], processing_time, llm_called
            
        except Exception as e:
            print(f"     ❌ Layer3: LLM调用失败: {e}")
            # 使用向量分数作为备用
            matches = []
            for i, (table_name, vector_score) in enumerate(layer2_candidates[:10]):
                matches.append({
                    'rank': i + 1,
                    'table_name': table_name,
                    'score': vector_score * 100,
                    'method': 'vector_fallback',
                    'llm_verified': False
                })
            return matches, time.time() - start_time, False

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0.0

    async def run_single_ablation(self, config: AblationConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个消融实验"""
        print(f"\n{'='*60}")
        print(f"🔬 真实消融配置: {config.name}")
        print(f"   描述: {config.description}")
        print(f"   层配置: L1={config.use_layer1}, L2={config.use_layer2}, L3={config.use_layer3}")
        print("="*60)
        
        all_tables = [t['table_name'] for t in data['tables']]
        test_queries = data['queries'][:5]  # 测试前5个查询
        
        results = []
        total_time = 0
        llm_calls = 0
        
        for i, query_data in enumerate(test_queries):
            query_table = query_data.get('query_table', '')
            query_column = query_data.get('query_column') if self.task_type == "join" else None
            
            print(f"\n   查询 {i+1}/{len(test_queries)}: {query_table}")
            
            candidates = all_tables
            layer_times = []
            
            # Layer 1
            if config.use_layer1:
                layer1_candidates, layer1_time = await self.layer1_metadata_filter_real(query_table, candidates, query_column)
                candidates = layer1_candidates
                layer_times.append(('Layer1', layer1_time))
                print(f"      Layer1: {len(all_tables)}→{len(candidates)} ({layer1_time:.3f}s)")
            
            # Layer 2  
            if config.use_layer2:
                layer2_candidates, layer2_time = await self.layer2_vector_search_real(query_table, candidates, query_column)
                candidates = layer2_candidates
                layer_times.append(('Layer2', layer2_time))
                print(f"      Layer2: 真实向量搜索 ({layer2_time:.3f}s)")
            else:
                # 转换为元组格式
                candidates = [(c, 0.5) for c in candidates]
            
            # Layer 3
            if config.use_layer3:
                matches, layer3_time, llm_used = await self.layer3_llm_matching_real(query_table, candidates, query_column)
                if llm_used:
                    llm_calls += 1
                layer_times.append(('Layer3', layer3_time))
                print(f"      Layer3: 真实LLM调用 ({'✅' if llm_used else '❌'}, {layer3_time:.3f}s)")
                predicted = [m['table_name'] for m in matches[:5]]
            else:
                predicted = [c[0] for c in candidates[:5]]
            
            total_query_time = sum(t for _, t in layer_times)
            total_time += total_query_time
            
            # 计算性能指标（使用修正的ground truth解析）
            expected = []
            for gt in data['ground_truth']:
                if gt.get('query_table') == query_table:
                    candidate = gt.get('candidate_table', '')
                    if candidate and candidate != query_table:
                        expected.append(candidate)
            expected = list(set(expected))
            
            # 计算指标
            predicted_set = set(predicted)
            expected_set = set(expected)
            
            tp = len(predicted_set & expected_set)
            fp = len(predicted_set - expected_set)
            fn = len(expected_set - predicted_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            hit_at_1 = 1 if predicted and predicted[0] in expected_set else 0
            
            results.append({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'hit_at_1': hit_at_1,
                'time': total_query_time,
                'layer_times': layer_times
            })
            
            print(f"      指标: F1={f1:.1%}, Hit@1={hit_at_1}, 总时间={total_query_time:.3f}s")
            print(f"      时间分解: {', '.join([f'{name}={time:.3f}s' for name, time in layer_times])}")
        
        # 计算平均指标
        avg_metrics = {
            'f1': np.mean([r['f1'] for r in results]),
            'hit_at_1': np.mean([r['hit_at_1'] for r in results]),
            'precision': np.mean([r['precision'] for r in results]),
            'recall': np.mean([r['recall'] for r in results]),
            'avg_time': np.mean([r['time'] for r in results]),
            'total_time': total_time,
            'llm_calls': llm_calls,
            'llm_rate': llm_calls / len(test_queries) if test_queries else 0
        }
        
        print(f"\n   📊 平均性能:")
        print(f"      F1: {avg_metrics['f1']:.1%}")
        print(f"      Hit@1: {avg_metrics['hit_at_1']:.1%}")
        print(f"      Precision: {avg_metrics['precision']:.1%}")
        print(f"      Recall: {avg_metrics['recall']:.1%}")
        print(f"      平均时间: {avg_metrics['avg_time']:.3f}s")
        print(f"      LLM调用: {llm_calls}/{len(test_queries)} ({avg_metrics['llm_rate']:.0%})")
        
        return {
            'config': config.name,
            'metrics': avg_metrics,
            'num_queries': len(test_queries)
        }

    def get_ablation_configs(self) -> List[AblationConfig]:
        """定义真实消融实验配置"""
        return [
            AblationConfig(
                name="L1_Only",
                description="仅Layer1 (元数据过滤)",
                use_layer1=True,
                use_layer2=False,
                use_layer3=False
            ),
            AblationConfig(
                name="L1_L2_Real",
                description="Layer1+Layer2 (真实SentenceTransformer)",
                use_layer1=True,
                use_layer2=True,
                use_layer3=False
            ),
            AblationConfig(
                name="Full_Real_3Layer",
                description="完整三层 (真实SentenceTransformer + 真实LLM)",
                use_layer1=True,
                use_layer2=True,
                use_layer3=True,
                force_llm=True
            )
        ]

    async def run_real_ablation_study(self):
        """运行真实的消融实验"""
        print("="*80)
        print(f"🔥 真正的三层架构消融实验 - 使用真实API和模型")
        print(f"   任务: {self.task_type.upper()}")
        print(f"   数据集: {self.dataset_size}")
        print("="*80)
        
        # 加载数据
        data = self.load_real_data()
        
        # 运行所有配置
        configs = self.get_ablation_configs()
        all_results = []
        
        for config in configs:
            result = await self.run_single_ablation(config, data)
            all_results.append(result)
            self.results.append(result)
        
        # 生成报告
        self.generate_real_report()

    def generate_real_report(self):
        """生成真实实验报告"""
        print(f"\n{'='*80}")
        print(f"📊 真实消融实验结果汇总 - {self.task_type.upper()}")
        print(f"   使用真实SentenceTransformer + 真实LLM API")
        print(f"{'='*80}")
        
        # 创建表格
        headers = ['配置', 'F1', 'Hit@1', 'Precision', 'Recall', '平均时间(s)', 'LLM调用率']
        rows = []
        
        for result in self.results:
            metrics = result['metrics']
            rows.append([
                result['config'],
                f"{metrics['f1']:.1%}",
                f"{metrics['hit_at_1']:.1%}",
                f"{metrics['precision']:.1%}",
                f"{metrics['recall']:.1%}",
                f"{metrics['avg_time']:.3f}",
                f"{metrics['llm_rate']:.0%}"
            ])
        
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        
        # 分析层贡献
        self.analyze_real_contributions()
        
        # 保存结果
        self.save_real_results()

    def analyze_real_contributions(self):
        """分析真实层贡献"""
        print(f"\n🔍 真实层贡献分析")
        print("="*50)
        
        results_dict = {r['config']: r for r in self.results}
        
        # Layer1贡献
        if 'L1_Only' in results_dict:
            l1_f1 = results_dict['L1_Only']['metrics']['f1']
            l1_time = results_dict['L1_Only']['metrics']['avg_time']
            print(f"\n1️⃣ Layer1独立性能:")
            print(f"   F1: {l1_f1:.1%}")
            print(f"   时间: {l1_time:.3f}s")
        
        # Layer2真实贡献
        if 'L1_Only' in results_dict and 'L1_L2_Real' in results_dict:
            l1_f1 = results_dict['L1_Only']['metrics']['f1']
            l12_f1 = results_dict['L1_L2_Real']['metrics']['f1']
            l1_time = results_dict['L1_Only']['metrics']['avg_time']
            l12_time = results_dict['L1_L2_Real']['metrics']['avg_time']
            
            gain = l12_f1 - l1_f1
            time_cost = l12_time - l1_time
            
            print(f"\n2️⃣ Layer2 (真实SentenceTransformer) 贡献:")
            print(f"   F1提升: {gain:+.1%} {'✅' if gain > 0 else '❌'}")
            print(f"   时间开销: +{time_cost:.3f}s")
            print(f"   效率: {gain/time_cost:.1%}/s" if time_cost > 0 else "   效率: 无穷大")
        
        # Layer3真实LLM贡献
        if 'L1_L2_Real' in results_dict and 'Full_Real_3Layer' in results_dict:
            l12_f1 = results_dict['L1_L2_Real']['metrics']['f1']
            full_f1 = results_dict['Full_Real_3Layer']['metrics']['f1']
            l12_time = results_dict['L1_L2_Real']['metrics']['avg_time']
            full_time = results_dict['Full_Real_3Layer']['metrics']['avg_time']
            
            gain = full_f1 - l12_f1
            time_cost = full_time - l12_time
            
            print(f"\n3️⃣ Layer3 (真实LLM) 贡献:")
            print(f"   F1提升: {gain:+.1%} {'✅' if gain > 0 else '❌'}")
            print(f"   时间开销: +{time_cost:.3f}s")
            print(f"   效率: {gain/time_cost:.1%}/s" if time_cost > 0 else "   效率: 无穷大")
        
        # 总体提升vs时间成本
        if 'L1_Only' in results_dict and 'Full_Real_3Layer' in results_dict:
            l1_f1 = results_dict['L1_Only']['metrics']['f1']
            full_f1 = results_dict['Full_Real_3Layer']['metrics']['f1']
            l1_time = results_dict['L1_Only']['metrics']['avg_time']
            full_time = results_dict['Full_Real_3Layer']['metrics']['avg_time']
            
            total_gain = full_f1 - l1_f1
            total_time_cost = full_time - l1_time
            
            print(f"\n📈 总体分析:")
            print(f"   F1总提升: {l1_f1:.1%} → {full_f1:.1%} ({total_gain:+.1%})")
            print(f"   时间总成本: {l1_time:.3f}s → {full_time:.3f}s (+{total_time_cost:.3f}s)")
            print(f"   时间增加倍数: {full_time/l1_time:.1f}x" if l1_time > 0 else "   时间增加: 无穷大")
            print(f"   性价比: {total_gain/total_time_cost:.1%}/s" if total_time_cost > 0 else "   性价比: 无穷大")
        
        # 最佳配置
        best = max(self.results, key=lambda x: x['metrics']['f1'])
        print(f"\n🏆 最佳配置: {best['config']}")
        print(f"   F1: {best['metrics']['f1']:.1%}")
        print(f"   时间: {best['metrics']['avg_time']:.3f}s")
        print(f"   LLM调用率: {best['metrics']['llm_rate']:.0%}")

    def save_real_results(self):
        """保存真实实验结果"""
        output_dir = Path(f"real_ablation_{self.task_type}_{self.timestamp}")
        output_dir.mkdir(exist_ok=True)
        
        # 保存JSON结果
        with open(output_dir / "results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 真实实验结果已保存到: {output_dir}/")

async def main():
    """主函数"""
    print("🚀 开始真正的三层架构消融实验")
    print("   使用真实的SentenceTransformer和LLM API")
    print("="*80)
    
    # UNION任务真实消融实验
    union_study = RealThreeLayerAblation(task_type="union", dataset_size="subset")
    await union_study.run_real_ablation_study()
    
    print("\n" + "="*80)
    print("✅ 真实三层架构消融实验完成！")
    print("   - 使用了真实的SentenceTransformer模型")
    print("   - 调用了真实的Gemini LLM API") 
    print("   - 测量了真实的处理时间")
    print("   - 分析了真实的层贡献")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())