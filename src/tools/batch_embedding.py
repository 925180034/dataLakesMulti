"""
批量嵌入生成器 - 支持高效的批量向量计算
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


class BatchEmbeddingGenerator:
    """批量嵌入生成器 - 优化版本"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # all-MiniLM-L6-v2 的维度
        self._initialized = False
        
    def initialize(self):
        """初始化模型（延迟加载）"""
        if self._initialized:
            return
        
        # 检查是否强制跳过模型下载
        import os
        if os.getenv('SKIP_EMBEDDING_DOWNLOAD', '').lower() in ['true', '1', 'yes']:
            logger.info("⚠️ 检测到 SKIP_EMBEDDING_DOWNLOAD=true，跳过模型下载，使用虚拟嵌入")
            self.model = None
            self._initialized = True
            return
            
        if os.getenv('USE_DUMMY_EMBEDDINGS', '').lower() in ['true', '1', 'yes']:
            logger.info("⚠️ 检测到 USE_DUMMY_EMBEDDINGS=true，使用虚拟嵌入向量")
            self.model = None
            self._initialized = True
            return
            
        try:
            # 设置离线模式环境变量
            import os
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HOME'] = '/root/.cache/huggingface'
            os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/.cache/huggingface/hub'
            os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
            
            from sentence_transformers import SentenceTransformer
            logger.info(f"🚀 正在初始化批量嵌入模型: {self.model_name}")
            
            # 尝试从本地缓存加载
            local_model_path = f"/root/.cache/huggingface/hub/models--sentence-transformers--{self.model_name}"
            if os.path.exists(local_model_path):
                logger.info(f"   使用本地缓存模型 (离线模式)")
            else:
                logger.info("   首次运行可能需要下载模型文件 (~90MB)...")
            
            start_time = time.time()
            self.model = SentenceTransformer(self.model_name, cache_folder="/root/.cache/huggingface/hub")
            init_time = time.time() - start_time
            
            logger.info(f"✅ 批量嵌入模型初始化完成，耗时: {init_time:.2f}秒")
            self._initialized = True
            
        except ImportError:
            logger.error("❌ sentence-transformers库未安装: pip install sentence-transformers")
            logger.info("💡 自动切换到虚拟嵌入模式")
            self.model = None
            self._initialized = True
        except Exception as e:
            logger.error(f"❌ 初始化嵌入模型失败: {e}")
            logger.info("💡 自动切换到虚拟嵌入模式")
            self.model = None
            self._initialized = True
    
    def generate_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """批量生成嵌入向量 - 核心优化方法"""
        if not self._initialized:
            self.initialize()
        
        if not texts:
            return []
        
        # 如果模型不可用，使用虚拟向量
        if self.model is None:
            logger.warning(f"使用虚拟嵌入向量，文本数量: {len(texts)}")
            return [self._generate_dummy_embedding(text) for text in texts]
        
        try:
            start_time = time.time()
            
            # 批量编码，这是性能关键
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 50,
                convert_to_numpy=True,
                normalize_embeddings=True  # 归一化嵌入
            )
            
            elapsed = time.time() - start_time
            logger.info(f"批量生成 {len(texts)} 个嵌入，耗时: {elapsed:.2f}秒 "
                       f"({len(texts)/elapsed:.1f} 个/秒)")
            
            return [emb for emb in embeddings]
            
        except Exception as e:
            logger.error(f"批量嵌入生成失败: {e}")
            # 降级到虚拟嵌入
            return [self._generate_dummy_embedding(text) for text in texts]
    
    def generate_table_texts(self, tables: List[Dict]) -> List[str]:
        """为表生成文本表示"""
        texts = []
        for table in tables:
            # 提取表信息
            table_name = table.get('table_name', '')
            columns = table.get('columns', [])
            
            # 构建文本表示
            text_parts = [f"表名: {table_name}"]
            
            # 添加列信息（限制数量避免过长）
            if columns:
                column_names = []
                column_types = []
                sample_values = []
                
                for col in columns[:15]:  # 最多15列
                    col_name = col.get('column_name', col.get('name', ''))
                    col_type = col.get('data_type', col.get('type', ''))
                    samples = col.get('sample_values', [])
                    
                    if col_name:
                        column_names.append(col_name)
                    if col_type:
                        column_types.append(col_type)
                    if samples:
                        sample_values.extend([str(s) for s in samples[:2] if s])
                
                if column_names:
                    text_parts.append(f"列名: {' '.join(column_names[:10])}")
                if column_types:
                    text_parts.append(f"类型: {' '.join(list(set(column_types))[:5])}")
                if sample_values:
                    text_parts.append(f"样本: {' '.join(sample_values[:10])}")
            
            texts.append(' | '.join(text_parts))
        
        return texts
    
    def _generate_dummy_embedding(self, text: str) -> np.ndarray:
        """生成虚拟嵌入向量（确定性）"""
        import hashlib
        
        # 基于文本内容生成确定性的虚拟向量
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # 使用哈希值生成固定维度的向量
        np.random.seed(int(text_hash[:8], 16))  # 使用哈希前8位作为种子
        embedding = np.random.normal(0, 0.3, self.dimension).astype(np.float32)
        
        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def precompute_table_embeddings(self, tables: List[Dict]) -> Dict[str, np.ndarray]:
        """预计算所有表的嵌入向量"""
        logger.info("开始预计算表嵌入向量...")
        
        start_time = time.time()
        
        # 生成文本表示
        texts = self.generate_table_texts(tables)
        table_names = [t.get('table_name', f'table_{i}') for i, t in enumerate(tables)]
        
        # 批量生成嵌入
        embeddings = self.generate_batch(texts, batch_size=64)  # 更大的批处理
        
        # 创建映射
        embedding_map = {}
        for name, embedding in zip(table_names, embeddings):
            embedding_map[name] = embedding
        
        elapsed = time.time() - start_time
        logger.info(f"预计算完成: {len(tables)} 个表，耗时: {elapsed:.2f}秒")
        
        return embedding_map


class OptimizedLLMBatcher:
    """优化的LLM批量调用器"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.response_cache = {}
        
    async def batch_verify_with_smartskip(self,
                                        query_table: Dict,
                                        candidates: List[Dict],
                                        existing_scores: List[float],
                                        batch_size: int = 10,
                                        confidence_threshold: float = 0.95) -> List[Dict]:
        """智能批量验证 - 跳过高置信度候选"""
        
        # 分离高置信度和低置信度候选
        high_confidence = []
        need_llm_verification = []
        
        for i, (candidate, score) in enumerate(zip(candidates, existing_scores)):
            if score > confidence_threshold:
                # 高置信度直接通过，无需LLM验证
                high_confidence.append({
                    'index': i,
                    'candidate': candidate,
                    'result': {
                        'is_match': True,
                        'confidence': score,
                        'reason': f'High confidence score: {score:.3f}, skipped LLM verification'
                    }
                })
            else:
                need_llm_verification.append({
                    'index': i,
                    'candidate': candidate,
                    'score': score
                })
        
        logger.info(f"智能跳过: {len(high_confidence)} 个高置信度候选，"
                   f"需要LLM验证: {len(need_llm_verification)} 个")
        
        # 批量验证低置信度候选
        llm_results = []
        if need_llm_verification:
            llm_results = await self._batch_llm_verify(
                query_table,
                [item['candidate'] for item in need_llm_verification],
                [item['score'] for item in need_llm_verification],
                batch_size
            )
        
        # 合并结果
        final_results = [None] * len(candidates)
        
        # 填入高置信度结果
        for item in high_confidence:
            final_results[item['index']] = item['result']
        
        # 填入LLM验证结果
        for i, llm_result in enumerate(llm_results):
            original_index = need_llm_verification[i]['index']
            final_results[original_index] = llm_result
        
        return final_results
    
    async def _batch_llm_verify(self,
                              query_table: Dict,
                              candidates: List[Dict],
                              existing_scores: List[float],
                              batch_size: int) -> List[Dict]:
        """真正的批量LLM验证"""
        
        # 创建批量请求
        batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
        all_results = []
        
        for batch_idx, batch in enumerate(batches):
            batch_start = time.time()
            
            # 并发调用LLM
            tasks = []
            for candidate in batch:
                task = self._single_llm_verify(query_table, candidate)
                tasks.append(task)
            
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理异常结果
                processed_results = []
                for result in batch_results:
                    if isinstance(result, Exception):
                        processed_results.append({
                            'is_match': False,
                            'confidence': 0.0,
                            'reason': f'LLM verification failed: {str(result)}'
                        })
                    else:
                        processed_results.append(result)
                
                all_results.extend(processed_results)
                
                batch_time = time.time() - batch_start
                logger.info(f"LLM批次 {batch_idx + 1}/{len(batches)} 完成，"
                           f"耗时: {batch_time:.2f}秒，"
                           f"速度: {len(batch)/batch_time:.1f} 个/秒")
                
            except Exception as e:
                logger.error(f"批量LLM验证失败: {e}")
                # 返回失败结果
                all_results.extend([{
                    'is_match': False,
                    'confidence': 0.0,
                    'reason': f'Batch verification failed: {str(e)}'
                }] * len(batch))
        
        return all_results
    
    async def _single_llm_verify(self, query_table: Dict, candidate: Dict) -> Dict:
        """单个LLM验证（带缓存）"""
        # 生成缓存键
        cache_key = f"{query_table.get('table_name')}_{candidate.get('table_name')}"
        
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # 构建简化的prompt以减少token使用
        prompt = self._build_efficient_prompt(query_table, candidate)
        
        try:
            response = await self.llm_client.generate(prompt)
            result = self._parse_llm_response(response)
            
            # 缓存结果
            self.response_cache[cache_key] = result
            return result
            
        except Exception as e:
            result = {
                'is_match': False,
                'confidence': 0.0,
                'reason': f'LLM call failed: {str(e)}'
            }
            return result
    
    def _build_efficient_prompt(self, query_table: Dict, candidate: Dict) -> str:
        """构建高效的prompt（减少token）"""
        query_name = query_table.get('table_name', 'unknown')
        candidate_name = candidate.get('table_name', 'unknown')
        
        # 只使用前5列的信息以减少token
        query_cols = [col.get('column_name', '') 
                     for col in query_table.get('columns', [])[:5]]
        candidate_cols = [col.get('column_name', '') 
                         for col in candidate.get('columns', [])[:5]]
        
        prompt = f"""Compare these tables for JOIN compatibility:

Query: {query_name}
Columns: {', '.join(query_cols)}

Candidate: {candidate_name}
Columns: {', '.join(candidate_cols)}

Can they join? Return JSON:
{{"is_match": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict:
        """解析LLM响应"""
        try:
            import json
            
            # 查找JSON部分
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                
                # 确保有必需字段
                result.setdefault('is_match', False)
                result.setdefault('confidence', 0.0)
                result.setdefault('reason', 'No reason provided')
                
                return result
        except:
            pass
        
        # 解析失败，返回保守结果
        return {
            'is_match': False,
            'confidence': 0.0,
            'reason': 'Failed to parse LLM response'
        }