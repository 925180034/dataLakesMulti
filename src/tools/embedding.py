from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
from src.core.models import ColumnInfo, TableInfo
from src.config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator(ABC):
    """嵌入向量生成器抽象基类"""
    
    @abstractmethod
    async def generate_text_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        pass
    
    @abstractmethod
    async def generate_column_embedding(self, column_info: ColumnInfo) -> List[float]:
        """生成列的嵌入向量"""
        pass
    
    @abstractmethod
    async def generate_table_embedding(self, table_info: TableInfo) -> List[float]:
        """生成表的嵌入向量"""
        pass


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """基于OpenAI API的嵌入向量生成器"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.model_name = model_name
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化OpenAI客户端"""
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=settings.llm.api_key
            )
            logger.info(f"初始化OpenAI嵌入客户端完成，模型: {self.model_name}")
        except ImportError:
            logger.error("OpenAI库未安装，请安装: pip install openai")
            raise
        except Exception as e:
            logger.error(f"初始化OpenAI客户端失败: {e}")
            raise
    
    async def generate_text_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        try:
            if not text or not text.strip():
                logger.warning("输入文本为空")
                return [0.0] * 1536  # 返回零向量
            
            # 清理和截断文本
            text = text.strip()[:8000]  # OpenAI限制约8000字符
            
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"生成文本嵌入向量，维度: {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"生成文本嵌入失败: {e}")
            # 返回零向量作为后备
            return [0.0] * 1536
    
    async def generate_column_embedding(self, column_info: ColumnInfo) -> List[float]:
        """生成列的嵌入向量"""
        try:
            # 构建列的文本描述
            text_parts = [
                f"列名: {column_info.column_name}",
                f"表名: {column_info.table_name}"
            ]
            
            if column_info.data_type:
                text_parts.append(f"数据类型: {column_info.data_type}")
            
            if column_info.sample_values:
                # 只使用前5个样本值
                sample_values = [str(v) for v in column_info.sample_values[:5] if v is not None]
                if sample_values:
                    text_parts.append(f"样本值: {', '.join(sample_values)}")
            
            # 添加统计信息
            if column_info.unique_count is not None:
                text_parts.append(f"唯一值数量: {column_info.unique_count}")
            
            if column_info.null_count is not None:
                text_parts.append(f"空值数量: {column_info.null_count}")
            
            column_text = "\n".join(text_parts)
            
            embedding = await self.generate_text_embedding(column_text)
            logger.debug(f"生成列嵌入向量: {column_info.full_name}")
            return embedding
            
        except Exception as e:
            logger.error(f"生成列嵌入向量失败: {e}")
            return [0.0] * 1536
    
    async def generate_table_embedding(self, table_info: TableInfo) -> List[float]:
        """生成表的嵌入向量"""
        try:
            # 构建表的文本描述
            text_parts = [
                f"表名: {table_info.table_name}"
            ]
            
            if table_info.description:
                text_parts.append(f"描述: {table_info.description}")
            
            if table_info.row_count is not None:
                text_parts.append(f"行数: {table_info.row_count}")
            
            # 添加列信息
            if table_info.columns:
                column_names = [col.column_name for col in table_info.columns]
                text_parts.append(f"列名: {', '.join(column_names)}")
                
                # 添加数据类型信息
                data_types = [col.data_type for col in table_info.columns if col.data_type]
                if data_types:
                    text_parts.append(f"数据类型: {', '.join(set(data_types))}")
                
                # 添加一些样本值
                all_samples = []
                for col in table_info.columns[:3]:  # 只使用前3列的样本
                    if col.sample_values:
                        samples = [str(v) for v in col.sample_values[:2] if v is not None]
                        all_samples.extend(samples)
                
                if all_samples:
                    text_parts.append(f"样本数据: {', '.join(all_samples[:10])}")  # 限制样本数量
            
            table_text = "\n".join(text_parts)
            
            embedding = await self.generate_text_embedding(table_text)
            logger.debug(f"生成表嵌入向量: {table_info.table_name}")
            return embedding
            
        except Exception as e:
            logger.error(f"生成表嵌入向量失败: {e}")
            return [0.0] * 1536


class SentenceTransformerEmbeddingGenerator(EmbeddingGenerator):
    """基于SentenceTransformers的本地嵌入向量生成器"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._model_initialized = False
    
    def _initialize_model(self):
        """延迟初始化SentenceTransformer模型"""
        if self._model_initialized:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"正在初始化SentenceTransformer模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self._model_initialized = True
            logger.info(f"初始化SentenceTransformer模型完成: {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers库未安装，请安装: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"初始化SentenceTransformer模型失败: {e}")
            logger.warning("将使用虚拟嵌入向量作为后备方案")
            self.model = None
            self._model_initialized = True  # 标记为已尝试初始化
    
    def generate_text_embedding_sync(self, text: str) -> List[float]:
        """同步生成文本嵌入向量"""
        # 延迟初始化模型
        if not self._model_initialized:
            self._initialize_model()
        
        try:
            if not text or not text.strip():
                logger.warning("输入文本为空")
                # 返回默认维度的零向量
                return [0.0] * 384  # all-MiniLM-L6-v2的维度
            
            # 如果模型初始化失败，返回虚拟向量
            if self.model is None:
                logger.warning(f"模型未初始化，返回虚拟嵌入向量")
                return self._generate_dummy_embedding(text)
            
            text = text.strip()
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            logger.debug(f"生成文本嵌入向量，维度: {len(embedding)}")
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"生成文本嵌入失败: {e}")
            return self._generate_dummy_embedding(text)
    
    async def generate_text_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        # 延迟初始化模型
        if not self._model_initialized:
            self._initialize_model()
        
        try:
            if not text or not text.strip():
                logger.warning("输入文本为空")
                # 返回默认维度的零向量
                return [0.0] * 384  # all-MiniLM-L6-v2的维度
            
            # 如果模型初始化失败，返回虚拟向量
            if self.model is None:
                logger.warning(f"模型未初始化，返回虚拟嵌入向量")
                return self._generate_dummy_embedding(text)
            
            text = text.strip()
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            logger.debug(f"生成文本嵌入向量，维度: {len(embedding)}")
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"生成文本嵌入失败: {e}")
            return self._generate_dummy_embedding(text)
    
    def _generate_dummy_embedding(self, text: str) -> List[float]:
        """生成虚拟嵌入向量（用于离线模式）"""
        import hashlib
        # 基于文本哈希生成一致的虚拟向量
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # 将哈希转换为384维向量
        dummy_vector = []
        for i in range(0, 96):  # 384/4 = 96
            hex_chunk = text_hash[i % 32:(i % 32) + 1]
            dummy_vector.extend([
                int(hex_chunk, 16) / 15.0 - 0.5,  # 归一化到[-0.5, 0.5]
                (int(hex_chunk, 16) * 7) % 16 / 15.0 - 0.5,
                (int(hex_chunk, 16) * 13) % 16 / 15.0 - 0.5,
                (int(hex_chunk, 16) * 17) % 16 / 15.0 - 0.5
            ])
        return dummy_vector[:384]
    
    async def generate_column_embedding(self, column_info: ColumnInfo) -> List[float]:
        """生成列的嵌入向量"""
        # 重用OpenAI的逻辑来构建文本描述
        openai_generator = OpenAIEmbeddingGenerator()
        try:
            # 构建列的文本描述（复用OpenAI的逻辑）
            text_parts = [
                f"列名: {column_info.column_name}",
                f"表名: {column_info.table_name}"
            ]
            
            if column_info.data_type:
                text_parts.append(f"数据类型: {column_info.data_type}")
            
            if column_info.sample_values:
                sample_values = [str(v) for v in column_info.sample_values[:5] if v is not None]
                if sample_values:
                    text_parts.append(f"样本值: {', '.join(sample_values)}")
            
            column_text = "\n".join(text_parts)
            return await self.generate_text_embedding(column_text)
            
        except Exception as e:
            logger.error(f"生成列嵌入向量失败: {e}")
            return [0.0] * 384
    
    async def generate_table_embedding(self, table_info: TableInfo) -> List[float]:
        """生成表的嵌入向量"""
        try:
            # 构建表的文本描述（复用OpenAI的逻辑）
            text_parts = [f"表名: {table_info.table_name}"]
            
            if table_info.description:
                text_parts.append(f"描述: {table_info.description}")
            
            if table_info.columns:
                column_names = [col.column_name for col in table_info.columns]
                text_parts.append(f"列名: {', '.join(column_names)}")
            
            table_text = "\n".join(text_parts)
            return await self.generate_text_embedding(table_text)
            
        except Exception as e:
            logger.error(f"生成表嵌入向量失败: {e}")
            return [0.0] * 384


def create_embedding_generator() -> EmbeddingGenerator:
    """创建嵌入向量生成器实例"""
    if settings.llm.provider == "openai" and settings.llm.api_key:
        return OpenAIEmbeddingGenerator()
    else:
        # 使用本地模型作为后备
        return SentenceTransformerEmbeddingGenerator()


# 全局嵌入向量生成器实例 - 延迟初始化
embedding_generator = None

def get_embedding_generator() -> EmbeddingGenerator:
    """获取嵌入向量生成器实例（延迟初始化）"""
    global embedding_generator
    if embedding_generator is None:
        embedding_generator = create_embedding_generator()
    return embedding_generator