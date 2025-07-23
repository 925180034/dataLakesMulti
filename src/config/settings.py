from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import os
import yaml
from pathlib import Path


class LLMConfig(BaseModel):
    """LLM配置"""
    provider: str = "openai"  # openai, anthropic
    model_name: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 30


class VectorDBConfig(BaseModel):
    """向量数据库配置"""
    provider: str = "faiss"  # faiss, chromadb
    dimension: int = 1536
    index_type: str = "IVFFlat"
    db_path: str = "./data/vector_db"
    
    
class IndexConfig(BaseModel):
    """倒排索引配置"""
    provider: str = "whoosh"
    index_path: str = "./data/index_db"
    

class ThresholdConfig(BaseModel):
    """阈值配置"""
    # 匹配阈值
    semantic_similarity_threshold: float = 0.7
    value_overlap_threshold: float = 0.3
    column_match_confidence_threshold: float = 0.6
    
    # 搜索参数
    max_candidates: int = 50
    top_k_results: int = 10
    
    # 表评分权重
    column_count_weight: float = 0.3
    confidence_weight: float = 0.5
    key_column_bonus: float = 0.2


class CacheConfig(BaseModel):
    """缓存配置"""
    enabled: bool = True
    cache_dir: str = "./cache"
    ttl_seconds: int = 3600  # 1小时
    

class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_rotation: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 5


class PerformanceConfig(BaseModel):
    """性能配置"""
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    batch_size: int = 100


class SecurityConfig(BaseModel):
    """安全配置"""
    enable_api_key_validation: bool = True
    rate_limit_per_minute: int = 60
    allowed_file_types: list = Field(default_factory=lambda: [".csv", ".json", ".parquet"])
    max_file_size_mb: int = 100


class Settings(BaseModel):
    """系统设置"""
    # 基础配置
    project_name: str = "DataLakes Multi-Agent System"
    version: str = "1.0.0"
    debug: bool = False
    
    # 路径配置
    data_dir: Path = Path("./data")
    cache_dir: Path = Path("./cache") 
    log_dir: Path = Path("./logs")
    
    # 组件配置
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Prompt模板配置
    prompt_templates_dir: str = "./src/config/prompts"
    
    @classmethod
    def from_yaml(cls, config_path: str = "config.yml") -> "Settings":
        """从YAML文件加载配置"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            print(f"配置文件 {config_path} 不存在，使用默认配置")
            return cls.from_env()
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 创建设置实例
            settings = cls()
            
            # 更新基础配置
            if 'project' in config_data:
                project_config = config_data['project']
                settings.project_name = project_config.get('name', settings.project_name)
                settings.version = project_config.get('version', settings.version)
                settings.debug = project_config.get('debug', settings.debug)
            
            # 更新路径配置
            if 'paths' in config_data:
                paths_config = config_data['paths']
                settings.data_dir = Path(paths_config.get('data_dir', settings.data_dir))
                settings.cache_dir = Path(paths_config.get('cache_dir', settings.cache_dir))
                settings.log_dir = Path(paths_config.get('log_dir', settings.log_dir))
                settings.prompt_templates_dir = paths_config.get('prompt_templates_dir', settings.prompt_templates_dir)
            
            # 更新LLM配置
            if 'llm' in config_data:
                llm_config = config_data['llm']
                settings.llm = LLMConfig(**llm_config)
            
            # 更新向量数据库配置
            if 'vector_db' in config_data:
                vector_db_config = config_data['vector_db']
                settings.vector_db = VectorDBConfig(**vector_db_config)
            
            # 更新索引配置
            if 'index' in config_data:
                index_config = config_data['index']
                settings.index = IndexConfig(**index_config)
            
            # 更新阈值配置
            if 'thresholds' in config_data:
                thresholds_config = config_data['thresholds']
                settings.thresholds = ThresholdConfig(**thresholds_config)
            
            # 更新缓存配置
            if 'cache' in config_data:
                cache_config = config_data['cache']
                settings.cache = CacheConfig(**cache_config)
            
            # 更新日志配置
            if 'logging' in config_data:
                logging_config = config_data['logging']
                settings.logging = LoggingConfig(**logging_config)
            
            # 更新性能配置
            if 'performance' in config_data:
                performance_config = config_data['performance']
                settings.performance = PerformanceConfig(**performance_config)
            
            # 更新安全配置
            if 'security' in config_data:
                security_config = config_data['security']
                settings.security = SecurityConfig(**security_config)
            
            # 从环境变量覆盖敏感配置
            settings._override_with_env()
            
            return settings
            
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            print("使用默认配置和环境变量")
            return cls.from_env()
    
    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量加载配置"""
        settings = cls()
        settings._override_with_env()
        return settings
    
    def _override_with_env(self):
        """使用环境变量覆盖配置"""
        # LLM配置
        if os.getenv("OPENAI_API_KEY"):
            self.llm.api_key = os.getenv("OPENAI_API_KEY")
        elif os.getenv("ANTHROPIC_API_KEY"):
            self.llm.provider = "anthropic"
            self.llm.model_name = "claude-3-sonnet-20240229"
            self.llm.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # 路径配置
        if os.getenv("VECTOR_DB_PATH"):
            self.vector_db.db_path = os.getenv("VECTOR_DB_PATH")
        if os.getenv("INDEX_DB_PATH"):
            self.index.index_path = os.getenv("INDEX_DB_PATH")
        if os.getenv("CACHE_DIR"):
            self.cache.cache_dir = os.getenv("CACHE_DIR")
        
        # 调试模式
        if os.getenv("DEBUG"):
            self.debug = os.getenv("DEBUG").lower() == "true"
        
        # 缓存配置
        if os.getenv("CACHE_ENABLED"):
            self.cache.enabled = os.getenv("CACHE_ENABLED").lower() == "true"
    
    def create_directories(self):
        """创建必要的目录"""
        directories = [
            self.data_dir,
            self.cache_dir,
            self.log_dir,
            Path(self.vector_db.db_path).parent,
            Path(self.index.index_path).parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# 全局设置实例
settings = Settings.from_yaml()