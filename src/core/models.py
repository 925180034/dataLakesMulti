from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class TaskStrategy(str, Enum):
    """任务策略枚举"""
    BOTTOM_UP = "BOTTOM_UP"  # 自下而上：先匹配列，再聚合表
    TOP_DOWN = "TOP_DOWN"    # 自上而下：先发现表，再匹配列


class ColumnInfo(BaseModel):
    """列信息模型"""
    table_name: str
    column_name: str
    data_type: Optional[str] = None
    sample_values: List[Any] = Field(default_factory=list)
    null_count: Optional[int] = None
    unique_count: Optional[int] = None
    
    @property
    def full_name(self) -> str:
        return f"{self.table_name}.{self.column_name}"


class TableInfo(BaseModel):
    """表信息模型"""
    table_name: str
    columns: List[ColumnInfo]
    row_count: Optional[int] = None
    file_path: Optional[str] = None
    description: Optional[str] = None
    
    def get_column(self, column_name: str) -> Optional[ColumnInfo]:
        """获取指定列信息"""
        for col in self.columns:
            if col.column_name == column_name:
                return col
        return None


class MatchResult(BaseModel):
    """匹配结果模型"""
    source_column: str
    target_column: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    match_type: str = "semantic"  # semantic, syntactic, value_overlap


class TableMatchResult(BaseModel):
    """表匹配结果模型"""
    source_table: str
    target_table: str
    score: float = Field(ge=0.0, le=100.0)
    matched_columns: List[MatchResult] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def match_count(self) -> int:
        return len(self.matched_columns)


class AgentState(BaseModel):
    """智能体状态模型"""
    # 输入数据
    user_query: str = ""
    query_tables: List[TableInfo] = Field(default_factory=list)
    query_columns: List[ColumnInfo] = Field(default_factory=list)
    
    # 策略和控制
    strategy: Optional[TaskStrategy] = None
    current_step: str = "planning"
    
    # 中间结果
    column_matches: List[MatchResult] = Field(default_factory=list)
    table_candidates: List[str] = Field(default_factory=list)
    table_matches: List[TableMatchResult] = Field(default_factory=list)
    
    # 最终结果
    final_results: List[TableMatchResult] = Field(default_factory=list)
    final_report: str = ""
    
    # 元数据
    processing_log: List[str] = Field(default_factory=list)
    error_messages: List[str] = Field(default_factory=list)
    
    def add_log(self, message: str):
        """添加处理日志"""
        self.processing_log.append(message)
    
    def add_error(self, error: str):
        """添加错误信息"""
        self.error_messages.append(error)
    
    def clear_intermediates(self):
        """清除中间结果"""
        self.column_matches = []
        self.table_candidates = []
        self.table_matches = []


class SearchResult(BaseModel):
    """搜索结果模型"""
    item_id: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorSearchResult(SearchResult):
    """向量搜索结果"""
    embedding: Optional[List[float]] = None
    

class ValueSearchResult(SearchResult):
    """值搜索结果"""
    matched_values: List[str] = Field(default_factory=list)
    overlap_ratio: float = 0.0