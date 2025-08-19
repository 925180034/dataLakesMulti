"""
LangGraph Workflow State Definition
"""
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class QueryTask:
    """Query task definition"""
    query: str
    task_type: str  # 'join' or 'union'
    table_name: str = ""
    query_id: Optional[str] = None
    ground_truth: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class TableInfo:
    """Table information"""
    table_name: str
    columns: List[Dict[str, Any]]
    row_count: Optional[int] = None
    description: Optional[str] = None


@dataclass
class OptimizationConfig:
    """Optimization configuration from OptimizerAgent"""
    parallel_workers: int = 8
    llm_concurrency: int = 20
    cache_level: str = "L2"  # L1, L2, or L3
    use_vector_search: bool = True
    use_llm_verification: bool = True
    batch_size: int = 20
    
    # L3层LLM匹配参数（新增）
    llm_confidence_threshold: float = 0.5
    aggregator_min_score: float = 0.2
    aggregator_max_results: int = 50
    vector_top_k: int = 150


@dataclass
class ExecutionStrategy:
    """Execution strategy from PlannerAgent"""
    name: str  # 'bottom-up', 'top-down', 'hybrid'
    use_metadata: bool = True
    use_vector: bool = True
    use_llm: bool = True
    top_k: int = 50
    confidence_threshold: float = 0.5


@dataclass
class TableAnalysis:
    """Table analysis from AnalyzerAgent"""
    column_count: int
    column_types: Dict[str, int]
    key_columns: List[str]
    table_type: str  # 'dimension', 'fact', 'unknown'
    column_names: List[str]
    patterns: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateTable:
    """Candidate table with score"""
    table_name: str
    score: float
    source: str  # 'metadata', 'vector', 'llm'
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchResult:
    """Match result from MatcherAgent"""
    query_table: str
    matched_table: str
    score: float
    match_type: str  # 'join' or 'union'
    confidence: float
    agent_used: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    total_time: float = 0.0
    agent_times: Dict[str, float] = field(default_factory=dict)
    candidates_generated: int = 0
    llm_calls_made: int = 0
    cache_hits: int = 0
    errors_count: int = 0


class WorkflowState(TypedDict):
    """
    Complete workflow state for LangGraph
    This state is passed between agents
    """
    # Input
    query_task: QueryTask
    query_table: Dict[str, Any]  # Raw table data
    all_tables: List[Dict[str, Any]]  # All available tables
    
    # Agent outputs
    optimization_config: Optional[OptimizationConfig]
    strategy: Optional[ExecutionStrategy]
    analysis: Optional[TableAnalysis]
    candidates: List[CandidateTable]
    matches: List[MatchResult]
    final_results: List[MatchResult]
    
    # Metadata
    execution_path: List[str]  # Track which agents were executed
    errors: List[str]  # Collect errors
    metrics: PerformanceMetrics  # Performance tracking
    
    # Control flow
    should_use_llm: bool  # Conditional flag for LLM usage
    skip_matcher: bool  # Skip matcher if high confidence candidates