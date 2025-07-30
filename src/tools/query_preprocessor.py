"""
查询预处理器 - 智能预筛选和查询优化
"""

from typing import List, Dict, Any, Set, Optional, Tuple
import re
import logging
from collections import defaultdict

from src.core.models import ColumnInfo, TableInfo, AgentState

logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """查询预处理器 - 提取查询意图和优化搜索策略"""
    
    def __init__(self):
        # 领域关键词映射
        self.domain_keywords = {
            'customer': {'customer', 'client', 'user', 'account', 'buyer', 'consumer'},
            'order': {'order', 'purchase', 'transaction', 'sale', 'invoice', 'receipt'},
            'product': {'product', 'item', 'goods', 'merchandise', 'inventory', 'catalog'},
            'employee': {'employee', 'staff', 'worker', 'personnel', 'team', 'member'},
            'financial': {'payment', 'price', 'cost', 'revenue', 'profit', 'expense', 'budget'},
            'temporal': {'date', 'time', 'timestamp', 'created', 'updated', 'modified'},
            'location': {'address', 'city', 'country', 'region', 'location', 'place'},
            'contact': {'email', 'phone', 'contact', 'communication', 'message'}
        }
        
        # 操作意图关键词
        self.intent_keywords = {
            'join': {'join', 'joinable', 'link', 'connect', 'relate', 'match', 'merge'},
            'union': {'union', 'combine', 'similar', 'same', 'equivalent', 'identical'},
            'analysis': {'analyze', 'analysis', 'insight', 'pattern', 'trend', 'statistics'},
            'search': {'find', 'search', 'discover', 'locate', 'identify', 'look'}
        }
        
        # 数据类型关键词
        self.type_keywords = {
            'numeric': {'number', 'numeric', 'integer', 'decimal', 'float', 'count', 'amount'},
            'text': {'text', 'string', 'name', 'title', 'description', 'label'},
            'temporal': {'date', 'time', 'timestamp', 'datetime', 'year', 'month'},
            'identifier': {'id', 'key', 'identifier', 'reference', 'code', 'index'}
        }
        
        logger.info("初始化查询预处理器")
    
    def analyze_query(self, query: str, query_tables: List[TableInfo] = None) -> Dict[str, Any]:
        """分析查询意图和提取关键信息"""
        try:
            analysis = {
                'original_query': query,
                'extracted_keywords': set(),
                'domain_hints': set(),
                'operation_intent': None,
                'data_type_hints': set(),
                'table_schema_patterns': [],
                'complexity_score': 0.0,
                'optimization_suggestions': []
            }
            
            # 文本预处理
            query_lower = query.lower()
            query_words = self._extract_words(query_lower)
            analysis['extracted_keywords'] = query_words
            
            # 领域分析
            analysis['domain_hints'] = self._detect_domains(query_words)
            
            # 操作意图分析
            analysis['operation_intent'] = self._detect_operation_intent(query_words)
            
            # 数据类型提示
            analysis['data_type_hints'] = self._detect_data_types(query_words)
            
            # 查询表结构分析
            if query_tables:
                analysis['table_schema_patterns'] = self._analyze_table_schemas(query_tables)
                analysis['query_table_signatures'] = self._generate_table_signatures(query_tables)
            
            # 复杂度评估
            analysis['complexity_score'] = self._calculate_complexity(analysis, query_tables)
            
            # 优化建议
            analysis['optimization_suggestions'] = self._generate_optimization_suggestions(analysis)
            
            logger.debug(f"查询分析完成: {analysis['operation_intent']}, "
                        f"领域: {analysis['domain_hints']}, "
                        f"复杂度: {analysis['complexity_score']:.2f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            return self._get_default_analysis(query)
    
    def _extract_words(self, text: str) -> Set[str]:
        """提取关键词"""
        # 使用正则表达式提取单词，处理驼峰命名和下划线
        words = re.findall(r'[a-zA-Z]+', text)
        
        # 进一步分割驼峰命名
        expanded_words = []
        for word in words:
            # 分割驼峰命名
            camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', word)
            expanded_words.extend([part.lower() for part in camel_parts if len(part) > 1])
        
        # 过滤停用词
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        return {word for word in expanded_words if word not in stop_words and len(word) > 2}
    
    def _detect_domains(self, query_words: Set[str]) -> Set[str]:
        """检测查询涉及的领域"""
        detected_domains = set()
        
        for domain, keywords in self.domain_keywords.items():
            if query_words & keywords:  # 有交集
                detected_domains.add(domain)
        
        return detected_domains
    
    def _detect_operation_intent(self, query_words: Set[str]) -> Optional[str]:
        """检测操作意图"""
        intent_scores = defaultdict(int)
        
        for intent, keywords in self.intent_keywords.items():
            matches = query_words & keywords
            intent_scores[intent] = len(matches)
        
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return 'search'  # 默认为搜索意图
    
    def _detect_data_types(self, query_words: Set[str]) -> Set[str]:
        """检测数据类型提示"""
        detected_types = set()
        
        for data_type, keywords in self.type_keywords.items():
            if query_words & keywords:
                detected_types.add(data_type)
        
        return detected_types
    
    def _analyze_table_schemas(self, tables: List[TableInfo]) -> List[Dict[str, Any]]:
        """分析查询表的结构模式"""
        schema_patterns = []
        
        for table in tables:
            pattern = {
                'table_name': table.table_name,
                'column_count': len(table.columns),
                'column_types': self._get_column_type_distribution(table.columns),
                'identifier_columns': self._find_identifier_columns(table.columns),
                'semantic_groups': self._group_columns_by_semantics(table.columns)
            }
            schema_patterns.append(pattern)
        
        return schema_patterns
    
    def _get_column_type_distribution(self, columns: List[ColumnInfo]) -> Dict[str, int]:
        """获取列类型分布"""
        type_dist = defaultdict(int)
        
        for col in columns:
            if col.data_type:
                normalized_type = self._normalize_column_type(col.data_type)
                type_dist[normalized_type] += 1
            else:
                # 基于列名和样本值推断类型
                inferred_type = self._infer_column_type(col)
                type_dist[inferred_type] += 1
        
        return dict(type_dist)
    
    def _normalize_column_type(self, data_type: str) -> str:
        """标准化列类型"""
        data_type = data_type.lower()
        
        if any(t in data_type for t in ['int', 'bigint', 'smallint', 'serial']):
            return 'integer'
        elif any(t in data_type for t in ['float', 'double', 'decimal', 'numeric', 'real']):
            return 'numeric'
        elif any(t in data_type for t in ['varchar', 'char', 'text', 'string']):
            return 'text'
        elif any(t in data_type for t in ['date', 'timestamp', 'datetime', 'time']):
            return 'temporal'
        elif any(t in data_type for t in ['bool', 'boolean']):
            return 'boolean'
        else:
            return 'other'
    
    def _infer_column_type(self, column: ColumnInfo) -> str:
        """基于列名和样本值推断类型"""
        col_name = column.column_name.lower()
        
        # 基于列名推断
        if 'id' in col_name or col_name.endswith('_id') or col_name.startswith('id_'):
            return 'identifier'
        elif any(kw in col_name for kw in ['date', 'time', 'created', 'updated']):
            return 'temporal'
        elif any(kw in col_name for kw in ['name', 'title', 'description', 'text']):
            return 'text'
        elif any(kw in col_name for kw in ['price', 'cost', 'amount', 'count', 'number']):
            return 'numeric'
        
        # 基于样本值推断
        if column.sample_values:
            numeric_count = 0
            for value in column.sample_values[:5]:
                try:
                    float(str(value))
                    numeric_count += 1
                except (ValueError, TypeError):
                    pass
            
            if numeric_count >= len(column.sample_values) * 0.8:
                return 'numeric'
        
        return 'text'  # 默认为文本
    
    def _find_identifier_columns(self, columns: List[ColumnInfo]) -> List[str]:
        """找到标识符列"""
        identifiers = []
        
        for col in columns:
            col_name = col.column_name.lower()
            if ('id' in col_name or 
                col_name.endswith('_id') or 
                col_name.startswith('id_') or
                'key' in col_name or
                'ref' in col_name):
                identifiers.append(col.column_name)
        
        return identifiers
    
    def _group_columns_by_semantics(self, columns: List[ColumnInfo]) -> Dict[str, List[str]]:
        """按语义对列进行分组"""
        semantic_groups = defaultdict(list)
        
        for col in columns:
            col_name = col.column_name.lower()
            
            # 语义分组
            if any(kw in col_name for kw in ['name', 'title', 'label']):
                semantic_groups['naming'].append(col.column_name)
            elif any(kw in col_name for kw in ['date', 'time', 'created', 'updated']):
                semantic_groups['temporal'].append(col.column_name)
            elif any(kw in col_name for kw in ['email', 'phone', 'address', 'contact']):
                semantic_groups['contact'].append(col.column_name)
            elif any(kw in col_name for kw in ['price', 'cost', 'amount', 'value']):
                semantic_groups['financial'].append(col.column_name)
            elif 'id' in col_name or 'key' in col_name:
                semantic_groups['identifier'].append(col.column_name)
            else:
                semantic_groups['general'].append(col.column_name)
        
        return dict(semantic_groups)
    
    def _generate_table_signatures(self, tables: List[TableInfo]) -> List[str]:
        """生成表签名用于快速匹配"""
        signatures = []
        
        for table in tables:
            # 提取关键特征
            col_count = len(table.columns)
            col_types = sorted(self._get_column_type_distribution(table.columns).keys())
            identifiers = self._find_identifier_columns(table.columns)
            
            # 生成签名
            signature = f"cols:{col_count}|types:{','.join(col_types)}|ids:{len(identifiers)}"
            signatures.append(signature)
        
        return signatures
    
    def _calculate_complexity(self, analysis: Dict[str, Any], query_tables: List[TableInfo] = None) -> float:
        """计算查询复杂度"""
        complexity = 0.0
        
        # 基于查询词汇复杂度
        keyword_count = len(analysis['extracted_keywords'])
        complexity += min(keyword_count * 0.05, 0.3)
        
        # 基于领域数量
        domain_count = len(analysis['domain_hints'])
        complexity += domain_count * 0.1
        
        # 基于表数量和结构复杂度
        if query_tables:
            table_count = len(query_tables)
            complexity += min(table_count * 0.1, 0.4)
            
            # 表结构复杂度
            total_columns = sum(len(table.columns) for table in query_tables)
            complexity += min(total_columns * 0.01, 0.3)
        
        return min(complexity, 1.0)
    
    def _generate_optimization_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        # 基于复杂度的建议
        if analysis['complexity_score'] > 0.7:
            suggestions.append("high_complexity_query")
            suggestions.append("enable_hierarchical_search")
            suggestions.append("use_prefiltering")
        
        # 基于操作意图的建议
        if analysis['operation_intent'] == 'join':
            suggestions.append("focus_on_identifier_columns")
            suggestions.append("prioritize_foreign_key_relationships")
        elif analysis['operation_intent'] == 'union':
            suggestions.append("focus_on_schema_similarity")
            suggestions.append("prioritize_column_type_matching")
        
        # 基于领域提示的建议
        if len(analysis['domain_hints']) > 1:
            suggestions.append("multi_domain_query")
            suggestions.append("use_domain_specific_filtering")
        
        return suggestions
    
    def _get_default_analysis(self, query: str) -> Dict[str, Any]:
        """获取默认分析结果"""
        return {
            'original_query': query,
            'extracted_keywords': set(),
            'domain_hints': set(),
            'operation_intent': 'search',
            'data_type_hints': set(),
            'table_schema_patterns': [],
            'complexity_score': 0.5,
            'optimization_suggestions': ['standard_search']
        }


class SmartPrefilter:
    """智能预筛选器"""
    
    def __init__(self):
        self.query_processor = QueryPreprocessor()
        logger.info("初始化智能预筛选器")
    
    def generate_search_strategy(self, 
                                query: str, 
                                query_tables: List[TableInfo] = None) -> Dict[str, Any]:
        """生成搜索策略"""
        try:
            # 分析查询
            analysis = self.query_processor.analyze_query(query, query_tables)
            
            # 生成搜索策略
            strategy = {
                'use_hierarchical_search': analysis['complexity_score'] > 0.5,
                'prefilter_settings': self._generate_prefilter_settings(analysis),
                'search_parameters': self._generate_search_parameters(analysis),
                'result_optimization': self._generate_result_optimization(analysis)
            }
            
            logger.debug(f"生成搜索策略: 分层搜索={strategy['use_hierarchical_search']}")
            return strategy
            
        except Exception as e:
            logger.error(f"生成搜索策略失败: {e}")
            return self._get_default_strategy()
    
    def _generate_prefilter_settings(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成预筛选设置"""
        settings = {
            'use_domain_filtering': len(analysis['domain_hints']) > 0,
            'domain_keywords': analysis['domain_hints'],
            'use_type_filtering': len(analysis['data_type_hints']) > 0,
            'preferred_types': analysis['data_type_hints'],
            'use_schema_filtering': len(analysis['table_schema_patterns']) > 0,
            'max_candidates': self._calculate_max_candidates(analysis['complexity_score'])
        }
        
        return settings
    
    def _generate_search_parameters(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成搜索参数"""
        params = {
            'similarity_threshold': self._calculate_similarity_threshold(analysis),
            'top_k_tables': self._calculate_top_k_tables(analysis),
            'top_k_columns': self._calculate_top_k_columns(analysis),
            'enable_cross_table_search': analysis['operation_intent'] in ['join', 'analysis']
        }
        
        return params
    
    def _generate_result_optimization(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成结果优化设置"""
        optimization = {
            'enable_clustering': analysis['complexity_score'] > 0.6,
            'diversity_boost': analysis['operation_intent'] == 'analysis',
            'domain_boost': len(analysis['domain_hints']) > 0,
            'recency_boost': 'temporal' in analysis['data_type_hints']
        }
        
        return optimization
    
    def _calculate_max_candidates(self, complexity_score: float) -> int:
        """根据复杂度计算最大候选数"""
        if complexity_score < 0.3:
            return 200
        elif complexity_score < 0.6:
            return 500
        else:
            return 1000
    
    def _calculate_similarity_threshold(self, analysis: Dict[str, Any]) -> float:
        """计算相似度阈值"""
        base_threshold = 0.7
        
        # 复杂查询降低阈值
        if analysis['complexity_score'] > 0.7:
            base_threshold -= 0.1
        
        # 多领域查询降低阈值
        if len(analysis['domain_hints']) > 2:
            base_threshold -= 0.05
        
        return max(base_threshold, 0.5)
    
    def _calculate_top_k_tables(self, analysis: Dict[str, Any]) -> int:
        """计算表级搜索的top-k"""
        base_k = 50
        
        if analysis['complexity_score'] > 0.7:
            base_k = 100
        elif analysis['complexity_score'] < 0.3:
            base_k = 20
        
        return base_k
    
    def _calculate_top_k_columns(self, analysis: Dict[str, Any]) -> int:
        """计算列级搜索的top-k"""
        base_k = 50
        
        if analysis['operation_intent'] == 'join':
            base_k = 30  # join操作通常需要较少但精确的结果
        elif analysis['operation_intent'] == 'analysis':
            base_k = 100  # 分析操作需要更多候选
        
        return base_k
    
    def _get_default_strategy(self) -> Dict[str, Any]:
        """获取默认搜索策略"""
        return {
            'use_hierarchical_search': True,
            'prefilter_settings': {
                'use_domain_filtering': False,
                'domain_keywords': set(),
                'use_type_filtering': False,
                'preferred_types': set(),
                'use_schema_filtering': False,
                'max_candidates': 500
            },
            'search_parameters': {
                'similarity_threshold': 0.7,
                'top_k_tables': 50,
                'top_k_columns': 50,
                'enable_cross_table_search': True
            },
            'result_optimization': {
                'enable_clustering': True,
                'diversity_boost': False,
                'domain_boost': False,
                'recency_boost': False
            }
        }


# 创建全局实例
query_preprocessor = QueryPreprocessor()
smart_prefilter = SmartPrefilter()