"""
SearcherAgent - Multi-layer candidate discovery
"""
import asyncio
import json
import os
from typing import List, Dict, Any
import numpy as np
from src.agents.base_agent import BaseAgent
from src.core.state import WorkflowState, CandidateTable
from src.tools.metadata_filter_tool import MetadataFilterTool
from src.tools.vector_search_tool import VectorSearchTool
from src.tools.smd_enhanced_metadata_filter import SMDEnhancedMetadataFilter
from src.config.prompts import get_agent_prompt, format_user_prompt


class SearcherAgent(BaseAgent):
    """
    Searcher Agent responsible for finding candidate tables using multiple search layers
    通过智能策略协调Layer 1和Layer 2工具
    """
    
    def __init__(self):
        super().__init__(
            name="SearcherAgent",
            description="Intelligently searches for candidate tables using Layer 1 and Layer 2 tools",
            use_llm=True  # Enable LLM for intelligent search strategy
        )
        
        # Use centralized prompt from config
        self.system_prompt = get_agent_prompt("SearcherAgent", "system_prompt")
        
        # Initialize search tools (Layer 1 and Layer 2)
        # 使用SMD增强过滤器替代原有的元数据过滤器
        self.use_enhanced_filter = os.getenv('USE_SMD_ENHANCED', 'true').lower() == 'true'
        if self.use_enhanced_filter:
            self.logger.info("Using SMD Enhanced Metadata Filter (Layer 1)")
            self.metadata_filter = SMDEnhancedMetadataFilter(max_features=1000)
        else:
            self.metadata_filter = MetadataFilterTool()
        self.vector_search = VectorSearchTool()  # Layer 2
        
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Search for candidate tables using intelligent Layer 1 and Layer 2 coordination
        通过LLM智能决策来协调Layer 1和Layer 2工具的使用
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with candidate tables
        """
        # Check if this is an NLCTables query
        query_type = state.get('query_type', 'webtables')
        
        if query_type == 'nlctables':
            return self._process_nlctables(state)
        
        self.logger.info("SearcherAgent: Starting intelligent multi-layer candidate search")
        
        # Get required inputs
        query_table = state.get('query_table')
        all_tables = state.get('all_tables', [])
        analysis = state.get('analysis')
        optimization_config = state.get('optimization_config')
        
        if not query_table:
            self.logger.error("No query table found")
            return state
        
        # Get strategy from state (set by PlannerAgent)
        strategy = state.get('strategy')
        task_type = state.get('query_task', {}).get('task_type', 'join') if state.get('query_task') else 'join'
        
        # Use LLM to determine optimal search strategy
        search_strategy = self._determine_search_strategy(
            query_table=query_table,
            task_type=task_type,
            total_tables=len(all_tables),
            analysis=analysis,
            optimization_config=optimization_config
        )
        
        # Initialize candidates tracking
        candidates = []
        candidate_scores = {}  # Track scores from different sources
        
        # Execute Layer 1: Metadata Filtering (if recommended by strategy)
        if search_strategy.get('use_metadata', True):
            self.logger.info("SearcherAgent: Executing Layer 1 - Metadata Filtering")
            
            # Get metadata filter configuration from LLM strategy
            metadata_config = search_strategy.get('metadata_config', {})
            threshold = metadata_config.get('threshold', 0.25)
            max_candidates = metadata_config.get('max_candidates', 100)
            
            metadata_candidates = self._execute_layer1(
                query_table=query_table,
                all_tables=all_tables,
                threshold=threshold,
                max_candidates=max_candidates
            )
            
            for table_name, score in metadata_candidates:
                if table_name not in candidate_scores:
                    candidate_scores[table_name] = {'metadata': 0, 'vector': 0}
                candidate_scores[table_name]['metadata'] = score
            
            self.logger.info(f"  Layer 1 returned {len(metadata_candidates)} candidates (threshold: {threshold})")
        
        # Execute Layer 2: Vector Search (if recommended by strategy)
        if search_strategy.get('use_vector', True):
            self.logger.info("SearcherAgent: Executing Layer 2 - Vector Similarity Search")
            
            # Get vector search configuration from LLM strategy
            vector_config = search_strategy.get('vector_config', {})
            similarity_threshold = vector_config.get('similarity_threshold', 0.6)
            max_candidates = vector_config.get('max_candidates', 50)
            
            vector_candidates = self._execute_layer2(
                query_table=query_table,
                all_tables=all_tables,
                similarity_threshold=similarity_threshold,
                max_candidates=max_candidates
            )
            
            for table_name, score in vector_candidates:
                if table_name not in candidate_scores:
                    candidate_scores[table_name] = {'metadata': 0, 'vector': 0}
                candidate_scores[table_name]['vector'] = score
            
            self.logger.info(f"  Layer 2 returned {len(vector_candidates)} candidates (threshold: {similarity_threshold})")
        
        # Combine and rank candidates using intelligent weighting
        self.logger.info("SearcherAgent: Combining Layer 1 and Layer 2 results")
        
        # Get weights from strategy (可以根据任务类型动态调整)
        weights = search_strategy.get('layer_weights', {})
        metadata_weight = weights.get('metadata', 0.4)
        vector_weight = weights.get('vector', 0.6)
        
        for table_name, scores in candidate_scores.items():
            # Calculate combined score using strategy weights
            combined_score = (
                scores['metadata'] * metadata_weight + 
                scores['vector'] * vector_weight
            )
            
            # Apply confidence threshold from strategy
            confidence_threshold = search_strategy.get('confidence_threshold', 0.5)
            if combined_score < confidence_threshold:
                continue  # Skip low-confidence candidates
            
            # Determine source
            if scores['metadata'] > 0 and scores['vector'] > 0:
                source = 'both'
            elif scores['metadata'] > 0:
                source = 'metadata'
            else:
                source = 'vector'
            
            # Create candidate
            candidate = CandidateTable(
                table_name=table_name,
                score=combined_score,
                source=source,
                evidence={
                    'metadata_score': scores['metadata'],
                    'vector_score': scores['vector'],
                    'layer1_weight': metadata_weight,
                    'layer2_weight': vector_weight
                }
            )
            candidates.append(candidate)
        
        # Sort by score
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Apply final candidate limit from strategy (保留top-k计算，但基于阈值筛选)
        final_limit = search_strategy.get('final_candidates', 50)
        if len(candidates) > final_limit:
            candidates = candidates[:final_limit]
        
        self.logger.info(f"SearcherAgent: Final candidate list: {len(candidates)} tables")
        if candidates:
            self.logger.info(f"  Top candidate: {candidates[0].table_name} (score: {candidates[0].score:.3f})")
            self.logger.info(f"  Score distribution: min={min(c.score for c in candidates):.3f}, max={max(c.score for c in candidates):.3f}")
        
        # Update state
        state['candidates'] = candidates
        state['search_strategy'] = search_strategy  # Store strategy for analysis
        
        # Update metrics
        if state.get('metrics'):
            state['metrics'].candidates_generated = len(candidates)
        
        return state
    
    def _determine_search_strategy(self, query_table: Dict, task_type: str, 
                                   total_tables: int, analysis: Any,
                                   optimization_config: Any) -> Dict:
        """
        Use LLM to determine optimal search strategy
        通过LLM智能决策搜索策略
        """
        # Format prompt with context
        user_prompt = format_user_prompt(
            "SearcherAgent",
            query_table=query_table.get('table_name', 'unknown'),
            task_type=task_type,
            characteristics=str(analysis) if analysis else "N/A",
            total_tables=total_tables,
            performance_target="3-5 seconds"
        )
        
        try:
            # Temporarily disable LLM strategy to avoid async issues
            # TODO: Make this method properly async
            strategy_response = None
            
            if strategy_response:
                self.logger.info(f"SearcherAgent LLM Strategy: {strategy_response.get('reasoning', 'N/A')}")
                
                # Extract strategy configuration
                search_strategy = strategy_response.get('search_strategy', {})
                metadata_filters = strategy_response.get('metadata_filters', {})
                vector_config = strategy_response.get('vector_config', {})
                
                # Prepare comprehensive strategy
                return {
                    'use_metadata': search_strategy.get('use_metadata', True),
                    'use_vector': search_strategy.get('use_vector', True),
                    'use_smart': search_strategy.get('use_smart', False),  # Layer 3 handled by MatcherAgent
                    'metadata_config': {
                        'threshold': 0.25,  # 基于阈值而不是固定数量
                        'max_candidates': 100,
                        'filters': metadata_filters
                    },
                    'vector_config': {
                        'similarity_threshold': vector_config.get('similarity_threshold', 0.6),
                        'max_candidates': vector_config.get('max_candidates', 50)
                    },
                    'layer_weights': {
                        'metadata': 0.4 if task_type == 'join' else 0.3,
                        'vector': 0.6 if task_type == 'join' else 0.7
                    },
                    'confidence_threshold': 0.5,
                    'final_candidates': 50,
                    'optimization': strategy_response.get('optimization', 'balanced'),
                    'reasoning': strategy_response.get('reasoning', '')
                }
            
        except Exception as e:
            self.logger.warning(f"LLM strategy determination failed: {e}, using defaults")
        
        # Default strategy if LLM fails
        return {
            'use_metadata': True,
            'use_vector': True,
            'use_smart': False,
            'metadata_config': {
                'threshold': 0.25,
                'max_candidates': 100
            },
            'vector_config': {
                'similarity_threshold': 0.6,
                'max_candidates': 50
            },
            'layer_weights': {
                'metadata': 0.4,
                'vector': 0.6
            },
            'confidence_threshold': 0.5,
            'final_candidates': 50,
            'optimization': 'balanced'
        }
    
    def _execute_layer1(self, query_table: Dict, all_tables: List[Dict],
                       threshold: float, max_candidates: int) -> List[tuple]:
        """
        Execute Layer 1 - Metadata Filtering
        执行第一层元数据过滤
        """
        try:
            if self.use_enhanced_filter:
                # Use SMD Enhanced filter with threshold
                candidates = self.metadata_filter.filter_candidates(
                    query_table=query_table,
                    all_tables=all_tables,
                    threshold=threshold,
                    max_candidates=max_candidates
                )
            else:
                # Use standard metadata filter
                candidates = self.metadata_filter.filter_candidates(
                    query_table=query_table,
                    all_tables=all_tables,
                    threshold=threshold,
                    max_candidates=max_candidates
                )
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Layer 1 execution failed: {e}")
            return []
    
    def _execute_layer2(self, query_table: Dict, all_tables: List[Dict],
                       similarity_threshold: float, max_candidates: int) -> List[tuple]:
        """
        Execute Layer 2 - Vector Search
        执行第二层向量搜索
        """
        try:
            # Use vector search with similarity threshold
            candidates = self.vector_search.search(
                query_table,
                all_tables,
                top_k=max_candidates  # 这里仍保留top_k，但会根据threshold过滤
            )
            
            # Filter by similarity threshold
            filtered_candidates = [
                (name, score) for name, score in candidates 
                if score >= similarity_threshold
            ]
            
            return filtered_candidates
            
        except Exception as e:
            self.logger.error(f"Layer 2 execution failed: {e}")
            return []
    
    def _metadata_search(self, query_table: Dict, all_tables: List[Dict], 
                        analysis: Any) -> List[tuple]:
        """
        Perform metadata-based filtering
        """
        try:
            # 根据使用的过滤器类型调用不同的方法
            if self.use_enhanced_filter:
                # SMD增强过滤器
                candidates = self.metadata_filter.filter_candidates(
                    query_table=query_table,
                    all_tables=all_tables,
                    threshold=0.25,  # 降低阈值以捕获同组表
                    max_candidates=100
                )
            else:
                # 原有的元数据过滤器
                criteria = {
                    'column_count': len(query_table.get('columns', [])),
                    'column_names': analysis.column_names if analysis else [],
                    'key_columns': analysis.key_columns if analysis else [],
                    'table_type': analysis.table_type if analysis else 'unknown'
                }
                
                candidates = self.metadata_filter.filter(
                    query_table,
                    all_tables,
                    criteria
                )
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Metadata search failed: {e}")
            return []
    
    def _vector_search(self, query_table: Dict, all_tables: List[Dict], 
                       top_k: int = 50) -> List[tuple]:
        """
        Perform vector similarity search
        """
        try:
            # Use vector search tool
            candidates = self.vector_search.search(
                query_table,
                all_tables,
                top_k=top_k
            )
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
    
    def validate_input(self, state: WorkflowState) -> bool:
        """
        Validate required inputs
        """
        if 'query_table' not in state:
            self.logger.error("Missing query_table in state")
            return False
        
        if 'all_tables' not in state:
            self.logger.warning("Missing all_tables in state")
            state['all_tables'] = []
        
        return True
    
    def _process_nlctables(self, state: WorkflowState) -> WorkflowState:
        """
        Process NLCTables queries with semantic search
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with semantically matched candidates
        """
        self.logger.info("Starting semantic search for NLCTables query")
        
        # Get required inputs
        query_text = state.get('query_text', '')
        nl_features = state.get('nl_features', {})
        all_tables = state.get('all_tables', [])
        strategy = state.get('strategy')
        analysis = state.get('analysis')
        
        if not query_text and not nl_features:
            self.logger.error("No NL query information found")
            return state
        
        # Initialize candidates list
        candidates = []
        candidate_scores = {}
        
        # For NLCTables, prioritize semantic search
        self.logger.info("Performing semantic search for NL query")
        
        # Layer 1: Keyword-based metadata filtering (if keywords available)
        keywords = nl_features.get('keywords', [])
        column_mentions = nl_features.get('column_mentions', [])
        
        if keywords or column_mentions:
            self.logger.info(f"Layer 1: Keyword/column filtering with {len(keywords)} keywords, {len(column_mentions)} columns")
            
            # Filter tables that contain mentioned columns or keywords
            for table in all_tables:
                table_name = table.get('table_name', '')
                columns = table.get('columns', [])
                column_names = [col.get('column_name', col.get('name', '')).lower() for col in columns]
                
                score = 0.0
                
                # Check column mentions
                for col_mention in column_mentions:
                    if any(col_mention.lower() in col_name for col_name in column_names):
                        score += 0.5
                
                # Check keywords in table name and columns
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in table_name.lower():
                        score += 0.3
                    if any(keyword_lower in col_name for col_name in column_names):
                        score += 0.2
                
                if score > 0:
                    candidate_scores[table_name] = {'keyword': score, 'semantic': 0}
            
            self.logger.info(f"  - Found {len(candidate_scores)} candidates from keyword matching")
        
        # Layer 2: Semantic vector search
        self.logger.info("Layer 2: Semantic similarity search")
        
        # Create a pseudo query table from NL features for vector search
        pseudo_query = {
            'table_name': query_text[:50],  # Use first 50 chars as table name
            'columns': []
        }
        
        # Add mentioned columns to pseudo query
        for col in column_mentions:
            pseudo_query['columns'].append({
                'column_name': col,
                'data_type': 'text',  # Default type for NL queries
                'sample_values': keywords[:3]  # Use keywords as sample values
            })
        
        # If no columns mentioned, create columns from keywords
        if not pseudo_query['columns'] and keywords:
            for keyword in keywords[:5]:  # Limit to 5 keywords
                pseudo_query['columns'].append({
                    'column_name': keyword,
                    'data_type': 'text',
                    'sample_values': [keyword]
                })
        
        # Perform vector search with pseudo query
        try:
            vector_candidates = self.vector_search.search(
                pseudo_query,
                all_tables,
                top_k=strategy.top_k if strategy else 150  # More candidates for semantic search
            )
            
            for table_name, score in vector_candidates:
                if table_name not in candidate_scores:
                    candidate_scores[table_name] = {'keyword': 0, 'semantic': 0}
                candidate_scores[table_name]['semantic'] = score
            
            self.logger.info(f"  - Found {len(vector_candidates)} candidates from semantic search")
        except Exception as e:
            self.logger.error(f"Semantic vector search failed: {e}")
        
        # Combine and rank candidates with semantic weighting
        self.logger.info("Combining semantic search results")
        
        for table_name, scores in candidate_scores.items():
            # For NLCTables, weight semantic similarity higher
            keyword_weight = 0.3
            semantic_weight = 0.7
            
            combined_score = (
                scores['keyword'] * keyword_weight + 
                scores['semantic'] * semantic_weight
            )
            
            # Create candidate
            candidate = CandidateTable(
                table_name=table_name,
                score=combined_score,
                source='semantic_search',
                evidence={
                    'keyword_score': scores['keyword'],
                    'semantic_score': scores['semantic'],
                    'query_text': query_text[:100]
                }
            )
            candidates.append(candidate)
        
        # Sort by score
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Apply top-k limit
        if strategy and strategy.top_k:
            candidates = candidates[:strategy.top_k]
        else:
            candidates = candidates[:150]  # Default higher limit for NL queries
        
        self.logger.info(f"Final NL candidate list: {len(candidates)} tables")
        if candidates:
            self.logger.info(f"  - Top candidate: {candidates[0].table_name} (score: {candidates[0].score:.3f})")
            # Log top 5 for debugging
            for i, cand in enumerate(candidates[:5]):
                self.logger.debug(f"    {i+1}. {cand.table_name}: {cand.score:.3f}")
        
        # Update state
        state['candidates'] = candidates
        state['semantic_search'] = True  # Flag to indicate semantic search was used
        
        # Update metrics
        if state.get('metrics'):
            state['metrics'].candidates_generated = len(candidates)
        
        return state