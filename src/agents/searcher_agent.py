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
    """
    
    def __init__(self):
        super().__init__(
            name="SearcherAgent",
            description="Searches for candidate tables using metadata filtering and vector search",
            use_llm=True  # Enable LLM for intelligent search strategy
        )
        
        # Use centralized prompt from config
        self.system_prompt = get_agent_prompt("SearcherAgent", "system_prompt")
        
        # Initialize search tools
        # 使用SMD增强过滤器替代原有的元数据过滤器
        self.use_enhanced_filter = os.getenv('USE_SMD_ENHANCED', 'true').lower() == 'true'
        if self.use_enhanced_filter:
            self.logger.info("Using SMD Enhanced Metadata Filter")
            self.metadata_filter = SMDEnhancedMetadataFilter(max_features=1000)
        else:
            self.metadata_filter = MetadataFilterTool()
        self.vector_search = VectorSearchTool()
        
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Search for candidate tables using multi-layer approach
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with candidate tables
        """
        # Check if this is an NLCTables query
        query_type = state.get('query_type', 'webtables')
        
        if query_type == 'nlctables':
            return self._process_nlctables(state)
        
        # Original WebTables processing
        self.logger.info("Starting multi-layer candidate search")
        
        # Get required inputs
        query_table = state.get('query_table')
        all_tables = state.get('all_tables', [])
        strategy = state.get('strategy')
        analysis = state.get('analysis')
        
        if not query_table:
            self.logger.error("No query table found")
            return state
        
        # Initialize candidates list
        candidates = []
        candidate_scores = {}  # Track scores from different sources
        
        # Check environment variables for layer control
        skip_metadata = os.environ.get('SKIP_METADATA', 'false').lower() == 'true'
        skip_vector = os.environ.get('SKIP_VECTOR', 'false').lower() == 'true'
        
        # Layer 1: Metadata Filtering
        if strategy and strategy.use_metadata and not skip_metadata:
            self.logger.info("Layer 1: Metadata filtering")
            
            metadata_candidates = self._metadata_search(
                query_table, 
                all_tables, 
                analysis
            )
            
            for table_name, score in metadata_candidates:
                if table_name not in candidate_scores:
                    candidate_scores[table_name] = {'metadata': 0, 'vector': 0}
                candidate_scores[table_name]['metadata'] = score
            
            self.logger.info(f"  - Found {len(metadata_candidates)} candidates from metadata")
        
        # Layer 2: Vector Search
        if strategy and strategy.use_vector and not skip_vector:
            self.logger.info("Layer 2: Vector similarity search")
            
            vector_candidates = self._vector_search(
                query_table,
                all_tables,
                strategy.top_k if strategy else 50
            )
            
            for table_name, score in vector_candidates:
                if table_name not in candidate_scores:
                    candidate_scores[table_name] = {'metadata': 0, 'vector': 0}
                candidate_scores[table_name]['vector'] = score
            
            self.logger.info(f"  - Found {len(vector_candidates)} candidates from vector search")
        
        # Combine and rank candidates
        self.logger.info("Combining and ranking candidates")
        
        for table_name, scores in candidate_scores.items():
            # Calculate combined score (weighted average)
            metadata_weight = 0.4
            vector_weight = 0.6
            
            combined_score = (
                scores['metadata'] * metadata_weight + 
                scores['vector'] * vector_weight
            )
            
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
                    'vector_score': scores['vector']
                }
            )
            candidates.append(candidate)
        
        # Sort by score
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Apply top-k limit
        if strategy and strategy.top_k:
            candidates = candidates[:strategy.top_k]
        
        self.logger.info(f"Final candidate list: {len(candidates)} tables")
        if candidates:
            self.logger.info(f"  - Top candidate: {candidates[0].table_name} (score: {candidates[0].score:.3f})")
        
        # Update state
        state['candidates'] = candidates
        
        # Update metrics
        if state.get('metrics'):
            state['metrics'].candidates_generated = len(candidates)
        
        # Check if we should skip matcher based on high confidence
        if candidates and candidates[0].score > 0.95:
            self.logger.info("Top candidate has very high score (>0.95), may skip LLM verification")
            # Note: The workflow can decide whether to skip based on this
        
        return state
    
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