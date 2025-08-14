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
            # Use metadata filter tool
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