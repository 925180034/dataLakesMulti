"""
MatcherAgent - LLM-based table matching and verification
"""
import asyncio
import json
from typing import List, Dict, Any
from src.agents.base_agent import BaseAgent
from src.core.state import WorkflowState, MatchResult, CandidateTable
from src.tools.llm_matcher import LLMMatcherTool
from src.config.prompts import get_agent_prompt, format_user_prompt


class MatcherAgent(BaseAgent):
    """
    Matcher Agent responsible for LLM-based verification of table matches
    """
    
    def __init__(self):
        super().__init__(
            name="MatcherAgent",
            description="Uses LLM to verify and score table matches with intelligent prompts",
            use_llm=True  # Enable LLM for matching
        )
        
        # Initialize LLM matcher tool (for batch processing)
        self.llm_matcher = LLMMatcherTool()
        
        # Use centralized prompt from config
        self.system_prompt = get_agent_prompt("MatcherAgent", "system_prompt")
        
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Verify candidate tables using LLM with intelligent prompts
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with verified matches
        """
        # Check if we should skip LLM matching
        if state.get('should_use_llm', True) == False:
            self.logger.info("Skipping LLM matcher as per optimization config")
            # Pass through top candidates without LLM verification
            candidates = state.get('candidates', [])
            state['matches'] = []
            for c in candidates[:10]:
                # Create proper MatchResult with required fields
                state['matches'].append(MatchResult(
                    query_table=state.get('query_task', {}).table_name if state.get('query_task') else 'unknown',
                    matched_table=c.table_name,
                    score=c.score,
                    match_type='unknown',
                    confidence=c.score * 0.5,
                    agent_used='SearcherAgent',
                    evidence={'reason': 'No LLM verification performed'}
                ))
            return state
        
        self.logger.info("Starting LLM-based table matching with intelligent prompts")
        
        # Get candidates and query info
        candidates = state.get('candidates', [])
        query_task = state.get('query_task')
        optimization_config = state.get('optimization_config')
        
        if not candidates:
            self.logger.warning("No candidates to match")
            state['matches'] = []
            return state
        
        # Get query table info
        query_table_name = query_task.table_name if query_task else 'unknown'
        query_table_info = state.get('query_table')
        
        if not query_table_info:
            # Try to find query table in all_tables
            all_tables = state.get('all_tables', [])
            for t in all_tables:
                if t.get('table_name') == query_table_name:
                    query_table_info = t
                    break
        
        if not query_table_info:
            self.logger.warning("Query table info not found, using minimal info")
            query_table_info = {'table_name': query_table_name}
        
        # Determine batch size for parallel processing
        batch_size = optimization_config.llm_concurrency if optimization_config else 3
        max_candidates = min(len(candidates), 10)  # Limit total candidates
        
        self.logger.info(f"Processing {max_candidates} candidates with batch size {batch_size}")
        
        # Use the batch_verify method which exists in LLMMatcherTool
        try:
            # Prepare candidate table info
            candidate_tables = []
            for c in candidates[:max_candidates]:
                # Find full table info for each candidate
                candidate_info = None
                for t in state.get('all_tables', []):
                    if t.get('table_name') == c.table_name:
                        candidate_info = t
                        break
                
                if candidate_info:
                    candidate_tables.append(candidate_info)
                else:
                    # Use minimal info if full info not found
                    candidate_tables.append({
                        'table_name': c.table_name,
                        'columns': []
                    })
            
            # Run batch LLM verification
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            llm_results = loop.run_until_complete(
                self.llm_matcher.batch_verify(
                    query_table=query_table_info,
                    candidate_tables=candidate_tables,
                    task_type=query_task.task_type if query_task else 'join',
                    max_concurrent=batch_size,  # Fixed parameter name
                    existing_scores=[c.score for c in candidates[:max_candidates]]  # Pass existing scores
                )
            )
            loop.close()
            
            # Convert LLM results to MatchResult objects
            match_results = []
            for i, result in enumerate(llm_results):
                if result.get('is_match', False):
                    # Create MatchResult with proper structure
                    match = MatchResult(
                        query_table=query_table_name,
                        matched_table=candidates[i].table_name,
                        score=result.get('confidence', 0.5),
                        match_type=query_task.task_type if query_task else 'unknown',
                        confidence=result.get('confidence', 0.5),
                        agent_used='MatcherAgent',
                        evidence={'reason': result.get('reason', 'LLM verification successful')}
                    )
                    match_results.append(match)
                else:
                    self.logger.debug(f"Table {candidates[i].table_name} not matched: {result.get('reason')}")
            
            # Sort by score
            match_results.sort(key=lambda x: x.score, reverse=True)
            
            # Store results
            state['matches'] = match_results
            
            self.logger.info(f"LLM matching complete: {len(match_results)} matches found")
            if match_results:
                self.logger.info(f"Top match: {match_results[0].matched_table} (score: {match_results[0].score:.3f})")
            
        except Exception as e:
            self.logger.error(f"Error in batch LLM matching: {e}")
            # Fallback to simple scoring
            state['matches'] = []
            for c in candidates[:5]:
                # Create proper MatchResult with required fields
                state['matches'].append(MatchResult(
                    query_table=query_table_name,
                    matched_table=c.table_name,
                    score=c.score * 0.5,  # Reduce confidence without LLM
                    match_type=query_task.task_type if query_task else 'unknown',
                    confidence=c.score * 0.5,
                    agent_used='MatcherAgent',
                    evidence={'reason': f'LLM matching failed: {str(e)}'}
                ))
        
        return state
    
    def validate_input(self, state: WorkflowState) -> bool:
        """
        Validate required inputs
        """
        if 'candidates' not in state:
            self.logger.warning("No candidates in state, will return empty results")
            state['candidates'] = []
        
        return True