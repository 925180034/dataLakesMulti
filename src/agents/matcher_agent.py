"""
MatcherAgent - Precise matching with parallel LLM verification
"""
import asyncio
from typing import List, Dict, Any
from src.agents.base_agent import BaseAgent
from src.core.state import WorkflowState, MatchResult, CandidateTable
from src.tools.llm_matcher import LLMMatcherTool


class MatcherAgent(BaseAgent):
    """
    Matcher Agent responsible for precise matching using parallel LLM verification
    """
    
    def __init__(self):
        super().__init__(
            name="MatcherAgent",
            description="Performs precise table matching using parallel LLM calls"
        )
        
        # Initialize LLM matcher tool
        self.llm_matcher = LLMMatcherTool()
        
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Perform precise matching on candidates using LLM
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with match results
        """
        self.logger.info("Starting LLM-based precise matching")
        
        # Check if we should skip matching
        if state.get('skip_matcher', False):
            self.logger.info("Skipping matcher (flag set by planner)")
            state['matches'] = []
            return state
        
        # Get required inputs
        candidates = state.get('candidates', [])
        query_table = state.get('query_table')
        all_tables = state.get('all_tables', [])
        query_task = state.get('query_task')
        optimization_config = state.get('optimization_config')
        strategy = state.get('strategy')
        
        if not candidates:
            self.logger.info("No candidates to match")
            state['matches'] = []
            return state
        
        if not query_table:
            self.logger.error("No query table found")
            state['matches'] = []
            return state
        
        # Determine LLM concurrency
        llm_concurrency = 20  # Default
        if optimization_config:
            llm_concurrency = optimization_config.llm_concurrency
        
        self.logger.info(f"Using LLM concurrency: {llm_concurrency}")
        
        # Filter candidates for LLM verification
        llm_candidates = self._select_candidates_for_llm(
            candidates, 
            strategy.confidence_threshold if strategy else 0.5
        )
        
        self.logger.info(f"Selected {len(llm_candidates)} candidates for LLM verification")
        
        # Run async LLM matching
        matches = asyncio.run(
            self._run_parallel_llm_matching(
                query_table,
                llm_candidates,
                all_tables,
                query_task.task_type if query_task else 'join',
                llm_concurrency
            )
        )
        
        self.logger.info(f"LLM matching complete: {len(matches)} matches found")
        
        # Add high-confidence candidates that didn't need LLM
        for candidate in candidates:
            if candidate.score > 0.95:
                # Very high confidence, add directly
                match = MatchResult(
                    query_table=query_table.get('table_name', ''),
                    matched_table=candidate.table_name,
                    score=candidate.score,
                    match_type=query_task.task_type if query_task else 'join',
                    confidence=candidate.score,
                    agent_used='MatcherAgent-HighConfidence',
                    evidence=candidate.evidence
                )
                matches.append(match)
                self.logger.debug(f"Added high-confidence match: {candidate.table_name}")
        
        # Sort matches by score
        matches.sort(key=lambda x: x.score, reverse=True)
        
        # Update state
        state['matches'] = matches
        
        # Update metrics
        if state.get('metrics'):
            state['metrics'].llm_calls_made = len(llm_candidates)
        
        return state
    
    def _select_candidates_for_llm(self, candidates: List[CandidateTable], 
                                   threshold: float) -> List[CandidateTable]:
        """
        Select which candidates need LLM verification
        """
        llm_candidates = []
        
        for candidate in candidates:
            # Skip very high confidence candidates
            if candidate.score > 0.95:
                continue
            
            # Skip very low confidence candidates
            if candidate.score < threshold * 0.5:
                continue
            
            # Add medium confidence candidates for LLM verification
            llm_candidates.append(candidate)
        
        # Limit to top candidates to control costs
        return llm_candidates[:20]
    
    async def _run_parallel_llm_matching(
        self,
        query_table: Dict,
        candidates: List[CandidateTable],
        all_tables: List[Dict],
        task_type: str,
        concurrency: int
    ) -> List[MatchResult]:
        """
        Run LLM matching in parallel with controlled concurrency
        """
        matches = []
        
        # Create table lookup
        table_lookup = {t['table_name']: t for t in all_tables}
        
        # Create tasks for all candidates
        tasks = []
        for candidate in candidates:
            candidate_table = table_lookup.get(candidate.table_name)
            if candidate_table:
                task = self.llm_matcher.verify_match(
                    query_table,
                    candidate_table,
                    task_type,
                    candidate.score  # Pass existing score as context
                )
                tasks.append((candidate, task))
        
        # Execute in batches with controlled concurrency
        all_matches = []
        for i in range(0, len(tasks), concurrency):
            batch = tasks[i:i + concurrency]
            batch_tasks = [task for _, task in batch]
            batch_candidates = [candidate for candidate, _ in batch]
            
            try:
                # Run batch in parallel
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for j, result in enumerate(results):
                    candidate = batch_candidates[j]
                    
                    if isinstance(result, Exception):
                        self.logger.error(f"LLM matching failed for {candidate.table_name}: {result}")
                        continue
                    
                    if result and result.get('is_match', False):
                        # Create match result
                        match = MatchResult(
                            query_table=query_table.get('table_name', ''),
                            matched_table=candidate.table_name,
                            score=self._combine_scores(
                                candidate.score, 
                                result.get('confidence', 0.5)
                            ),
                            match_type=task_type,
                            confidence=result.get('confidence', 0.5),
                            agent_used='MatcherAgent-LLM',
                            evidence={
                                **candidate.evidence,
                                'llm_result': result
                            }
                        )
                        all_matches.append(match)
                        
            except Exception as e:
                self.logger.error(f"Batch LLM matching failed: {e}")
        
        return all_matches
    
    def _combine_scores(self, candidate_score: float, llm_confidence: float) -> float:
        """
        Combine candidate score with LLM confidence
        """
        # Weighted average: 40% candidate score, 60% LLM confidence
        return candidate_score * 0.4 + llm_confidence * 0.6
    
    def validate_input(self, state: WorkflowState) -> bool:
        """
        Validate required inputs
        """
        if 'candidates' not in state:
            self.logger.warning("No candidates in state")
            state['candidates'] = []
        
        if 'query_table' not in state:
            self.logger.error("Missing query_table in state")
            return False
        
        return True