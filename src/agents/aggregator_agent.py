"""
AggregatorAgent - Result aggregation and ranking
"""
from typing import List, Dict, Any
from collections import defaultdict
from src.agents.base_agent import BaseAgent
from src.core.state import WorkflowState, MatchResult, CandidateTable


class AggregatorAgent(BaseAgent):
    """
    Aggregator Agent responsible for combining results and final ranking
    """
    
    def __init__(self):
        super().__init__(
            name="AggregatorAgent",
            description="Aggregates and ranks final results from all sources"
        )
        
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Aggregate results from different sources and produce final ranking
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with final results
        """
        self.logger.info("Starting result aggregation and ranking")
        
        # Get matches and candidates
        matches = state.get('matches', [])
        candidates = state.get('candidates', [])
        strategy = state.get('strategy')
        
        # Combine all results
        all_results = {}
        
        # Add matches from MatcherAgent
        for match in matches:
            table_name = match.matched_table
            if table_name not in all_results or match.score > all_results[table_name].score:
                all_results[table_name] = match
        
        # Add high-confidence candidates that weren't matched
        for candidate in candidates:
            if candidate.table_name not in all_results:
                # Create a match result from candidate
                query_table = state.get('query_table', {})
                query_task = state.get('query_task')
                
                match = MatchResult(
                    query_table=query_table.get('table_name', ''),
                    matched_table=candidate.table_name,
                    score=candidate.score,
                    match_type=query_task.task_type if query_task else 'join',
                    confidence=candidate.score,
                    agent_used='AggregatorAgent-Candidate',
                    evidence=candidate.evidence
                )
                
                # Only add if meets minimum threshold
                min_threshold = strategy.confidence_threshold if strategy else 0.5
                if candidate.score >= min_threshold:
                    all_results[candidate.table_name] = match
        
        # Convert to list and sort
        final_results = list(all_results.values())
        
        # Apply multi-criteria ranking
        final_results = self._rank_results(final_results)
        
        # Apply Top-K limit
        top_k = 10  # Default
        if strategy and hasattr(strategy, 'top_k'):
            top_k = min(10, strategy.top_k // 5)  # Return top 20% of candidates
        
        final_results = final_results[:top_k]
        
        # Generate explanations for top results
        for i, result in enumerate(final_results[:5]):  # Explain top 5
            result.evidence['explanation'] = self._generate_explanation(result, i + 1)
        
        # Log results
        self.logger.info(f"Aggregation complete: {len(final_results)} final results")
        if final_results:
            self.logger.info("Top 5 results:")
            for i, result in enumerate(final_results[:5]):
                self.logger.info(
                    f"  {i+1}. {result.matched_table} "
                    f"(score: {result.score:.3f}, confidence: {result.confidence:.3f})"
                )
        
        # Update state
        state['final_results'] = final_results
        
        # Calculate final metrics
        if state.get('metrics'):
            state['metrics'].total_time = sum(
                state['metrics'].agent_times.values()
            )
        
        return state
    
    def _rank_results(self, results: List[MatchResult]) -> List[MatchResult]:
        """
        Apply multi-criteria ranking to results
        """
        # Calculate composite scores
        for result in results:
            # Factors to consider:
            # 1. Base score (40%)
            # 2. Confidence (30%)
            # 3. Source diversity bonus (20%)
            # 4. Agent trust level (10%)
            
            base_weight = 0.4
            confidence_weight = 0.3
            source_weight = 0.2
            agent_weight = 0.1
            
            # Source diversity bonus
            source_bonus = 0.0
            if 'metadata_score' in result.evidence and result.evidence['metadata_score'] > 0:
                source_bonus += 0.5
            if 'vector_score' in result.evidence and result.evidence['vector_score'] > 0:
                source_bonus += 0.5
            
            # Agent trust level
            agent_trust = {
                'MatcherAgent-LLM': 1.0,
                'MatcherAgent-HighConfidence': 0.9,
                'AggregatorAgent-Candidate': 0.7
            }.get(result.agent_used, 0.5)
            
            # Calculate composite score
            composite_score = (
                result.score * base_weight +
                result.confidence * confidence_weight +
                source_bonus * source_weight +
                agent_trust * agent_weight
            )
            
            # Store composite score
            result.evidence['composite_score'] = composite_score
        
        # Sort by composite score
        results.sort(
            key=lambda x: x.evidence.get('composite_score', x.score),
            reverse=True
        )
        
        return results
    
    def _generate_explanation(self, result: MatchResult, rank: int) -> str:
        """
        Generate human-readable explanation for a match
        """
        explanations = []
        
        # Add rank
        explanations.append(f"Rank #{rank}")
        
        # Add match type
        explanations.append(f"Match type: {result.match_type.upper()}")
        
        # Add score information
        explanations.append(f"Overall score: {result.score:.2%}")
        explanations.append(f"Confidence: {result.confidence:.2%}")
        
        # Add source information
        if 'metadata_score' in result.evidence:
            explanations.append(f"Metadata match: {result.evidence['metadata_score']:.2%}")
        if 'vector_score' in result.evidence:
            explanations.append(f"Semantic similarity: {result.evidence['vector_score']:.2%}")
        
        # Add LLM verification info
        if 'llm_result' in result.evidence:
            llm_result = result.evidence['llm_result']
            if 'reason' in llm_result:
                explanations.append(f"LLM reason: {llm_result['reason']}")
        
        # Add agent used
        explanations.append(f"Verified by: {result.agent_used}")
        
        return " | ".join(explanations)
    
    def validate_input(self, state: WorkflowState) -> bool:
        """
        Validate required inputs
        """
        # Aggregator can work with partial results
        return True