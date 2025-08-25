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
        # Check if this is an NLCTables query
        query_type = state.get('query_type', 'webtables')
        
        if query_type == 'nlctables':
            return self._process_nlctables(state)
        
        # Original WebTables processing
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
    
    def _process_nlctables(self, state: WorkflowState) -> WorkflowState:
        """
        Process NLCTables queries with natural language condition verification
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with NL-verified matches
        """
        self.logger.info("Starting NL-based table matching for NLCTables query")
        
        # Get candidates and NL query info
        candidates = state.get('candidates', [])
        query_text = state.get('query_text', '')
        nl_features = state.get('nl_features', {})
        
        if not candidates:
            self.logger.warning("No candidates to match for NL query")
            state['matches'] = []
            return state
        
        # Check if LLM should be used
        if not self.use_llm or state.get('should_use_llm', True) == False:
            self.logger.info("Passing through NL candidates without LLM verification")
            # Convert candidates to matches without LLM
            state['matches'] = []
            for c in candidates[:10]:
                state['matches'].append(MatchResult(
                    query_table=f"NL_Query: {query_text[:50]}",
                    matched_table=c.table_name,
                    score=c.score,
                    match_type='nl_search',
                    confidence=c.score * 0.6,  # Lower confidence without LLM
                    agent_used='SearcherAgent',
                    evidence={'reason': 'Semantic similarity without LLM verification'}
                ))
            return state
        
        # Process candidates with NL-specific LLM verification
        max_candidates = min(len(candidates), 15)  # Process more candidates for NL
        self.logger.info(f"Processing {max_candidates} NL candidates with LLM")
        
        match_results = []
        all_tables = state.get('all_tables', [])
        
        # Process candidates in batches
        batch_size = 5
        for i in range(0, max_candidates, batch_size):
            batch_candidates = candidates[i:i+batch_size]
            
            for candidate in batch_candidates:
                # Find full table info
                candidate_table = None
                for t in all_tables:
                    if t.get('table_name') == candidate.table_name:
                        candidate_table = t
                        break
                
                if not candidate_table:
                    continue
                
                # Verify if table satisfies NL conditions
                try:
                    is_match, confidence = self._verify_nl_match(
                        query_text,
                        nl_features,
                        candidate_table,
                        candidate.score
                    )
                    
                    if is_match:
                        match = MatchResult(
                            query_table=f"NL: {query_text[:50]}",
                            matched_table=candidate.table_name,
                            score=confidence,
                            match_type='nl_search',
                            confidence=confidence,
                            agent_used='MatcherAgent',
                            evidence={
                                'reason': 'NL condition verification successful',
                                'query_text': query_text,
                                'semantic_score': candidate.score
                            }
                        )
                        match_results.append(match)
                        self.logger.debug(f"NL match found: {candidate.table_name} (confidence: {confidence:.3f})")
                        
                except Exception as e:
                    self.logger.error(f"Error verifying NL match for {candidate.table_name}: {e}")
                    continue
        
        # Sort by confidence
        match_results.sort(key=lambda x: x.confidence, reverse=True)
        
        # Store results
        state['matches'] = match_results
        state['nl_matching'] = True  # Flag to indicate NL matching was performed
        
        self.logger.info(f"NL matching complete: {len(match_results)} matches found")
        if match_results:
            self.logger.info(f"Top NL match: {match_results[0].matched_table} (confidence: {match_results[0].confidence:.3f})")
            # Log top 3 for debugging
            for i, match in enumerate(match_results[:3]):
                self.logger.debug(f"  {i+1}. {match.matched_table}: {match.confidence:.3f}")
        
        return state
    
    def _verify_nl_match(self, query_text: str, nl_features: Dict, 
                        candidate_table: Dict, semantic_score: float) -> tuple:
        """
        Verify if a table matches the natural language query conditions
        
        Returns:
            (is_match, confidence) tuple
        """
        # Format table info
        table_name = candidate_table.get('table_name', '')
        columns = candidate_table.get('columns', [])
        column_info = []
        
        for col in columns[:20]:  # Limit columns for prompt
            col_name = col.get('column_name', col.get('name', ''))
            col_type = col.get('data_type', col.get('type', ''))
            samples = col.get('sample_values', [])
            
            col_str = f"{col_name} ({col_type})"
            if samples:
                col_str += f" [{', '.join(str(s) for s in samples[:3])}]"
            column_info.append(col_str)
        
        # Create NL verification prompt
        prompt = f"""
        Verify if this table satisfies the natural language query.
        
        Query: {query_text}
        Keywords: {nl_features.get('keywords', [])}
        Topics: {nl_features.get('topics', [])}
        Column Mentions: {nl_features.get('column_mentions', [])}
        
        Table Name: {table_name}
        Columns: {'; '.join(column_info)}
        
        Semantic Similarity Score: {semantic_score:.3f}
        
        Based on the query requirements and table structure:
        1. Does this table contain relevant data for the query?
        2. Are the mentioned columns or similar columns present?
        3. Does the table's domain match the query intent?
        
        Respond in JSON format:
        {{
            "is_match": true/false,
            "confidence": 0.0-1.0,
            "reason": "explanation"
        }}
        """
        
        try:
            # Use the agent's LLM call method
            loop = None
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.call_llm_json(prompt, self.system_prompt))
                    response = future.result(timeout=10)
            except RuntimeError:
                response = asyncio.run(self.call_llm_json(prompt, self.system_prompt))
            
            if response:
                is_match = response.get('is_match', False)
                # Combine LLM confidence with semantic score
                llm_confidence = response.get('confidence', 0.5)
                final_confidence = (llm_confidence * 0.7 + semantic_score * 0.3)
                return is_match, final_confidence
            else:
                # Fallback to semantic score
                return semantic_score > 0.5, semantic_score
                
        except Exception as e:
            self.logger.error(f"NL verification LLM call failed: {e}")
            # Fallback to semantic score
            return semantic_score > 0.5, semantic_score