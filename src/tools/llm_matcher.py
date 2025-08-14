"""
LLMMatcherTool - Layer 3: LLM-based precise matching with parallel calls
"""
import asyncio
import time
import json
from typing import Dict, Any, List, Optional
import logging
from src.utils.llm_client_proxy import get_llm_client


class LLMMatcherTool:
    """
    LLM-based matcher for precise table matching
    Uses parallel LLM calls for efficiency
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm_client = None
        self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize LLM client"""
        try:
            self.llm_client = get_llm_client()
            self.logger.info("LLM client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            
    async def verify_match(self, query_table: Dict[str, Any], 
                          candidate_table: Dict[str, Any],
                          task_type: str = 'join',
                          existing_score: float = 0.5) -> Dict[str, Any]:
        """
        Verify if two tables match using LLM
        
        Args:
            query_table: Query table dictionary
            candidate_table: Candidate table dictionary  
            task_type: 'join' or 'union'
            existing_score: Score from previous layers
            
        Returns:
            Dictionary with match result and confidence
        """
        start_time = time.time()
        
        # Build prompt based on task type
        if task_type == 'join':
            prompt = self._build_join_prompt(query_table, candidate_table)
        else:
            prompt = self._build_union_prompt(query_table, candidate_table)
        
        try:
            # Call LLM
            response = await self.llm_client.generate(prompt)
            
            # Parse response
            result = self._parse_llm_response(response, existing_score)
            
            # Log performance
            elapsed = time.time() - start_time
            self.logger.debug(f"LLM verification took {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM verification failed: {e}")
            # Return conservative result on error
            return {
                'is_match': False,
                'confidence': existing_score * 0.5,
                'reason': f'LLM verification failed: {str(e)}'
            }
    
    def _build_join_prompt(self, query_table: Dict, candidate_table: Dict) -> str:
        """Build prompt for JOIN task verification"""
        prompt = f"""Analyze if these two tables can be joined effectively.

Query Table: {query_table.get('table_name')}
Columns: {self._format_columns(query_table.get('columns', [])[:10])}

Candidate Table: {candidate_table.get('table_name')}
Columns: {self._format_columns(candidate_table.get('columns', [])[:10])}

Consider:
1. Do they have joinable key columns (IDs, codes, etc)?
2. Are the data types compatible for joining?
3. Do the column names suggest a relationship?

Return JSON:
{{
  "is_match": true/false,
  "confidence": 0.0-1.0,
  "join_keys": ["column pairs that can be used for joining"],
  "reason": "brief explanation"
}}"""
        return prompt
    
    def _build_union_prompt(self, query_table: Dict, candidate_table: Dict) -> str:
        """Build prompt for UNION task verification"""
        prompt = f"""Analyze if these two tables can be unioned (have similar schema).

Query Table: {query_table.get('table_name')}
Columns: {self._format_columns(query_table.get('columns', [])[:10])}

Candidate Table: {candidate_table.get('table_name')}  
Columns: {self._format_columns(candidate_table.get('columns', [])[:10])}

Consider:
1. Do they have similar column structure?
2. Are the data types compatible?
3. Do they represent the same type of entity?

Return JSON:
{{
  "is_match": true/false,
  "confidence": 0.0-1.0,
  "column_overlap": 0.0-1.0,
  "reason": "brief explanation"
}}"""
        return prompt
    
    def _format_columns(self, columns: List[Dict]) -> str:
        """Format columns for prompt"""
        formatted = []
        for col in columns:
            name = col.get('column_name', col.get('name', ''))
            dtype = col.get('data_type', col.get('type', 'unknown'))
            samples = col.get('sample_values', [])
            
            col_str = f"{name} ({dtype})"
            if samples:
                col_str += f" [{', '.join(str(s) for s in samples[:2])}]"
            formatted.append(col_str)
            
        return ', '.join(formatted)
    
    def _parse_llm_response(self, response: str, existing_score: float) -> Dict[str, Any]:
        """Parse LLM response"""
        try:
            # Try to parse as JSON
            if isinstance(response, str):
                # Find JSON in response
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    result = json.loads(json_str)
                else:
                    # Fallback to simple parsing
                    result = {
                        'is_match': 'true' in response.lower() or 'yes' in response.lower(),
                        'confidence': existing_score,
                        'reason': response[:200]
                    }
            else:
                result = response
                
            # Ensure required fields
            if 'is_match' not in result:
                result['is_match'] = result.get('confidence', 0) > 0.5
            if 'confidence' not in result:
                result['confidence'] = existing_score
            if 'reason' not in result:
                result['reason'] = 'LLM verification completed'
                
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return {
                'is_match': False,
                'confidence': existing_score * 0.5,
                'reason': 'Failed to parse LLM response'
            }
    
    async def batch_verify(self, query_table: Dict[str, Any],
                          candidate_tables: List[Dict[str, Any]], 
                          task_type: str = 'join',
                          max_concurrent: int = 10,
                          existing_scores: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Verify multiple candidates in parallel
        
        Args:
            query_table: Query table
            candidate_tables: List of candidate tables
            task_type: 'join' or 'union'
            max_concurrent: Maximum concurrent LLM calls
            existing_scores: Optional list of scores from previous layers
            
        Returns:
            List of verification results
        """
        # Create verification tasks
        tasks = []
        for i, candidate in enumerate(candidate_tables):
            # Get existing score if provided
            existing_score = existing_scores[i] if existing_scores and i < len(existing_scores) else 0.5
            
            task = self.verify_match(
                query_table,
                candidate,
                task_type,
                existing_score
            )
            tasks.append(task)
        
        # Execute in batches
        results = []
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            # Process results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Batch verification failed: {result}")
                    results.append({
                        'is_match': False,
                        'confidence': 0.0,
                        'reason': f'Verification failed: {str(result)}'
                    })
                else:
                    results.append(result)
        
        return results