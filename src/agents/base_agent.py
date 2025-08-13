"""
Base Agent Class for LangGraph Multi-Agent System
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import time
from src.core.state import WorkflowState, PerformanceMetrics


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize base agent
        
        Args:
            name: Agent name for identification
            description: Agent description
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"Agent.{name}")
        self.execution_count = 0
        
    def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Main entry point for LangGraph to call the agent
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        start_time = time.time()
        self.execution_count += 1
        
        # Track execution path
        if 'execution_path' not in state:
            state['execution_path'] = []
        state['execution_path'].append(self.name)
        
        # Initialize metrics if not present
        if 'metrics' not in state or state['metrics'] is None:
            state['metrics'] = PerformanceMetrics()
        
        try:
            # Log agent start
            self.logger.info(f"Starting {self.name} (execution #{self.execution_count})")
            
            # Execute agent logic
            state = self.process(state)
            
            # Record execution time
            execution_time = time.time() - start_time
            state['metrics'].agent_times[self.name] = execution_time
            
            # Log agent completion
            self.logger.info(f"Completed {self.name} in {execution_time:.2f}s")
            
        except Exception as e:
            # Handle errors gracefully
            error_msg = f"Error in {self.name}: {str(e)}"
            self.logger.error(error_msg)
            
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(error_msg)
            
            # Record error in metrics
            state['metrics'].errors_count += 1
            
            # Call error handler
            state = self.handle_error(state, e)
        
        return state
    
    @abstractmethod
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Main processing logic for the agent
        Must be implemented by each specific agent
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        pass
    
    def handle_error(self, state: WorkflowState, error: Exception) -> WorkflowState:
        """
        Error handling logic
        Can be overridden by specific agents for custom error handling
        
        Args:
            state: Current workflow state
            error: The exception that occurred
            
        Returns:
            Updated workflow state
        """
        # Default error handling - just log and continue
        self.logger.error(f"Error in {self.name}: {error}")
        return state
    
    def validate_input(self, state: WorkflowState) -> bool:
        """
        Validate that the agent has required inputs in state
        Can be overridden by specific agents
        
        Args:
            state: Current workflow state
            
        Returns:
            True if inputs are valid, False otherwise
        """
        return True
    
    def log_metrics(self, **kwargs):
        """
        Log metrics for monitoring
        
        Args:
            **kwargs: Metric key-value pairs to log
        """
        for key, value in kwargs.items():
            self.logger.debug(f"Metric - {key}: {value}")