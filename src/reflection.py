"""
Reflection agent for the AI-Coscientist system.
"""
from pydantic_ai import Agent, RunContext
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from typing import Dict, Any, List, Optional
import json

from .base import AgentDependencies

class ReflectionAgent:
    """
    Reflection agent that reviews and critiques hypotheses.
    
    The Reflection agent is responsible for:
    1. Initial review
    2. Full review with web search
    3. Deep verification review
    4. Observation review
    5. Simulation review
    6. Tournament review
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """Initialize the Reflection agent."""
        self.agent = Agent(
            model_name,
            deps_type=AgentDependencies,
            tools=[duckduckgo_search_tool()],
            system_prompt=(
                "You are the Reflection agent for the AI co-scientist system. "
                "Your role is to critically examine the correctness, quality, and novelty "
                "of generated hypotheses. You should identify flaws, verify claims, "
                "and assess the novelty of hypotheses against existing literature."
            ),
        )
        
        # Register system prompt function
        @self.agent.system_prompt
        async def add_research_context(ctx: RunContext[AgentDependencies]) -> str:
            """Add research context to the system prompt."""
            return (
                f"Current research goal: {ctx.deps.research_plan.goal}\n"
                f"Research plan: {json.dumps(ctx.deps.research_plan.to_dict(), indent=2)}"
            )
        
        # Register tools
        @self.agent.tool
        async def initial_review(
            ctx: RunContext[AgentDependencies], 
            hypothesis_id: str
        ) -> Dict[str, Any]:
            """Perform an initial review of a hypothesis without using external tools."""
            hypothesis = ctx.deps.context_memory.get_hypothesis_by_id(hypothesis_id)
            if not hypothesis:
                return {"error": f"Hypothesis with ID {hypothesis_id} not found"}
            
            # In a real implementation, this would use the agent to generate a review
            review = {
                "type": "initial",
                "hypothesis_id": hypothesis_id,
                "correctness_score": 0.8,
                "quality_score": 0.7,
                "novelty_score": 0.9,
                "safety_score": 1.0,
                "comments": f"Initial review of hypothesis: {hypothesis.get('title', 'Untitled')}",
                "created_at": "2025-04-14",  # In a real implementation, use actual timestamp
            }
            
            ctx.deps.context_memory.add_review(hypothesis_id, review)
            return review
        
        @self.agent.tool
        async def full_review(
            ctx: RunContext[AgentDependencies], 
            hypothesis_id: str
        ) -> Dict[str, Any]:
            """Perform a full review of a hypothesis using web search."""
            hypothesis = ctx.deps.context_memory.get_hypothesis_by_id(hypothesis_id)
            if not hypothesis:
                return {"error": f"Hypothesis with ID {hypothesis_id} not found"}
            
            # In a real implementation, this would use web search to find relevant literature
            # and then use the agent to generate a review
            review = {
                "type": "full",
                "hypothesis_id": hypothesis_id,
                "correctness_score": 0.75,
                "quality_score": 0.8,
                "novelty_score": 0.85,
                "safety_score": 1.0,
                "literature_references": ["ref1", "ref2", "ref3"],
                "comments": f"Full review of hypothesis: {hypothesis.get('title', 'Untitled')}",
                "created_at": "2025-04-14",  # In a real implementation, use actual timestamp
            }
            
            ctx.deps.context_memory.add_review(hypothesis_id, review)
            return review
        
        @self.agent.tool
        async def deep_verification_review(
            ctx: RunContext[AgentDependencies], 
            hypothesis_id: str
        ) -> Dict[str, Any]:
            """Perform a deep verification review by decomposing the hypothesis into assumptions."""
            hypothesis = ctx.deps.context_memory.get_hypothesis_by_id(hypothesis_id)
            if not hypothesis:
                return {"error": f"Hypothesis with ID {hypothesis_id} not found"}
            
            # In a real implementation, this would decompose the hypothesis into assumptions
            # and evaluate each one independently
            review = {
                "type": "deep_verification",
                "hypothesis_id": hypothesis_id,
                "assumptions": [
                    {"text": "Assumption 1", "correctness": 0.9, "evidence": "Evidence for assumption 1"},
                    {"text": "Assumption 2", "correctness": 0.7, "evidence": "Evidence for assumption 2"},
                ],
                "overall_correctness": 0.8,
                "comments": f"Deep verification of hypothesis: {hypothesis.get('title', 'Untitled')}",
                "created_at": "2025-04-14",  # In a real implementation, use actual timestamp
            }
            
            ctx.deps.context_memory.add_review(hypothesis_id, review)
            return review
    
    async def review_hypothesis(self, deps: AgentDependencies, hypothesis_id: str) -> Dict[str, Any]:
        """
        Review a hypothesis with the given ID.
        
        Args:
            deps: The agent dependencies
            hypothesis_id: The ID of the hypothesis to review
            
        Returns:
            The review results
        """
        # First perform an initial review
        result = await self.agent.run(
            f"Perform an initial review of hypothesis {hypothesis_id}",
            deps=deps
        )
        
        # Then perform a full review if the initial review is positive
        initial_review = result.data
        if initial_review.get("correctness_score", 0) > 0.5:
            result = await self.agent.run(
                f"Perform a full review of hypothesis {hypothesis_id}",
                deps=deps
            )
            
            # Perform a deep verification review for high-quality hypotheses
            full_review = result.data
            if full_review.get("quality_score", 0) > 0.7:
                result = await self.agent.run(
                    f"Perform a deep verification review of hypothesis {hypothesis_id}",
                    deps=deps
                )
        
        return result.data
