"""
Evolution agent for the AI-Coscientist system.
"""
from pydantic_ai import Agent, RunContext
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

from typing import Dict, Any, List, Optional
import json

from .base import AgentDependencies

class EvolutionAgent:
    """
    Evolution agent that refines and improves existing hypotheses.
    
    The Evolution agent is responsible for:
    1. Enhancement through grounding
    2. Coherence and feasibility improvements
    3. Inspiration from existing hypotheses
    4. Combination of top hypotheses
    5. Simplification
    6. Out-of-box thinking
    """
    
    def __init__(self, model_name: str = "google-gla:gemini-2.0-flash"):
        """Initialize the Evolution agent."""
        self.agent = Agent(
            model_name,
            deps_type=AgentDependencies,
            tools=[duckduckgo_search_tool()],
            system_prompt=(
                "You are the Evolution agent for the AI co-scientist system. "
                "Your role is to refine and improve existing research hypotheses "
                "through various approaches including grounding, coherence improvements, "
                "combination, simplification, and out-of-box thinking."
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
        async def enhance_through_grounding(
            ctx: RunContext[AgentDependencies], 
            hypothesis_id: str
        ) -> Dict[str, Any]:
            """Enhance a hypothesis by grounding it in existing literature."""
            hypothesis = ctx.deps.context_memory.get_hypothesis_by_id(hypothesis_id)
            if not hypothesis:
                return {"error": f"Hypothesis with ID {hypothesis_id} not found"}
            
            # In a real implementation, this would use web search to find relevant literature
            # and then use the agent to enhance the hypothesis
            enhanced_hypothesis = {
                "title": f"Enhanced: {hypothesis.get('title', 'Untitled')}",
                "description": f"Enhanced version of: {hypothesis.get('description', '')}",
                "rationale": f"Enhanced rationale with literature grounding: {hypothesis.get('rationale', '')}",
                "novelty_justification": hypothesis.get('novelty_justification', ''),
                "testability_plan": hypothesis.get('testability_plan', ''),
                "generation_method": "enhancement",
                "parent_hypothesis_id": hypothesis_id,
                "created_at": "2025-04-14",  # In a real implementation, use actual timestamp
            }
            
            new_id = ctx.deps.context_memory.add_hypothesis(enhanced_hypothesis)
            return {"id": new_id, **enhanced_hypothesis}
        
        @self.agent.tool
        async def improve_coherence(
            ctx: RunContext[AgentDependencies], 
            hypothesis_id: str
        ) -> Dict[str, Any]:
            """Improve the coherence and feasibility of a hypothesis."""
            hypothesis = ctx.deps.context_memory.get_hypothesis_by_id(hypothesis_id)
            if not hypothesis:
                return {"error": f"Hypothesis with ID {hypothesis_id} not found"}
            
            # In a real implementation, this would use the agent to improve coherence
            improved_hypothesis = {
                "title": f"More Coherent: {hypothesis.get('title', 'Untitled')}",
                "description": f"More coherent version of: {hypothesis.get('description', '')}",
                "rationale": hypothesis.get('rationale', ''),
                "novelty_justification": hypothesis.get('novelty_justification', ''),
                "testability_plan": f"More feasible testing plan: {hypothesis.get('testability_plan', '')}",
                "generation_method": "coherence_improvement",
                "parent_hypothesis_id": hypothesis_id,
                "created_at": "2025-04-14",  # In a real implementation, use actual timestamp
            }
            
            new_id = ctx.deps.context_memory.add_hypothesis(improved_hypothesis)
            return {"id": new_id, **improved_hypothesis}
        
        @self.agent.tool
        async def combine_hypotheses(
            ctx: RunContext[AgentDependencies], 
            hypothesis_ids: List[str]
        ) -> Dict[str, Any]:
            """Combine multiple hypotheses into a new one."""
            hypotheses = [
                ctx.deps.context_memory.get_hypothesis_by_id(h_id) 
                for h_id in hypothesis_ids
            ]
            
            if not all(hypotheses) or len(hypotheses) < 2:
                return {"error": "One or more hypotheses not found or insufficient hypotheses"}
            
            # In a real implementation, this would use the agent to combine hypotheses
            combined_hypothesis = {
                "title": f"Combined: {' + '.join([h.get('title', 'Untitled') for h in hypotheses])}",
                "description": "A combination of multiple hypotheses",
                "rationale": "Rationale combining insights from multiple hypotheses",
                "novelty_justification": "Novel combination of existing ideas",
                "testability_plan": "Testing plan for the combined hypothesis",
                "generation_method": "combination",
                "parent_hypothesis_ids": hypothesis_ids,
                "created_at": "2025-04-14",  # In a real implementation, use actual timestamp
            }
            
            new_id = ctx.deps.context_memory.add_hypothesis(combined_hypothesis)
            return {"id": new_id, **combined_hypothesis}
        
        @self.agent.tool
        async def simplify_hypothesis(
            ctx: RunContext[AgentDependencies], 
            hypothesis_id: str
        ) -> Dict[str, Any]:
            """Simplify a hypothesis for easier verification and testing."""
            hypothesis = ctx.deps.context_memory.get_hypothesis_by_id(hypothesis_id)
            if not hypothesis:
                return {"error": f"Hypothesis with ID {hypothesis_id} not found"}
            
            # In a real implementation, this would use the agent to simplify the hypothesis
            simplified_hypothesis = {
                "title": f"Simplified: {hypothesis.get('title', 'Untitled')}",
                "description": f"Simplified version of: {hypothesis.get('description', '')}",
                "rationale": f"Simplified rationale: {hypothesis.get('rationale', '')}",
                "novelty_justification": hypothesis.get('novelty_justification', ''),
                "testability_plan": f"Simplified testing plan: {hypothesis.get('testability_plan', '')}",
                "generation_method": "simplification",
                "parent_hypothesis_id": hypothesis_id,
                "created_at": "2025-04-14",  # In a real implementation, use actual timestamp
            }
            
            new_id = ctx.deps.context_memory.add_hypothesis(simplified_hypothesis)
            return {"id": new_id, **simplified_hypothesis}
    
    async def evolve_hypothesis(
        self, 
        deps: AgentDependencies, 
        hypothesis_id: str,
        method: str = "enhance"
    ) -> Dict[str, Any]:
        """
        Evolve a hypothesis using the specified method.
        
        Args:
            deps: The agent dependencies
            hypothesis_id: The ID of the hypothesis to evolve
            method: The evolution method to use (enhance, improve, simplify)
            
        Returns:
            The evolved hypothesis
        """
        if method == "enhance":
            result = await self.agent.run(
                f"Enhance hypothesis {hypothesis_id} through grounding in literature",
                deps=deps
            )
        elif method == "improve":
            result = await self.agent.run(
                f"Improve the coherence and feasibility of hypothesis {hypothesis_id}",
                deps=deps
            )
        elif method == "simplify":
            result = await self.agent.run(
                f"Simplify hypothesis {hypothesis_id} for easier verification and testing",
                deps=deps
            )
        else:
            return {"error": f"Unknown evolution method: {method}"}
        
        return result.data
    
    async def combine_top_hypotheses(
        self, 
        deps: AgentDependencies, 
        num_hypotheses: int = 3
    ) -> Dict[str, Any]:
        """
        Combine top hypotheses into a new one.
        
        Args:
            deps: The agent dependencies
            num_hypotheses: The number of top hypotheses to combine
            
        Returns:
            The combined hypothesis
        """
        top_hypotheses = deps.context_memory.get_top_hypotheses(num_hypotheses)
        if len(top_hypotheses) < 2:
            return {"error": "Not enough hypotheses to combine"}
        
        hypothesis_ids = [h["id"] for h in top_hypotheses]
        result = await self.agent.run(
            f"Combine the following hypotheses into a new one: {', '.join(hypothesis_ids)}",
            deps=deps
        )
        
        return result.data
