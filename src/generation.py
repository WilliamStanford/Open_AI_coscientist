"""
Generation agent for the AI-Coscientist system.
"""
from pydantic_ai import Agent, RunContext
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from typing import Dict, Any, List, Optional
import json

from .base import AgentDependencies

class GenerationAgent:
    """
    Generation agent that creates initial hypotheses and research proposals.
    
    The Generation agent is responsible for:
    1. Literature exploration via web search
    2. Simulated scientific debates
    3. Iterative assumptions identification
    4. Research expansion
    """
    
    def __init__(self, model_name: str = "google-gla:gemini-2.0-flash"):
        """Initialize the Generation agent."""
        self.agent = Agent(
            model_name,
            deps_type=AgentDependencies,
            tools=[duckduckgo_search_tool()],
            system_prompt=(
                "You are the Generation agent for the AI co-scientist system. "
                "Your role is to create novel research hypotheses and proposals "
                "based on the research goal and existing literature. "
                "Use web search to ground your hypotheses in existing knowledge."
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
        async def literature_exploration(
            ctx: RunContext[AgentDependencies], 
            query: str
        ) -> str:
            """Search for and summarize relevant literature."""
            # This would use the duckduckgo_search_tool to search for literature
            # and then summarize the results
            return f"Literature exploration results for query: {query}"
        
        @self.agent.tool
        async def simulated_debate(
            ctx: RunContext[AgentDependencies],
            hypothesis: str
        ) -> str:
            """Conduct a simulated scientific debate to refine a hypothesis."""
            # This would implement self-critique and self-play techniques
            return f"Refined hypothesis after debate: {hypothesis}"
        
        @self.agent.tool
        async def save_hypothesis(
            ctx: RunContext[AgentDependencies],
            title: str,
            description: str,
            rationale: str,
            novelty_justification: str,
            testability_plan: str
        ) -> Dict[str, Any]:
            """Save a generated hypothesis to the context memory."""
            hypothesis = {
                "title": title,
                "description": description,
                "rationale": rationale,
                "novelty_justification": novelty_justification,
                "testability_plan": testability_plan,
                "generation_method": "direct",
                "created_at": "2025-04-14",  # In a real implementation, use actual timestamp
            }
            
            hypothesis_id = ctx.deps.context_memory.add_hypothesis(hypothesis)
            return {"id": hypothesis_id, **hypothesis}
    
    async def generate_hypothesis(self, deps: AgentDependencies) -> Dict[str, Any]:
        """
        Generate a new hypothesis based on the research goal.
        
        Args:
            deps: The agent dependencies
            
        Returns:
            The generated hypothesis
        """
        result = await self.agent.run(
            f"Generate a novel research hypothesis for the following research goal: {deps.research_plan.goal}",
            deps=deps
        )
        
        # In a real implementation, parse the result to extract the hypothesis
        return result.data
