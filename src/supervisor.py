"""
Supervisor agent for the AI-Coscientist system.
"""
from pydantic_ai import Agent, RunContext
from typing import Dict, Any, List
import json
import asyncio

from .base import AgentDependencies, ContextMemory, ResearchPlan

class SupervisorAgent:
    """
    Supervisor agent that orchestrates the entire AI-Coscientist system.
    
    The Supervisor agent is responsible for:
    1. Parsing research goals into research plan configurations
    2. Assigning tasks to specialized agents
    3. Monitoring system progress and computing statistics
    4. Managing context memory for persistent state
    """
    
    def __init__(self, model_name: str = "google-gla:gemini-2.0-flash"):
        """Initialize the Supervisor agent."""
        self.agent = Agent(
            model_name,
            deps_type=AgentDependencies,
            system_prompt=(
                "You are the Supervisor agent for the AI co-scientist system. "
                "Your role is to orchestrate the specialized agents, manage the task queue, "
                "and ensure the system progresses towards generating high-quality "
                "research hypotheses and proposals."
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
        async def parse_research_goal(
            ctx: RunContext[AgentDependencies], 
            research_goal: str
        ) -> Dict[str, Any]:
            """Parse a research goal into a research plan configuration."""
            research_plan = ResearchPlan.from_research_goal(research_goal)
            ctx.deps.research_plan = research_plan
            return research_plan.to_dict()
        
        @self.agent.tool
        async def compute_statistics(
            ctx: RunContext[AgentDependencies]
        ) -> Dict[str, Any]:
            """Compute statistics about the current state of the system."""
            stats = {
                "num_hypotheses": len(ctx.deps.context_memory.hypotheses),
                "num_reviews": sum(len(reviews) for reviews in ctx.deps.context_memory.reviews.values()),
                "num_tournament_matches": len(ctx.deps.context_memory.tournament_results),
                "top_hypotheses": [
                    {
                        "id": h["id"],
                        "title": h.get("title", "Untitled"),
                        "elo_rating": h.get("elo_rating", 1200)
                    }
                    for h in ctx.deps.context_memory.get_top_hypotheses(5)
                ]
            }
            ctx.deps.context_memory.update_statistics(stats)
            return stats
    
    async def run(
        self, 
        research_goal: str, 
        context_memory: ContextMemory = None
    ) -> ContextMemory:
        """
        Run the Supervisor agent with the given research goal.
        
        Args:
            research_goal: The research goal to process
            context_memory: Optional existing context memory to use
            
        Returns:
            The updated context memory
        """
        if context_memory is None:
            context_memory = ContextMemory()
        
        research_plan = ResearchPlan.from_research_goal(research_goal)
        deps = AgentDependencies(
            context_memory=context_memory,
            research_plan=research_plan
        )
        
        # Run the agent to parse the research goal
        result = await self.agent.run(
            f"Parse the following research goal and create a research plan: {research_goal}",
            deps=deps
        )
        
        # Compute initial statistics
        await self.agent.run(
            "Compute statistics about the current state of the system.",
            deps=deps
        )
        
        return context_memory
