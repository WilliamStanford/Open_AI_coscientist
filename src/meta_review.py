"""
Meta-review agent for the AI-Coscientist system.
"""
from pydantic_ai import Agent, RunContext
from typing import Dict, Any, List, Optional
import json

from .base import AgentDependencies

class MetaReviewAgent:
    """
    Meta-review agent that synthesizes insights from reviews and debates.
    
    The Meta-review agent is responsible for:
    1. Generating meta-review critiques
    2. Providing feedback to other agents
    3. Generating research overviews
    """
    
    def __init__(self, model_name: str = "google-gla:gemini-2.0-flash"):
        """Initialize the Meta-review agent."""
        self.agent = Agent(
            model_name,
            deps_type=AgentDependencies,
            system_prompt=(
                "You are the Meta-review agent for the AI co-scientist system. "
                "Your role is to synthesize insights from reviews and debates, "
                "provide feedback to other agents, and generate research overviews "
                "that summarize the top-ranked hypotheses."
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
        async def generate_meta_review(
            ctx: RunContext[AgentDependencies]
        ) -> Dict[str, Any]:
            """Generate a meta-review critique based on all reviews and tournament results."""
            reviews = ctx.deps.context_memory.reviews
            tournament_results = ctx.deps.context_memory.tournament_results
            
            if not reviews or not tournament_results:
                return {"error": "Not enough reviews or tournament results for meta-review"}
            
            # In a real implementation, this would analyze patterns in reviews and tournament results
            meta_review = {
                "common_strengths": [
                    "Strength 1: Many hypotheses demonstrate good grounding in literature",
                    "Strength 2: Most hypotheses propose testable experiments"
                ],
                "common_weaknesses": [
                    "Weakness 1: Some hypotheses lack sufficient novelty",
                    "Weakness 2: Experimental protocols often lack detail"
                ],
                "improvement_suggestions": [
                    "Suggestion 1: Focus more on cross-disciplinary connections",
                    "Suggestion 2: Provide more detailed experimental protocols"
                ],
                "created_at": "2025-04-14",  # In a real implementation, use actual timestamp
            }
            
            return meta_review
        
        @self.agent.tool
        async def generate_research_overview(
            ctx: RunContext[AgentDependencies],
            num_hypotheses: int = 5
        ) -> Dict[str, Any]:
            """Generate a research overview based on top-ranked hypotheses."""
            top_hypotheses = ctx.deps.context_memory.get_top_hypotheses(num_hypotheses)
            
            if not top_hypotheses:
                return {"error": "No hypotheses available for research overview"}
            
            # In a real implementation, this would synthesize the top hypotheses into a coherent overview
            overview = {
                "title": f"Research Overview for: {ctx.deps.research_plan.goal}",
                "summary": "This overview summarizes the top-ranked research hypotheses and proposals.",
                "key_areas": [
                    "Area 1: Description of first research direction",
                    "Area 2: Description of second research direction"
                ],
                "top_hypotheses_summary": [
                    {
                        "id": h["id"],
                        "title": h.get("title", "Untitled"),
                        "summary": f"Brief summary of {h.get('title', 'Untitled')}"
                    }
                    for h in top_hypotheses
                ],
                "future_directions": [
                    "Direction 1: Suggestion for future research",
                    "Direction 2: Another suggestion for future research"
                ],
                "created_at": "2025-04-14",  # In a real implementation, use actual timestamp
            }
            
            return overview
    
    async def generate_meta_review_critique(self, deps: AgentDependencies) -> Dict[str, Any]:
        """
        Generate a meta-review critique based on all reviews and tournament results.
        
        Args:
            deps: The agent dependencies
            
        Returns:
            The meta-review critique
        """
        result = await self.agent.run(
            "Generate a meta-review critique based on all reviews and tournament results",
            deps=deps
        )
        
        return result.data
    
    async def generate_overview(
        self, 
        deps: AgentDependencies,
        num_hypotheses: int = 5
    ) -> Dict[str, Any]:
        """
        Generate a research overview based on top-ranked hypotheses.
        
        Args:
            deps: The agent dependencies
            num_hypotheses: The number of top hypotheses to include
            
        Returns:
            The research overview
        """
        result = await self.agent.run(
            f"Generate a research overview based on the top {num_hypotheses} hypotheses",
            deps=deps
        )
        
        return result.data
