"""
Proximity agent for the AI-Coscientist system.
"""
from pydantic_ai import Agent, RunContext
from typing import Dict, Any, List, Optional, Tuple
import json

from .base import AgentDependencies

class ProximityAgent:
    """
    Proximity agent that calculates similarity between hypotheses.
    
    The Proximity agent is responsible for:
    1. Building a proximity graph of hypotheses
    2. Assisting the Ranking agent in organizing tournaments
    3. Helping showcase diverse ideas
    """
    
    def __init__(self, model_name: str = "google-gla:gemini-2.0-flash"):
        """Initialize the Proximity agent."""
        self.agent = Agent(
            model_name,
            deps_type=AgentDependencies,
            system_prompt=(
                "You are the Proximity agent for the AI co-scientist system. "
                "Your role is to calculate the similarity between research hypotheses "
                "and build a proximity graph. This helps organize tournament matches "
                "and showcase a diverse range of ideas."
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
        async def calculate_similarity(
            ctx: RunContext[AgentDependencies], 
            hypothesis_id_1: str,
            hypothesis_id_2: str
        ) -> Dict[str, Any]:
            """Calculate the similarity between two hypotheses."""
            hypothesis_1 = ctx.deps.context_memory.get_hypothesis_by_id(hypothesis_id_1)
            hypothesis_2 = ctx.deps.context_memory.get_hypothesis_by_id(hypothesis_id_2)
            
            if not hypothesis_1 or not hypothesis_2:
                return {"error": "One or both hypotheses not found"}
            
            # In a real implementation, this would use the agent to calculate similarity
            # based on semantic understanding of the hypotheses
            similarity = {
                "hypothesis_1": hypothesis_id_1,
                "hypothesis_2": hypothesis_id_2,
                "similarity_score": 0.5,  # Placeholder value
                "common_themes": ["Theme 1", "Theme 2"],
                "differences": ["Difference 1", "Difference 2"],
                "created_at": "2025-04-14",  # In a real implementation, use actual timestamp
            }
            
            return similarity
        
        @self.agent.tool
        async def build_proximity_graph(
            ctx: RunContext[AgentDependencies]
        ) -> Dict[str, Any]:
            """Build a proximity graph of all hypotheses."""
            hypotheses = ctx.deps.context_memory.hypotheses
            if len(hypotheses) < 2:
                return {"error": "Not enough hypotheses to build a graph"}
            
            # In a real implementation, this would calculate similarity between all pairs
            # of hypotheses and build a graph
            edges = []
            for i in range(len(hypotheses)):
                for j in range(i+1, len(hypotheses)):
                    h1 = hypotheses[i]
                    h2 = hypotheses[j]
                    # Placeholder similarity calculation
                    similarity = 0.5
                    edges.append({
                        "source": h1["id"],
                        "target": h2["id"],
                        "weight": similarity
                    })
            
            graph = {
                "nodes": [{"id": h["id"], "title": h.get("title", "Untitled")} for h in hypotheses],
                "edges": edges,
                "created_at": "2025-04-14",  # In a real implementation, use actual timestamp
            }
            
            return graph
        
        @self.agent.tool
        async def find_similar_hypotheses(
            ctx: RunContext[AgentDependencies],
            hypothesis_id: str,
            num_similar: int = 3
        ) -> Dict[str, Any]:
            """Find the most similar hypotheses to a given one."""
            hypothesis = ctx.deps.context_memory.get_hypothesis_by_id(hypothesis_id)
            if not hypothesis:
                return {"error": f"Hypothesis with ID {hypothesis_id} not found"}
            
            hypotheses = ctx.deps.context_memory.hypotheses
            if len(hypotheses) < 2:
                return {"error": "Not enough hypotheses to find similar ones"}
            
            # In a real implementation, this would calculate similarity between the given
            # hypothesis and all others, then return the most similar ones
            similar_hypotheses = []
            for h in hypotheses:
                if h["id"] != hypothesis_id:
                    # Placeholder similarity calculation
                    similarity = 0.5
                    similar_hypotheses.append({
                        "id": h["id"],
                        "title": h.get("title", "Untitled"),
                        "similarity_score": similarity
                    })
            
            # Sort by similarity score and take the top N
            similar_hypotheses.sort(key=lambda x: x["similarity_score"], reverse=True)
            similar_hypotheses = similar_hypotheses[:num_similar]
            
            return {
                "hypothesis_id": hypothesis_id,
                "similar_hypotheses": similar_hypotheses,
                "created_at": "2025-04-14",  # In a real implementation, use actual timestamp
            }
    
    async def build_graph(self, deps: AgentDependencies) -> Dict[str, Any]:
        """
        Build a proximity graph of all hypotheses.
        
        Args:
            deps: The agent dependencies
            
        Returns:
            The proximity graph
        """
        result = await self.agent.run(
            "Build a proximity graph of all hypotheses",
            deps=deps
        )
        
        return result.data
    
    async def find_similar(
        self, 
        deps: AgentDependencies, 
        hypothesis_id: str,
        num_similar: int = 3
    ) -> Dict[str, Any]:
        """
        Find the most similar hypotheses to a given one.
        
        Args:
            deps: The agent dependencies
            hypothesis_id: The ID of the hypothesis to find similar ones for
            num_similar: The number of similar hypotheses to find
            
        Returns:
            The similar hypotheses
        """
        result = await self.agent.run(
            f"Find {num_similar} hypotheses similar to {hypothesis_id}",
            deps=deps
        )
        
        return result.data
