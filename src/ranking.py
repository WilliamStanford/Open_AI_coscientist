"""
Ranking agent for the AI-Coscientist system.
"""
from pydantic_ai import Agent, RunContext
from typing import Dict, Any, List, Optional, Tuple
import json
import math

from .base import AgentDependencies

class RankingAgent:
    """
    Ranking agent that evaluates and ranks hypotheses through tournaments.
    
    The Ranking agent is responsible for:
    1. Pairwise hypothesis comparison
    2. Multi-turn scientific debates
    3. Tournament optimization strategies
    4. Elo-based ranking system
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """Initialize the Ranking agent."""
        self.agent = Agent(
            model_name,
            deps_type=AgentDependencies,
            system_prompt=(
                "You are the Ranking agent for the AI co-scientist system. "
                "Your role is to compare and rank research hypotheses through "
                "scientific debates and tournaments. You should evaluate hypotheses "
                "based on their novelty, correctness, and testability."
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
        async def compare_hypotheses(
            ctx: RunContext[AgentDependencies], 
            hypothesis_id_1: str,
            hypothesis_id_2: str
        ) -> Dict[str, Any]:
            """Compare two hypotheses and determine which is better."""
            hypothesis_1 = ctx.deps.context_memory.get_hypothesis_by_id(hypothesis_id_1)
            hypothesis_2 = ctx.deps.context_memory.get_hypothesis_by_id(hypothesis_id_2)
            
            if not hypothesis_1 or not hypothesis_2:
                return {"error": "One or both hypotheses not found"}
            
            # In a real implementation, this would use the agent to conduct a debate
            # and determine the winner
            comparison = {
                "match_id": f"{hypothesis_id_1}_vs_{hypothesis_id_2}",
                "hypothesis_1": hypothesis_id_1,
                "hypothesis_2": hypothesis_id_2,
                "winner": hypothesis_id_1 if hash(hypothesis_id_1) > hash(hypothesis_id_2) else hypothesis_id_2,
                "debate_summary": f"Comparison between {hypothesis_1.get('title', 'Untitled')} and {hypothesis_2.get('title', 'Untitled')}",
                "created_at": "2025-04-14",  # In a real implementation, use actual timestamp
            }
            
            # Update Elo ratings
            self._update_elo_ratings(ctx, comparison)
            
            ctx.deps.context_memory.add_tournament_result(comparison["match_id"], comparison)
            return comparison
        
        @self.agent.tool
        async def scientific_debate(
            ctx: RunContext[AgentDependencies], 
            hypothesis_id_1: str,
            hypothesis_id_2: str
        ) -> Dict[str, Any]:
            """Conduct a multi-turn scientific debate between two hypotheses."""
            hypothesis_1 = ctx.deps.context_memory.get_hypothesis_by_id(hypothesis_id_1)
            hypothesis_2 = ctx.deps.context_memory.get_hypothesis_by_id(hypothesis_id_2)
            
            if not hypothesis_1 or not hypothesis_2:
                return {"error": "One or both hypotheses not found"}
            
            # In a real implementation, this would use the agent to conduct a multi-turn debate
            debate = {
                "match_id": f"{hypothesis_id_1}_debate_{hypothesis_id_2}",
                "hypothesis_1": hypothesis_id_1,
                "hypothesis_2": hypothesis_id_2,
                "turns": [
                    {"speaker": "hypothesis_1", "text": "Argument for hypothesis 1"},
                    {"speaker": "hypothesis_2", "text": "Argument for hypothesis 2"},
                    {"speaker": "hypothesis_1", "text": "Rebuttal from hypothesis 1"},
                    {"speaker": "hypothesis_2", "text": "Rebuttal from hypothesis 2"},
                ],
                "winner": hypothesis_id_1 if hash(hypothesis_id_1) > hash(hypothesis_id_2) else hypothesis_id_2,
                "debate_summary": f"Scientific debate between {hypothesis_1.get('title', 'Untitled')} and {hypothesis_2.get('title', 'Untitled')}",
                "created_at": "2025-04-14",  # In a real implementation, use actual timestamp
            }
            
            # Update Elo ratings
            self._update_elo_ratings(ctx, debate)
            
            ctx.deps.context_memory.add_tournament_result(debate["match_id"], debate)
            return debate
    
    def _update_elo_ratings(self, ctx: RunContext[AgentDependencies], match_result: Dict[str, Any]) -> None:
        """Update Elo ratings based on a match result."""
        hypothesis_1_id = match_result["hypothesis_1"]
        hypothesis_2_id = match_result["hypothesis_2"]
        winner_id = match_result["winner"]
        
        hypothesis_1 = ctx.deps.context_memory.get_hypothesis_by_id(hypothesis_1_id)
        hypothesis_2 = ctx.deps.context_memory.get_hypothesis_by_id(hypothesis_2_id)
        
        if not hypothesis_1 or not hypothesis_2:
            return
        
        # Get current Elo ratings
        elo_1 = hypothesis_1.get("elo_rating", 1200)
        elo_2 = hypothesis_2.get("elo_rating", 1200)
        
        # Calculate expected scores
        expected_1 = 1 / (1 + 10 ** ((elo_2 - elo_1) / 400))
        expected_2 = 1 / (1 + 10 ** ((elo_1 - elo_2) / 400))
        
        # Calculate actual scores
        actual_1 = 1.0 if winner_id == hypothesis_1_id else 0.0
        actual_2 = 1.0 if winner_id == hypothesis_2_id else 0.0
        
        # Update Elo ratings (K-factor of 32)
        k_factor = 32
        new_elo_1 = elo_1 + k_factor * (actual_1 - expected_1)
        new_elo_2 = elo_2 + k_factor * (actual_2 - expected_2)
        
        # Update hypotheses in context memory
        hypothesis_1["elo_rating"] = new_elo_1
        hypothesis_2["elo_rating"] = new_elo_2
    
    async def run_tournament(
        self, 
        deps: AgentDependencies, 
        num_matches: int = 10
    ) -> Dict[str, Any]:
        """
        Run a tournament to rank hypotheses.
        
        Args:
            deps: The agent dependencies
            num_matches: The number of tournament matches to run
            
        Returns:
            Tournament results summary
        """
        hypotheses = deps.context_memory.hypotheses
        if len(hypotheses) < 2:
            return {"error": "Not enough hypotheses for a tournament"}
        
        # Run tournament matches
        for _ in range(min(num_matches, len(hypotheses) * (len(hypotheses) - 1) // 2)):
            # Select two hypotheses to compare
            # In a real implementation, this would use a more sophisticated selection strategy
            # based on proximity and prioritizing newer/top-ranking hypotheses
            hypothesis_1, hypothesis_2 = self._select_hypotheses_for_match(deps)
            
            # Run a scientific debate for top-ranked hypotheses
            if (hypothesis_1.get("elo_rating", 1200) > 1300 and 
                hypothesis_2.get("elo_rating", 1200) > 1300):
                await self.agent.run(
                    f"Conduct a scientific debate between hypothesis {hypothesis_1['id']} and {hypothesis_2['id']}",
                    deps=deps
                )
            else:
                # Run a simple comparison for lower-ranked hypotheses
                await self.agent.run(
                    f"Compare hypothesis {hypothesis_1['id']} and {hypothesis_2['id']}",
                    deps=deps
                )
        
        # Return tournament summary
        return {
            "num_matches": num_matches,
            "top_hypotheses": [
                {
                    "id": h["id"],
                    "title": h.get("title", "Untitled"),
                    "elo_rating": h.get("elo_rating", 1200)
                }
                for h in deps.context_memory.get_top_hypotheses(5)
            ]
        }
    
    def _select_hypotheses_for_match(
        self, 
        deps: AgentDependencies
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Select two hypotheses for a tournament match."""
        # In a real implementation, this would use a more sophisticated selection strategy
        # based on proximity and prioritizing newer/top-ranking hypotheses
        hypotheses = deps.context_memory.hypotheses
        import random
        idx1, idx2 = random.sample(range(len(hypotheses)), 2)
        return hypotheses[idx1], hypotheses[idx2]
