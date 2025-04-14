"""
Main entry point for the AI-Coscientist system.
"""
import asyncio
from typing import Dict, Any, List, Optional

from .base import ContextMemory, ResearchPlan, AgentDependencies
from .supervisor import SupervisorAgent
from .generation import GenerationAgent
from .reflection import ReflectionAgent
from .ranking import RankingAgent
from .evolution import EvolutionAgent
from .proximity import ProximityAgent
from .meta_review import MetaReviewAgent

class AICoscientist:
    """
    Main class for the AI-Coscientist system.
    
    This class integrates all specialized agents and provides a simple interface
    for running the system with a given research goal.
    """
    
    def __init__(self, model_name: str = "google-gla:gemini-2.0-flash"):
        """Initialize the AI-Coscientist system."""
        self.model_name = model_name
        self.supervisor = SupervisorAgent(model_name)
        self.generation = GenerationAgent(model_name)
        self.reflection = ReflectionAgent(model_name)
        self.ranking = RankingAgent(model_name)
        self.evolution = EvolutionAgent(model_name)
        self.proximity = ProximityAgent(model_name)
        self.meta_review = MetaReviewAgent(model_name)
        
        self.context_memory = ContextMemory()
    
    async def run(
        self, 
        research_goal: str,
        num_iterations: int = 3,
        hypotheses_per_iteration: int = 2,
        tournament_matches_per_iteration: int = 5
    ) -> Dict[str, Any]:
        """
        Run the AI-Coscientist system with the given research goal.
        
        Args:
            research_goal: The research goal to process
            num_iterations: The number of iterations to run
            hypotheses_per_iteration: The number of hypotheses to generate per iteration
            tournament_matches_per_iteration: The number of tournament matches to run per iteration
            
        Returns:
            The final research overview
        """
        # Initialize the system with the research goal
        self.context_memory = await self.supervisor.run(research_goal, self.context_memory)
        
        # Create dependencies object
        research_plan = ResearchPlan.from_research_goal(research_goal)
        deps = AgentDependencies(
            context_memory=self.context_memory,
            research_plan=research_plan
        )
        
        # Run the system for the specified number of iterations
        for iteration in range(num_iterations):
            print(f"Starting iteration {iteration + 1}/{num_iterations}")
            
            # Generate hypotheses
            for _ in range(hypotheses_per_iteration):
                await self.generation.generate_hypothesis(deps)

            print('------------HYPOTHESES GENERATED------------')
            
            for ind, hypothesis in enumerate(deps.context_memory.hypotheses):
                print(f'Hypothesis {ind}: {hypothesis}')
                print('')
            
            # Review all hypotheses
            for hypothesis in deps.context_memory.hypotheses:
                if not any(r.get("type") == "initial" for r in deps.context_memory.reviews.get(hypothesis["id"], [])):
                    await self.reflection.review_hypothesis(deps, hypothesis["id"])
            
            print('------------HYPOTHESES REVIEWED------------')

            # Run tournament
            await self.ranking.run_tournament(deps, tournament_matches_per_iteration)
            
            print('------------TOURNAMENT COMPLETED------------')

            # Evolve top hypotheses
            top_hypotheses = deps.context_memory.get_top_hypotheses(3)
            for hypothesis in top_hypotheses:
                await self.evolution.evolve_hypothesis(deps, hypothesis["id"], "enhance")
            
            print('------------HYPOTHESES EVOLVED------------')

            # Build proximity graph
            await self.proximity.build_graph(deps)
            
            print('------------PROXIMITY GRAPH BUILT------------')

            # Generate meta-review
            await self.meta_review.generate_meta_review_critique(deps)

            print('------------META REVIEW COMPLETE------------')
        
        # Generate final research overview
        overview = await self.meta_review.generate_overview(deps)
        
        return overview

async def run_ai_coscientist(
    research_goal: str,
    model_name: str = "google-gla:gemini-2.0-flash",
    num_iterations: int = 3
) -> Dict[str, Any]:
    """
    Run the AI-Coscientist system with the given research goal.
    
    Args:
        research_goal: The research goal to process
        model_name: The model to use for all agents
        num_iterations: The number of iterations to run
        
    Returns:
        The final research overview
    """
    coscientist = AICoscientist(model_name)
    return await coscientist.run(research_goal, num_iterations)

def main():
    """Main entry point for the AI-Coscientist system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the AI-Coscientist system")
    parser.add_argument("research_goal", help="The research goal to process")
    parser.add_argument("--model", default="google-gla:gemini-2.0-flash", help="The model to use for all agents")
    parser.add_argument("--iterations", type=int, default=3, help="The number of iterations to run")
    
    args = parser.parse_args()
    
    overview = asyncio.run(run_ai_coscientist(
        args.research_goal,
        args.model,
        args.iterations
    ))
    
    print(f"Research Overview: {overview['title']}")
    print(f"Summary: {overview['summary']}")
    print("Key Areas:")
    for area in overview["key_areas"]:
        print(f"- {area}")
    print("Top Hypotheses:")
    for hypothesis in overview["top_hypotheses_summary"]:
        print(f"- {hypothesis['title']}: {hypothesis['summary']}")
    print("Future Directions:")
    for direction in overview["future_directions"]:
        print(f"- {direction}")

if __name__ == "__main__":
    main()
