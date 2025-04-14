"""
Simple test script for the AI-Coscientist system.
"""
import asyncio
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import AICoscientist

async def test_ai_coscientist():
    """Test the AI-Coscientist system with a simple research goal."""
    # Create an instance of the AI-Coscientist system
    coscientist = AICoscientist()
    
    # Define a simple research goal for testing
    research_goal = "Propose a novel hypothesis about the role of mitochondrial dysfunction in neurodegenerative diseases."
    
    # Run the system with a reduced number of iterations for testing
    overview = await coscientist.run(
        research_goal=research_goal,
        num_iterations=1,
        hypotheses_per_iteration=1,
        tournament_matches_per_iteration=2
    )
    
    # Print the results
    print("\n=== AI-Coscientist Test Results ===")
    print(f"Research Goal: {research_goal}")
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
    
    return overview

if __name__ == "__main__":
    asyncio.run(test_ai_coscientist())
