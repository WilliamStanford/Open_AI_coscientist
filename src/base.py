"""
Base models and shared components for the AI-Coscientist system.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import json

@dataclass
class ContextMemory:
    """Shared memory for all agents in the system."""
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    reviews: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    tournament_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def add_hypothesis(self, hypothesis: Dict[str, Any]) -> str:
        """Add a hypothesis to the context memory and return its ID."""
        hypothesis_id = f"hyp_{len(self.hypotheses)}"
        hypothesis["id"] = hypothesis_id
        hypothesis["elo_rating"] = 1200  # Initial Elo rating
        self.hypotheses.append(hypothesis)
        self.reviews[hypothesis_id] = []
        return hypothesis_id
    
    def add_review(self, hypothesis_id: str, review: Dict[str, Any]) -> None:
        """Add a review for a hypothesis."""
        if hypothesis_id not in self.reviews:
            self.reviews[hypothesis_id] = []
        self.reviews[hypothesis_id].append(review)
    
    def add_tournament_result(self, match_id: str, result: Dict[str, Any]) -> None:
        """Add a tournament match result."""
        self.tournament_results[match_id] = result
        
    def update_statistics(self, new_stats: Dict[str, Any]) -> None:
        """Update system statistics."""
        self.statistics.update(new_stats)
    
    def get_top_hypotheses(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the top N hypotheses by Elo rating."""
        sorted_hypotheses = sorted(
            self.hypotheses, 
            key=lambda h: h.get("elo_rating", 0), 
            reverse=True
        )
        return sorted_hypotheses[:n]
    
    def get_hypothesis_by_id(self, hypothesis_id: str) -> Optional[Dict[str, Any]]:
        """Get a hypothesis by its ID."""
        for hypothesis in self.hypotheses:
            if hypothesis.get("id") == hypothesis_id:
                return hypothesis
        return None

@dataclass
class ResearchPlan:
    """Research plan configuration."""
    goal: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_research_goal(cls, research_goal: str) -> "ResearchPlan":
        """Parse a research goal into a research plan configuration."""
        # In a real implementation, this would use an LLM to parse the goal
        return cls(
            goal=research_goal,
            attributes={
                "novelty_required": True,
                "experimental_validation_required": True,
            },
            constraints={
                "domain_specific_constraints": [],
            },
            evaluation_criteria={
                "novelty": 0.3,
                "correctness": 0.3,
                "testability": 0.2,
                "relevance": 0.2,
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the research plan to a dictionary."""
        return {
            "goal": self.goal,
            "attributes": self.attributes,
            "constraints": self.constraints,
            "evaluation_criteria": self.evaluation_criteria,
        }

@dataclass
class AgentDependencies:
    """Dependencies injected into all agents."""
    context_memory: ContextMemory
    research_plan: ResearchPlan
