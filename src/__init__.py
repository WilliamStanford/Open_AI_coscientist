"""
Package initialization file for the AI-Coscientist system.
"""
from .base import ContextMemory, ResearchPlan, AgentDependencies
from .supervisor import SupervisorAgent
from .generation import GenerationAgent
from .reflection import ReflectionAgent
from .ranking import RankingAgent
from .evolution import EvolutionAgent
from .proximity import ProximityAgent
from .meta_review import MetaReviewAgent
from .main import AICoscientist, run_ai_coscientist

__all__ = [
    'ContextMemory',
    'ResearchPlan',
    'AgentDependencies',
    'SupervisorAgent',
    'GenerationAgent',
    'ReflectionAgent',
    'RankingAgent',
    'EvolutionAgent',
    'ProximityAgent',
    'MetaReviewAgent',
    'AICoscientist',
    'run_ai_coscientist',
]
