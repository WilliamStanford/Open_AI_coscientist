# AI-Coscientist Architecture Design

## Overview

This document outlines the architecture for implementing the AI-Coscientist multi-agent system using Pydantic's AI agent framework. The implementation follows the approach described in the paper "Towards an AI co-scientist" while leveraging Pydantic AI's capabilities for agent creation, communication, and orchestration.

## System Components

### 1. Core Components

#### 1.1 Supervisor Agent
- **Purpose**: Orchestrates the entire system, manages the task queue, and coordinates specialized agents
- **Implementation**: Uses Pydantic AI's Agent class with dependency injection for state management
- **Responsibilities**:
  - Parse research goals into research plan configurations
  - Assign tasks to specialized agents
  - Monitor system progress and compute statistics
  - Manage context memory for persistent state

#### 1.2 Context Memory
- **Purpose**: Provides persistent storage for system state and agent outputs
- **Implementation**: Implemented as a shared dependency injected into all agents
- **Responsibilities**:
  - Store hypotheses, reviews, tournament results
  - Track system progress and statistics
  - Enable iterative computation over long time horizons

### 2. Specialized Agents

#### 2.1 Generation Agent
- **Purpose**: Creates initial hypotheses and research proposals
- **Implementation**: Pydantic AI Agent with web search tools
- **Capabilities**:
  - Literature exploration via web search
  - Simulated scientific debates
  - Iterative assumptions identification
  - Research expansion

#### 2.2 Reflection Agent
- **Purpose**: Reviews and critiques hypotheses
- **Implementation**: Pydantic AI Agent with web search and validation tools
- **Review Types**:
  - Initial review
  - Full review with web search
  - Deep verification review
  - Observation review
  - Simulation review
  - Tournament review

#### 2.3 Ranking Agent
- **Purpose**: Evaluates and ranks hypotheses through tournaments
- **Implementation**: Pydantic AI Agent with Elo-based tournament system
- **Capabilities**:
  - Pairwise hypothesis comparison
  - Multi-turn scientific debates
  - Tournament optimization strategies

#### 2.4 Evolution Agent
- **Purpose**: Refines and improves existing hypotheses
- **Implementation**: Pydantic AI Agent with access to existing hypotheses
- **Improvement Approaches**:
  - Enhancement through grounding
  - Coherence and feasibility improvements
  - Inspiration from existing hypotheses
  - Combination of top hypotheses
  - Simplification
  - Out-of-box thinking

#### 2.5 Proximity Agent
- **Purpose**: Calculates similarity between hypotheses
- **Implementation**: Pydantic AI Agent with vector embedding capabilities
- **Responsibilities**:
  - Build proximity graph of hypotheses
  - Assist Ranking Agent in organizing tournaments
  - Help showcase diverse ideas

#### 2.6 Meta-review Agent
- **Purpose**: Synthesizes insights from reviews and debates
- **Implementation**: Pydantic AI Agent with access to all reviews and tournament results
- **Responsibilities**:
  - Generate meta-review critiques
  - Provide feedback to other agents
  - Generate research overviews

## System Architecture

### 1. Multi-Agent Workflow

The system will use Pydantic AI's multi-agent capabilities, specifically:

1. **Agent Delegation**: Agents will delegate specific tasks to other agents
2. **Programmatic Agent Hand-off**: Control will flow between agents based on the state of the system
3. **Graph-based Control Flow**: For complex interactions, especially in the tournament framework

### 2. Communication Patterns

1. **Dependency Injection**: Shared state and context through dependency injection
2. **Task Queue**: Asynchronous task execution managed by the Supervisor
3. **Event-based Communication**: Agents respond to system events and state changes

### 3. Tournament Framework

1. **Elo-based Ranking System**: Initial Elo rating of 1200 for new hypotheses
2. **Pairwise Comparisons**: Hypotheses compared through scientific debates
3. **Optimization Strategies**: Focus on similar hypotheses and prioritize newer/top-ranking ones

### 4. Web Search Integration

1. **Literature Exploration**: Using Pydantic AI's search tools (DuckDuckGo, Tavily)
2. **Grounding**: Ensuring hypotheses are grounded in existing literature
3. **Verification**: Checking novelty and correctness against published research

## Implementation Approach

### 1. Base Classes

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pydantic_ai import Agent, RunContext
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

@dataclass
class ContextMemory:
    """Shared memory for all agents in the system"""
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    reviews: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    tournament_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentDependencies:
    """Dependencies injected into all agents"""
    context_memory: ContextMemory
    research_goal: str
    research_plan: Dict[str, Any]
```

### 2. Supervisor Implementation

```python
supervisor_agent = Agent(
    'openai:gpt-4o',  # Or other model as needed
    deps_type=AgentDependencies,
    system_prompt=(
        'You are the Supervisor agent for the AI co-scientist system. '
        'Your role is to orchestrate the specialized agents, manage the task queue, '
        'and ensure the system progresses towards generating high-quality '
        'research hypotheses and proposals.'
    ),
)

@supervisor_agent.system_prompt
async def add_research_context(ctx: RunContext[AgentDependencies]) -> str:
    """Add research context to the system prompt"""
    return f"Current research goal: {ctx.deps.research_goal}\n" + \
           f"Research plan: {json.dumps(ctx.deps.research_plan, indent=2)}"
```

### 3. Specialized Agent Implementation (Example: Generation Agent)

```python
generation_agent = Agent(
    'openai:gpt-4o',  # Or other model as needed
    deps_type=AgentDependencies,
    tools=[duckduckgo_search_tool()],
    system_prompt=(
        'You are the Generation agent for the AI co-scientist system. '
        'Your role is to create novel research hypotheses and proposals '
        'based on the research goal and existing literature. '
        'Use web search to ground your hypotheses in existing knowledge.'
    ),
)

@generation_agent.tool
async def literature_exploration(
    ctx: RunContext[AgentDependencies], 
    query: str
) -> str:
    """Search for and summarize relevant literature"""
    # Implementation using web search tools
    pass

@generation_agent.tool
async def simulated_debate(
    ctx: RunContext[AgentDependencies],
    hypothesis: str
) -> str:
    """Conduct a simulated scientific debate to refine a hypothesis"""
    # Implementation of self-critique and self-play techniques
    pass
```

## Next Steps

1. **Define Agent Interfaces**: Complete the interface definitions for all specialized agents
2. **Implement Tournament Framework**: Create the Elo-based tournament system
3. **Design State Management**: Finalize the context memory structure and state transitions
4. **Implement Skeleton Code**: Create the basic implementation of all components
5. **Add Web Search Integration**: Implement literature exploration capabilities
