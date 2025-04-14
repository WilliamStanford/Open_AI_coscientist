"""
README for the AI-Coscientist implementation.
"""

# AI-Coscientist Implementation

This project implements a multi-agent system that mimics the approach described in the paper "Towards an AI co-scientist" using Pydantic's AI agent framework.

## Overview

The AI-Coscientist is a multi-agent system designed to help scientists generate novel research hypotheses and proposals. The system employs a "generate, debate, and evolve" approach to hypothesis generation, inspired by the scientific method.

## Architecture

The system consists of the following specialized agents:

1. **Supervisor Agent**: Orchestrates the entire system, manages the task queue, and coordinates specialized agents.

2. **Generation Agent**: Creates initial hypotheses through literature exploration and simulated debates.

3. **Reflection Agent**: Reviews and critiques hypotheses with various review types (initial, full, deep verification, etc.).

4. **Ranking Agent**: Evaluates hypotheses through Elo-based tournaments and scientific debates.

5. **Evolution Agent**: Refines and improves existing hypotheses through various approaches.

6. **Proximity Agent**: Calculates similarity between hypotheses to build a proximity graph.

7. **Meta-review Agent**: Synthesizes insights from reviews and debates to provide feedback and generate research overviews.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


Export gemini api key 
```bash
export GEMINI_API_KEY=your-api-key
```

## Usage

### As a Python Package

```python
import asyncio
from open_ai_coscientist import run_ai_coscientist

async def main():
    research_goal = "Propose a novel hypothesis about the role of mitochondrial dysfunction in neurodegenerative diseases."
    overview = await run_ai_coscientist(research_goal)
    print(overview)

if __name__ == "__main__":
    asyncio.run(main())
```

### As a Command-Line Tool

```bash
open_ai_coscientist "Propose a novel hypothesis about the role of mitochondrial dysfunction in neurodegenerative diseases."
```

## Testing

To run the test script:

```bash
python test_coscientist.py
```

## Requirements

- Python 3.8+
- pydantic-ai>=0.0.55
- pydantic>=2.0.0
- httpx>=0.24.0

## Implementation Details

The implementation follows the architecture described in the paper, with a few adaptations to leverage Pydantic AI's capabilities:

1. **Multi-Agent Workflow**: Uses Pydantic AI's agent delegation and programmatic hand-off for agent communication.

2. **Tournament Framework**: Implements an Elo-based tournament system for hypothesis ranking.

3. **Web Search Integration**: Uses DuckDuckGo search for literature exploration and grounding.

4. **Context Memory**: Provides persistent storage for system state and agent outputs.

## Limitations

This is a skeleton implementation that demonstrates the architecture and approach. In a production system, you would need to:

1. Implement more sophisticated natural language processing for hypothesis generation and evaluation.
2. Add more robust error handling and recovery mechanisms.
3. Optimize the tournament selection strategy based on the proximity graph.
4. Implement more sophisticated web search and literature analysis capabilities.

