"""
Setup script for the AI-Coscientist package.
"""
from setuptools import setup, find_packages

setup(
    name="ai_coscientist",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic-ai>=0.0.55",
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
    ],
    entry_points={
        "console_scripts": [
            "ai-coscientist=src.main:main",
        ],
    },
    description="An implementation of the AI-Coscientist multi-agent system using Pydantic AI",
    author="Manus AI",
    author_email="info@example.com",
    python_requires=">=3.8",
)
