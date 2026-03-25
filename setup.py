"""Setup script for LLM Server."""

from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="llm-server",
    version="0.1.0",
    author="LLM Server Team",
    description="High-performance LLM inference server",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/llm-server",
    packages=find_packages(exclude=["tests", "tests.*", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "pyyaml>=6.0.1",
        "loguru>=0.7.2",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-server=src.main:main",
        ],
    },
)