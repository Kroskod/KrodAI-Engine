"""
KROD Setup Configuration

Package configuration and installation settings.

"""

from setuptools import setup, find_packages
import os

# read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# read the requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="krodai",
    version="0.1.0",
    author="Sarthak Sharma",
    author_email="sarthak@kroskod.com",
    description="KrodAI - Knowledge Amplification Partner",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kroskod/KrodAI-Engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai",
        "python-dotenv",
        "pyyaml",
        "cmd2",
        "requests",
        "numpy",
        "rich",
        "prompt_toolkit",
        "networkx",
        "pandas",
        "fastapi",
        "uvicorn",
        "pydantic",
        "anthropic",
        "pytest",
        "black",
        "flake8",
        "mypy",
        "click",
        "click-testing",
        "shlex",
        "aiohttp",
        "beautifulsoup4",
        "PyMuPDF",
        "crawl4ai",
        "trafilatura",
        "arxiv",
        "scholarly",
        "habanero",
        "sentence-transformers",
        "qdrant-client",
    ],
    entry_points={
        "console_scripts": [
            "krod=krod.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "krod": [
            "config/*.yaml",
            "data/knowledge/*.json",
            "data/sessions/*.json",
        ],
    },
)
