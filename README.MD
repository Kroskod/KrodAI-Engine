# Krod AI : The Research Partner Built for Your Breakthroughs

Krod AI is your partner in research, development, and engineering—a new class of intelligent AI designed to amplify your ability to make breakthroughs in science, engineering, and academia.


## What Makes Krod AI Special?

Krod AI is built on the principle of **Modular Specialization with centralized orchestration**. This means that Krod AI is not a monolithic system, but a collection of specialized modules that work together to provide a unified experience. Each module is designed to perform a specific task, and the system as a whole is designed to coordinate the actions of these modules to provide a seamless experience.

## Vision

Krod is a **Research Amplification AI**—not a replacement for human ingenuity, but a powerful partner that helps you achieve more. It provides not only answers, but also alternative pathways, debate, and iterative refinement, helping you explore, discover, and document knowledge.

## Key Features

- **Literature Discovery & Knowledge Management**
- **Session Persistence**: Continue your research across days, weeks, or months.
- **Web Crawl & Indexing**: Bring in new knowledge on demand.
- **Adaptive Learning**: Learns from your sessions and adapts to your style.
- **Documentation & Transparency**: Every step and decision is tracked.
- **Agentic Reasoning & Hypothesis Generation**
- **Debate & Iterative Refinement**: Challenge your ideas and refine them.
- **Solution Branching**: Get alternative solutions, methods, or resources for every problem.


## Core Systems

- **Session Management**: Persistent, context-aware sessions.
- **Security Validation**: Keeps your work safe and compliant.
- **Reasoning System**: Chain-of-thought and explicit reasoning.
- **Clarification System**: Seeks clarity before acting.
- **Decision Making System**: Smart routing of queries to specialized modules.
- **Common Sense System**: Adds human-like judgment.
- **Knowledge Graphs**: Structured, queryable knowledge.
- **RAG & Vector Database**: State-of-the-art retrieval-augmented generation.


## Installation

```bash
# Clone the repository
git clone https://github.com/Kroskod/KrodAI-Engine.git
cd krod

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install KROD
pip install -e .
```

## Usage

```bash
# Start KROD CLI
krod

# Process a single query
krod --query "Your query here"

# Enable debug mode
krod --debug

# Use specific configuration
krod --config path/to/config.yaml
```

## Configuration

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_api_key_here # or OPENAI_API_KEY=your_api_key_here
KROD_CONFIG=./config/krod.yaml
KROD_LOG_LEVEL=INFO
KROD_DEBUG=False
```

## Uses 
<a href="https://github.com/unclecode/crawl4ai">
  <img src="https://raw.githubusercontent.com/unclecode/crawl4ai/main/docs/assets/powered-by-disco.svg" alt="Powered by Crawl4AI" width="200"/>
</a>



## License

MIT License

Copyright (c) 2025 Sarthak Sharma and Kroskod Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE