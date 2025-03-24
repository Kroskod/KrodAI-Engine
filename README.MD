# KROD - Knowledge-Reinforced Operational Developer

Krod is an AI research assistant specializing in coding, mathematics, and scientific research. It provides structured, well-reasoned responses while maintaining transparency about its capabilities and limitations.

## Features

- Code analysis and optimization
- Mathematical problem-solving
- Research assistance and literature analysis
- Reasoning capabilities
- Context-aware responses
- Security validation
- Session management

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
ANTHROPIC_API_KEY=your_api_key_here
KROD_CONFIG=./config/krod.yaml
KROD_LOG_LEVEL=INFO
KROD_DEBUG=False
```

## License

MIT License

Copyright (c) 2025 Sarthak Sharma

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