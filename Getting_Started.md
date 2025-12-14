# Getting Started

## Prerequisites

- Python 3.10+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/minddev25/model_as_agents
cd model_as_agents
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up your API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Running the Demo

```bash
python enterprise_agent_demo.py
```

Try different queries:
```python
app.run("What is our expense policy?")
app.run("Show me North region sales")
app.run("I want to take leave Dec 23-27")
```

That's it! The agents will handle routing and execution automatically.
