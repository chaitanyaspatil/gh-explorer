# gh-explorer

An agentic CLI that answers free-form questions about any GitHub repository.

Given a question like *"Is this project healthy?"* or *"What's a good first
issue to tackle?"*, the agent decides which GitHub data sources to consult
(README, recent commits, releases, individual issues, label taxonomy),
reads what they return, adapts based on what it observes, and produces
an answer with linked source references.

## Example questions

- *"I'm an open-source developer with 3 years of experience. What's a good first issue?"*
- *"Is this project healthy?"*
- *"Are there broader recurring problems across multiple issues?"*

## How to run

Requires Python 3.10+.

```bash
pip install -r requirements.txt
cp .env.example .env
# Then fill in:
#   ANTHROPIC_API_KEY  (required)
#   GITHUB_TOKEN       (strongly recommended; otherwise hits the 60 req/hr cap)
#   LANGSMITH_*        (optional; auto-traces every run)

python agent.py "Is this project healthy?"
