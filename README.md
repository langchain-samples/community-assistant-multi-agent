# Community Assistant: multi-agent reference

This is a small community assistant built with Deep Agents. The main agent (the orchestrator) hands work off to four subagents, and each one is built differently:

- `knowledge` is a LangGraph workflow that runs in the same process
- `forum_post` is a regular config-defined subagent with a prompt and tools, plus an approval step
- `support_case` is a LangGraph agent running on its own server, called with `RemoteGraph`
- `customer_success` is called over A2A, so it could be built with anything

Safety checks, the clarifying question for vague requests, and tool retries are all middleware.
Everything external (knowledge base, forum, support cases, training catalog) is mocked.

## Architecture

![Architecture](docs/img/architecture.svg)

Each run carries the user’s roles and entitlements. This context is passed to
every subagent, including the remote ones, and each one applies its own access rules.

## Choosing how to connect a subagent

| How it's connected | Example | Where it runs | Good fit when |
|---|---|---|---|
| Config-defined subagent | `forum_post` | in the orchestrator's process | your team owns it, and the model should decide the steps |
| LangGraph workflow | `knowledge` | in the orchestrator's process | your team owns it, and the steps are fixed and must always run |
| `RemoteGraph` | `support_case` | another team's Agent Server | another team owns it, and they also use LangGraph |
| A2A | `customer_success` | anywhere, on any stack | another team or vendor owns it, and you can't count on LangGraph |

The first three keep threads and approvals across the boundary. A2A is one message in and
one reply out.

## Running it

You'll need Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env     # add your model provider key
make dev-all             # orchestrator :2024, Support Case :2025, Customer Success :2026
```

The model defaults to Claude. To use another one, set `COMMUNITY_MODEL` to an
`init_chat_model` string, and set `COMMUNITY_MODEL_BASE_URL` if you go through a gateway.
The other settings are in `.env.example`. The scenarios have only been tested with Claude,
so re-run the notebook after switching models.

| Command | What it does |
|---|---|
| `make notebook` | re-runs `demo.ipynb` |

## Scenarios

`demo.ipynb` walks through each of these:

1. A knowledge question, answered with citations from articles the user can see
2. A vague question ("Tell me about payroll"), which gets a clarifying question first
3. An unsafe request, refused by the safety middleware
4. A forum post, which the user approves before it's posted
5. Opening a support case: the remote agent asks for approval, and a user without the
   right role is turned down
6. Training for the products the user owns, over A2A
7. One question that needs three subagents, which shows up as a single trace
8. A search that fails once and recovers on retry, and one that keeps failing and returns
   a friendly error message

## Deploying

Each `langgraph.*.json` file is its own LangSmith Deployment. On the orchestrator, set
`SUPPORT_CASE_URL`, `CUSTOMER_SUCCESS_URL`, `LANGSMITH_API_KEY`
and `CUSTOMER_SUCCESS_API_KEY`.
