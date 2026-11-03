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

Each run carries the user’s roles, entitlements and locale. This context is passed to
every subagent, including the remote ones, and each one applies its own access rules.

## Choosing how to connect a subagent

| | LangGraph workflow, in process | `RemoteGraph` | A2A |
|---|---|---|---|
| Owner | your team | another team | another team or vendor |
| What the subagent is built with | LangGraph, in your codebase | LangGraph, on its own Agent Server | anything that supports A2A |
| Keeps threads and interrupts | yes | yes | no, it's one message in and one reply out |
| Good fit when | you own the code and deploy it together | the other team also uses LangGraph | you can't count on LangGraph on the other side |

## Notes

- `RemoteGraph` doesn't forward the runtime context for you. `ContextForwardingRemoteGraph`
  in `front_door/remote_agents.py` sends it explicitly. It also gives each delegation its
  own remote thread, which is what lets an approval on the remote side be resumed.
- The Agent Server's A2A endpoint expects an assistant UUID, not the graph name. Set
  `CUSTOMER_SUCCESS_A2A_URL` to the full endpoint.
- The remote servers use `make_graph` from `tracing.py` so their runs show up inside the
  caller's trace in LangSmith.
- Only tools that are safe to call twice get retried. The `task` tool isn't one of them,
  because retrying it would re-run the whole subagent. `customer_success` retries its A2A
  call itself.
- User identity is passed in, not authenticated. In production you'd use Agent Server
  custom auth.

## Running it

You'll need Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env     # add your model provider key
make dev-all             # orchestrator :2024, Support Case :2025, Customer Success :2026
```

The model defaults to Claude. To use another one, set `COMMUNITY_MODEL` to an
`init_chat_model` string, and set `COMMUNITY_MODEL_BASE_URL` if you go through a gateway.
The other settings are in `.env.example`.

| Command | What it does |
|---|---|
| `make test` | unit tests (no model or servers needed) |
| `make smoke` | runs the scenarios below against the running servers |
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
   a localized error

## Deploying

Each `langgraph.*.json` file is its own LangSmith Deployment. On the orchestrator, set
`SUPPORT_CASE_URL`, `CUSTOMER_SUCCESS_URL`, `CUSTOMER_SUCCESS_A2A_URL`, `LANGSMITH_API_KEY`
and `CUSTOMER_SUCCESS_API_KEY`.
