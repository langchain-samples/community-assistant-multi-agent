.PHONY: dev-all dev-orchestrator dev-support-case dev-customer-success notebook

dev-orchestrator:
	uv run langgraph dev --config langgraph.orchestrator.json --port 2024 --no-browser

dev-support-case:
	uv run langgraph dev --config langgraph.support_case.json --port 2025 --no-browser

dev-customer-success:
	uv run langgraph dev --config langgraph.customer_success.json --port 2026 --no-browser

dev-all:
	@trap 'kill 0' INT TERM EXIT; \
	uv run langgraph dev --config langgraph.support_case.json --port 2025 --no-browser & \
	uv run langgraph dev --config langgraph.customer_success.json --port 2026 --no-browser & \
	uv run langgraph dev --config langgraph.orchestrator.json --port 2024 --no-browser & \
	wait

# Re-run demo.ipynb top to bottom against the live servers and save the outputs.
notebook:
	uv run --with nbconvert==7.17.0 jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 demo.ipynb
