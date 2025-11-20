.PHONY: clean test set-up build-agent build-server docker-run docker-stop


clean:
	@echo "Cleaning workspace..."
	@rm -rf venv
	@rm -rf .venv

test:
	@echo "Starting tests..."
	@pytest -v -s -n auto tests/

set-up: 
	@echo "Setting up project..."
	@pip install --upgrade pip
	@pip install -r src/inference/requirements.txt
	@pip install -r src/tools/requirements.txt
	@PYTHONPATH=. python scripts/setup.py

get-base-model:
	@echo "Downloading base model..."
	@PYTHONPATH=. python scripts/setup.py

run-agent:
	@echo "Starting agent in container..."
	@docker compose -f docker-compose-agent.yml -d --build

stop-agent:
	@echo "Shutting down agent..."
	@docker compose -f docker-compose-agent.yml down

run-server:
	@echo "Building MCP Server docker image..."
	@docker compose -f docker-compose-mcp.yml -d --build

stop-server:
	@echo "Stopping MCP server..."
	@docker compose -f docker-compose-mcp.yml down