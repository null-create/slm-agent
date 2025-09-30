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

build-agent:
	@echo "Building agent docker image..."
	@cd src/inference
	@docker build -t slm-agent .

build-server:
	@echo "Building MCP Server docker image..."
	@cd src/tools
	@docker build -t slm-mcp-server .

docker-run:
	@docker compose up -d 

docker-stop:
	@docker compose down
