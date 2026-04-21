.PHONY: data train eval serve demo docker-build test lint format ci merge

data:
	python -m src.data.prepare && python -m src.data.validate

train:
	python -m src.train.train --config configs/llama32_qlora.yaml

merge:
	python -m src.train.merge --adapter models/checkpoints/best --out models/merged

eval:
	python -m src.eval.benchmark

serve:
	uvicorn src.serve.api:app --host 0.0.0.0 --port 8000 --reload

demo:
	python -m src.serve.app

docker-build:
	docker build -f docker/Dockerfile -t legal-llm:local .

test:
	pytest tests/

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

ci: lint test
