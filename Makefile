.PHONY: up down logs data train api lint test

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

data:
	docker compose run --rm trainer python -m trainer.pipelines.download_movielens

train:
	docker compose run --rm trainer python -m trainer.pipelines.run_pipeline

api:
	docker compose up -d api

lint:
	docker compose run --rm trainer python -m compileall services/trainer/src
	docker compose run --rm api python -m compileall services/api/src

test:
	docker compose run --rm trainer python -m compileall services/trainer/src
	docker compose run --rm api python -m compileall services/api/src
