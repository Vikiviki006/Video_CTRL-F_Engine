.PHONY: up down logs test eval eval-dry plots migrate frontend-build clean backend frontend native-setup

# Docker (if available)
up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

# Native (no Docker)
native-setup:
	@echo "Creating data directories..."
	mkdir -p data/media data/frames data/clips data/models
	@echo "Copy .env.example -> .env if missing"
	@test -f .env || cp .env.example .env
	@echo "Done. See NATIVE_SETUP.md for full instructions."

backend:
	cd backend && . .venv/bin/activate 2>/dev/null; uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && VITE_API_URL=http://localhost:8000 npm run dev

frontend-build:
	cd frontend && npm run build

test:
	cd backend && . .venv/bin/activate 2>/dev/null; pytest -v

# --- Database schema ---
# init.sql only bootstraps a FRESH database volume. Use this to add the graph
# tables to an existing one.
migrate:
	cd backend && alembic upgrade head

migrate-down:
	cd backend && alembic downgrade -1

# --- Evaluation ---
# Always start with eval-dry: it validates the annotations and reports how many
# queries are scorable, without touching the database or the model.
eval-dry:
	python -m experiments.evaluation.evaluate_retrieval \
		--dataset experiments/datasets/evaluation.csv \
		--dry-run

eval:
	python -m experiments.evaluation.evaluate_retrieval \
		--dataset experiments/datasets/evaluation.csv \
		--model siglip2 \
		--top-k 10 \
		--graph --graph-depth 2 \
		--ablation --depth-sweep --index-metrics \
		--output experiments/results/

plots:
	@for s in recall precision mrr ndcg latency frame_reduction ablation comparison; do \
		echo "--- plot_$$s ---"; \
		python -m experiments.visualization.plot_$$s || exit 1; \
	done

clean:
	rm -rf backend/__pycache__ backend/app/**/__pycache__ backend/.venv
	rm -rf frontend/node_modules frontend/dist
	rm -rf experiments/results/figures
