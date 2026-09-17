.PHONY: up down logs test eval clean backend frontend native-setup

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
	@echo "Copy .env.example → .env if missing"
	@test -f .env || cp .env.example .env
	@echo "Done. See NATIVE_SETUP.md for full instructions."

backend:
	cd backend && . .venv/bin/activate 2>/dev/null; uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && VITE_API_URL=http://localhost:8000 npm run dev

test:
	cd backend && . .venv/bin/activate 2>/dev/null; pytest -v

eval:
	cd experiments && python evaluate.py

clean:
	rm -rf backend/__pycache__ backend/app/**/__pycache__ backend/.venv
	rm -rf frontend/node_modules frontend/dist
