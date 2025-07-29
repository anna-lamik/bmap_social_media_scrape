# Development Setup

## Backend
```bash
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

## Frontend
```bash
cd frontend
npm install
npm run dev
```

### Run Tests
```bash
# Backend
pytest
# Frontend unit tests
npm test
# Frontend e2e
npm run e2e
```

## Docker Compose
```yaml
docker-compose up --build
```

Create `.env` with:
```
OPENAI_API_KEY=your-key
VITE_API_URL=http://localhost:8000
```

Deploy on Render/Vercel by linking repo and using above commands.

## Setup Script

For convenience, you can create the environment and install all analysis
dependencies using `setup_env.sh`:

```bash
bash setup_env.sh
```
