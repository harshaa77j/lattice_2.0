# ---------- Stage 1: build the React website ----------
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------- Stage 2: the Python server ----------
FROM python:3.10-slim

# Print logs immediately; don't write .pyc files
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# CPU-only PyTorch first: far smaller than the default GPU build,
# and this slow step stays cached when only your code changes.
RUN pip install --no-cache-dir torch==2.14.1+cpu torchvision==0.29.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from torchvision.datasets import CIFAR10; CIFAR10('/app/data', train=False, download=True)"

COPY . .
# The finished website from stage 1 (only the built files, no Node.js)
COPY --from=frontend /frontend/dist ./frontend/dist

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]