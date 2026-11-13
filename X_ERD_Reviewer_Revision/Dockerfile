FROM python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY requirements.lock ./
RUN python -m pip install --no-cache-dir -r requirements.lock
COPY pyproject.toml ./
COPY src ./src
COPY tests ./tests
COPY config ./config
COPY scripts ./scripts
RUN python -m pip install --no-cache-dir --no-deps .
CMD ["python", "-m", "xerd.cli", "all"]
