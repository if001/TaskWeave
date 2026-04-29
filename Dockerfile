FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy project sources.
COPY src ./src
COPY examples ./examples

RUN uv pip install -e .

# Ensure runtime state directory exists.
RUN mkdir -p /app/.state

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "-m", "examples.discord_bot", "AO"]
