# STAGE 1 - BUILDER
FROM python:3.13-slim as builder
WORKDIR /app

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --no-cache-dir streamlit requests

# STAGE 2 - RUNTIME
FROM python:3.13-slim
WORKDIR /app

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

COPY src/frontend/streamlit_app.py .

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]