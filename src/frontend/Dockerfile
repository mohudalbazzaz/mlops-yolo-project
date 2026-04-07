# STAGE 1 - BUILDER
FROM python:3.13-slim as BUILDER
WORKDIR /app

RUN pip install --user --no-cache-dir streamlit requests

# STAGE 2 - RUNTIME
FROM python:3.13-slim
WORKDIR /app

COPY --from=BUILDER /root/.local /root/.local
COPY src/frontend /app/src/frontend

# installing packages to /root/.local 
ENV PATH=/root/.local/bin:$PATH

CMD ["streamlit", "run", "src/frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]