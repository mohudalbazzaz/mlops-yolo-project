# STAGE 1 - BUILDER
FROM python:3.13-slim as BUILDER
WORKDIR /app

COPY requirements.txt .
# the --user flag tells pip to install all packages in /root/.local to copy over
RUN pip install --user --no-cache-dir -r requirements.txt 

# STAGE 2 - RUNTIME
FROM python:3.13-slim
WORKDIR /app

COPY --from=BUILDER /root/.local /root/.local
COPY . . 

# installing packages to /root/.local 
ENV PATH=/root/.local/bin:$PATH

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

