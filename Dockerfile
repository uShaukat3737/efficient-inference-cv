FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .

#install dependencies in builder stage
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

#copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

#copy application code
COPY src/ src/
COPY proto/ proto/
COPY models/ models/

#default to fastapi server; can be overridden
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
