FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .[ai]

EXPOSE 8000

CMD ["uvicorn", "laundromat.report.app:app", "--host", "0.0.0.0", "--port", "8000"]
