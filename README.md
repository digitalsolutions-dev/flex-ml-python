# Flex ML (Python-only, Azure LLM)

FastAPI + Celery + Redis, Polars/DuckDB I/O, S3 presigned uploads, MariaDB metadata, and Azure OpenAI tool-calling.

Python version: 3.9

## Run
```bash
cp .env.example .env
# Initialize DB:
# mysql -h $DB_HOST -P ${DB_PORT:-3306} -u $DB_USER -p$DB_PASSWORD $DB_NAME < sql/init.sql
docker compose up
