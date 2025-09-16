import os
from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

# Explicitly include the module that defines tasks
celery = Celery(
    "flexml",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks"],
)

# sane defaults
celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)
