from fastapi import APIRouter
from .models import TaskStatus
from .celery_app import celery

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskStatus)
def task_status(task_id: str):
    a = celery.AsyncResult(task_id)
    result = None;
    error = None
    if a.successful():
        result = a.get()
    elif a.failed():
        error = str(a.result)
    return TaskStatus(task_id=task_id, state=a.state, result=result, error=error)
