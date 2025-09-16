import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers_presign import router as presign_router
from .routers_datasets import router as datasets_router
from .routers_tasks import router as tasks_router
from .routers_llm import router as llm_router
from .routers_predict import router as predict_router

app = FastAPI(title="Flex ML (Python-only)")

# CORS for Angular dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/health")
def health():
    return {"ok": True}


app.include_router(presign_router)
app.include_router(datasets_router)
app.include_router(tasks_router)
app.include_router(llm_router)
app.include_router(predict_router)
