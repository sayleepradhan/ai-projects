"""
FastAPI backend for the Analysis Report Agent.

Endpoints:
    GET  /api/health        -- health check
    POST /api/report        -- generate a report (synchronous)
    POST /api/report/async  -- start report generation (returns task_id)
    GET  /api/report/{id}   -- poll async task status
"""

import uuid
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import generate_report, ReportResult


# ---------------------------------------------------------------------------
# In-memory task store (swap for Redis/DB in production)
# ---------------------------------------------------------------------------
tasks: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ReportRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="The research topic for the analysis report.",
        examples=["Overview of AI regulation by country"],
    )


class StepSchema(BaseModel):
    description: str
    result: str
    status: str


class ReportResponse(BaseModel):
    topic: str
    plan: list[StepSchema]
    report: str
    error: str = ""


class AsyncReportResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str  # pending | running | done | error
    current_step: int = 0
    total_steps: int = 0
    report: ReportResponse | None = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(
    title="Analysis Report Agent API",
    version="1.0.0",
    description="Generate structured analysis reports using a Plan-and-Execute AI agent.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/report", response_model=ReportResponse)
def create_report_sync(req: ReportRequest):
    """Generate a report synchronously (blocks until done)."""
    result: ReportResult = generate_report(req.topic)

    if result.error and not result.report:
        raise HTTPException(status_code=500, detail=result.error)

    return ReportResponse(
        topic=result.topic,
        plan=[
            StepSchema(
                description=s.description,
                result=s.result,
                status=s.status,
            )
            for s in result.plan
        ],
        report=result.report,
        error=result.error,
    )


@app.post("/api/report/async", response_model=AsyncReportResponse)
def create_report_async(req: ReportRequest):
    """Start report generation in background, return a task_id to poll."""
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "pending",
        "current_step": 0,
        "total_steps": 0,
        "result": None,
    }

    def run_in_background(tid: str, topic: str):
        tasks[tid]["status"] = "running"

        def on_step(idx, desc, status):
            tasks[tid]["current_step"] = idx + 1

        result = generate_report(topic, on_step=on_step)
        tasks[tid]["result"] = result
        tasks[tid]["total_steps"] = len(result.plan)
        tasks[tid]["current_step"] = len(result.plan)
        tasks[tid]["status"] = "error" if (result.error and not result.report) else "done"

    thread = threading.Thread(target=run_in_background, args=(task_id, req.topic))
    thread.start()

    return AsyncReportResponse(
        task_id=task_id,
        status="pending",
        message="Report generation started.",
    )


@app.get("/api/report/{task_id}", response_model=TaskStatusResponse)
def get_report_status(task_id: str):
    """Poll for the status of an async report task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found.")

    task = tasks[task_id]
    report_response = None

    if task["status"] == "done" and task["result"]:
        r: ReportResult = task["result"]
        report_response = ReportResponse(
            topic=r.topic,
            plan=[
                StepSchema(description=s.description, result=s.result, status=s.status)
                for s in r.plan
            ],
            report=r.report,
            error=r.error,
        )

    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        current_step=task["current_step"],
        total_steps=task["total_steps"],
        report=report_response,
    )