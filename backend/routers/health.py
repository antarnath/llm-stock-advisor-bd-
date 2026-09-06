"""GET /healthz — minimal liveness probe."""
from fastapi import APIRouter

from backend.models import HealthOut

router = APIRouter()


@router.get("/healthz", response_model=HealthOut)
def healthz() -> HealthOut:
    """Always returns ok. We don't probe downstream deps in v1."""
    return HealthOut(status="ok")