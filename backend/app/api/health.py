from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    """
    Liveness probe for deployments.
    """
    return {"status": "ok"}
