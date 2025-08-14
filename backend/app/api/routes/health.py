from fastapi import APIRouter


router = APIRouter(tags=["health"]) 


@router.get("/healthz")
def healthcheck():
    return {"status": "ok"}


