from fastapi import APIRouter
from app.services.predictor import model_status

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok", "service": "steelvision-api"}

@router.get("/model-status")
def get_model_status():
    return model_status()
