from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.predictor import analyze_image, analyze_full_sheet

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp"}
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


async def _read_image(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, or BMP image.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 12 MB.",
        )
    return content


@router.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    content = await _read_image(file)
    try:
        return analyze_image(content, file.filename or "uploaded-image")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inspection failed: {exc}")


@router.post("/inspect/full-sheet")
async def inspect_full_sheet(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.60, ge=0.0, le=1.0),
):
    content = await _read_image(file)
    try:
        return analyze_full_sheet(
            content,
            file.filename or "steel-sheet",
            confidence_threshold=confidence_threshold,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Full-sheet inspection failed: {exc}")
