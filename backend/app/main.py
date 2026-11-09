from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import inspection, system

app = FastAPI(
    title="SteelVision AI API",
    description="Steel surface defect inspection powered by ResNet-34 and Grad-CAM",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router, prefix="/api", tags=["System"])
app.include_router(inspection.router, prefix="/api", tags=["Inspection"])

@app.get("/")
def root():
    return {"message": "SteelVision AI API is running", "docs": "/docs"}
