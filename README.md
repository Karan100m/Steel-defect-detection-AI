# SteelVision AI 2.0

Full-stack steel surface defect inspection application.

## Architecture
- `frontend/`: React + Vite + Tailwind CSS + Recharts
- `backend/`: FastAPI + PyTorch + ResNet-34 + Grad-CAM
- Existing checkpoint: `backend/steel_defect_model.pth`

## Prerequisites
- Python 3.10 or 3.11 recommended
- Node.js 20 LTS recommended
- Your trained `steel_defect_model.pth` checkpoint

## 1. Add your trained model
Copy the existing checkpoint from your Mac project into:
`backend/steel_defect_model.pth`

Do not rename it unless you also update `MODEL_PATH` in `backend/app/services/predictor.py`.

## 2. Start the backend (Terminal window 1)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Backend: http://127.0.0.1:8000  
Interactive API docs: http://127.0.0.1:8000/docs

## 3. Start the frontend (Terminal window 2)
```bash
cd frontend
npm install
npm run dev
```
Frontend: http://localhost:5173

## Notes
- Keep the backend and frontend running in separate Terminal windows.
- The model is loaded lazily on the first inspection request.
- CPU inference is configured to match the original local app.
- Class order is intentionally preserved from the original app. Verify it matches the exact ImageFolder class order used during training.
- This starter records inspection history in browser memory only. Persistent SQLite history, human verification, batch upload, and PDF reports are later milestones.
- Analytics shown here are session prediction statistics, not accuracy. Accuracy/precision/recall/F1 require verified ground-truth labels.
- Grad-CAM is an approximate explanation, not a precise defect segmentation.
