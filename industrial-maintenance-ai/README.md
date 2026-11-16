# AI-Based Industrial Maintenance and Process Optimization System

Full-stack predictive-maintenance starter project:
- React + Vite dashboard
- FastAPI REST API
- Random Forest failure prediction using the AI4I 2020 dataset
- SQLite storage for prediction history
- Optional OpenAI-powered explanation (API key is optional)

## Requirements
- Python 3.10+
- Node.js 18+
- Internet connection for the first dataset download and package installation

## 1. Backend setup (Windows PowerShell)
```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python scripts\download_dataset.py
python scripts\train_model.py
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
API docs: http://127.0.0.1:8000/docs

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Frontend setup (new terminal)
```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```
Open the local URL printed by Vite (normally http://localhost:5173).

## 3. Dataset
The downloader retrieves the public AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository and saves it as `backend/data/ai4i2020.csv`.
Do not rename the target column `Machine failure`. The training script uses the process and sensor features and excludes identifiers/target leakage columns.

## 4. API endpoints
- `GET /health` - service and model status
- `GET /metrics` - model evaluation metrics
- `GET /machines` - sample machine records
- `POST /predict` - predict failure risk for a machine
- `GET /history` - recent saved predictions

Example request:
```json
{
  "type": "M",
  "air_temperature_k": 300.0,
  "process_temperature_k": 310.0,
  "rotational_speed_rpm": 1500,
  "torque_nm": 40.0,
  "tool_wear_min": 120
}
```

## 5. API keys and security
The core predictive-maintenance system does **not** require a paid API key. The `.env.example` includes an optional `OPENAI_API_KEY` placeholder for generating a natural-language explanation. Add your own key only to your local `.env` file. Never commit `.env`, paste keys into source code, or share real keys in ZIP files. The project works without the optional key.

## 6. Important
This is an educational decision-support prototype, not a certified industrial safety controller. Validate it with plant-specific data and domain experts before using it for real equipment decisions.
