from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc
from .config import FRONTEND_ORIGIN
from .database import Base, engine, get_db
from .models import PredictionRecord
from .schemas import MachineInput
from .ml import get_model, get_metrics, predict
from .config import OPENAI_API_KEY, OPENAI_MODEL

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Industrial Maintenance AI API", version="1.0.0",
              description="Predictive maintenance API for the AI4I 2020 dataset.")
app.add_middleware(
    CORSMiddleware, allow_origins=[FRONTEND_ORIGIN, "http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.on_event("startup")
def startup():
    try:
        get_model()
    except Exception:
        pass

@app.get("/")
def root():
    return {"message": "Industrial Maintenance AI API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": get_model() is not None}

@app.get("/metrics")
def metrics():
    return get_metrics() or {"message": "Train the model to view evaluation metrics."}

@app.get("/machines")
def machines():
    return [
        {"machine_id": "M-001", "status": "Running", "type": "M", "area": "Assembly"},
        {"machine_id": "M-002", "status": "Monitoring", "type": "L", "area": "Packaging"},
        {"machine_id": "M-003", "status": "Running", "type": "H", "area": "Machining"},
        {"machine_id": "M-004", "status": "Inspection due", "type": "M", "area": "Assembly"},
    ]

@app.post("/predict")
def predict_failure(data: MachineInput, db: Session = Depends(get_db)):
    try:
        probability, failed, risk, recommendation = predict(data.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    record = PredictionRecord(
        machine_type=data.type, failure_probability=probability,
        predicted_failure=int(failed), risk_level=risk
    )
    db.add(record)
    db.commit()
    return {
        "failure_probability": round(probability, 4),
        "predicted_failure": failed,
        "risk_level": risk,
        "recommendation": recommendation
    }

@app.get("/history")
def history(limit: int = 20, db: Session = Depends(get_db)):
    rows = db.query(PredictionRecord).order_by(desc(PredictionRecord.created_at)).limit(min(limit, 100)).all()
    return [{
        "id": r.id, "machine_type": r.machine_type,
        "failure_probability": r.failure_probability,
        "predicted_failure": bool(r.predicted_failure),
        "risk_level": r.risk_level,
        "created_at": r.created_at.isoformat()
    } for r in rows]


@app.post("/explain")
def explain_prediction(data: MachineInput):
    """Optional AI explanation; works only when OPENAI_API_KEY is configured."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="Optional explanation is disabled. Add OPENAI_API_KEY to backend/.env.")
    try:
        from openai import OpenAI
        probability, failed, risk, recommendation = predict(data.model_dump())
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Explain predictive-maintenance results in plain language. Do not claim certainty or replace an engineer's inspection."},
                {"role": "user", "content": f"Machine inputs: {data.model_dump()}. Model failure probability: {probability:.1%}. Risk: {risk}. Recommendation: {recommendation}. Explain the result briefly."}
            ],
            max_tokens=180
        )
        return {"explanation": response.choices[0].message.content}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Explanation provider error: {str(exc)}")
