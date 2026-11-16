from pydantic import BaseModel, Field

class MachineInput(BaseModel):
    type: str = Field(default="M", pattern="^[LMH]$")
    air_temperature_k: float = Field(gt=0)
    process_temperature_k: float = Field(gt=0)
    rotational_speed_rpm: float = Field(gt=0)
    torque_nm: float = Field(gt=0)
    tool_wear_min: float = Field(ge=0)

class PredictionOutput(BaseModel):
    failure_probability: float
    predicted_failure: bool
    risk_level: str
    recommendation: str
