# API reference

Base URL: `http://127.0.0.1:8000`

## POST /predict
Request body:
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
Returns `failure_probability` (0–1), `predicted_failure`, `risk_level`, and `recommendation`.

Interactive OpenAPI documentation is available at `/docs`.
