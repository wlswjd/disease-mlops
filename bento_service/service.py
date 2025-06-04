import bentoml
import pandas as pd
from bentoml.io import JSON

# 모델 로딩
model_ref = bentoml.sklearn.get("disease_model:latest")
model_runner = model_ref.to_runner()

svc = bentoml.Service("disease_service", runners=[model_runner])

input_schema = JSON()
output_schema = JSON()

@svc.api(input=input_schema, output=output_schema)
def predict(input_data: dict) -> dict:
    df = pd.DataFrame([input_data])
    result = model_runner.run(df)
    return {"prediction": result[0]}

